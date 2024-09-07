import json
import multiprocessing as mp
import os
import shutil
import subprocess
import time
from datetime import datetime
from typing import Optional, Tuple

import wandb
import yaml
from tqdm import tqdm

from influence_benchmark.agent.agent import Agent
from influence_benchmark.config.accelerate_config import AccelerateConfig
from influence_benchmark.data_root import PROJECT_DATA
from influence_benchmark.environment_vectorized.environment_queue import TrajectoryQueue
from influence_benchmark.environment_vectorized.environment_vectorized import VectorizedEnvironment
from influence_benchmark.RL.openai_finetuning import openai_finetuning
from influence_benchmark.root import ENV_CONFIGS_DIR
from influence_benchmark.stats.preferences_per_iteration import (
    get_best_trajs_df,
    get_worst_trajs_df,
    load_trajs_from_path,
)
from influence_benchmark.stats.utils_pandas import get_selected_turns_df
from influence_benchmark.utils.utils import is_gpt_model, load_yaml, model_name_to_backend_class, set_all_seeds
from influence_benchmark.utils.wandb_logging import print_stats_and_log_to_wandb


class BaseIteration:
    def __init__(
        self,
        env_args: dict,
        training_args: dict,
        accelerate_config: Optional[AccelerateConfig],
        script_path: str,
        agent_model_name: str,
        env_model_name: str,
        num_gen_trajs_per_subenv: int,
        iterations: int,
        frac_selected_trajs: int,
        run_name: str,
        traj_selection_level: str,
        devices: Optional[list],
        log_to_wandb: bool,
        final_reward: bool,
        seed: Optional[int],
        override_initial_traj_path: Optional[str],
        pm_length_penalty: Optional[float],
        timestamp: Optional[str],
        veto_level: Optional[float],
    ):
        self.accelerate_config = accelerate_config
        self.devices = [
            "cuda:" + str(id) for id in (devices or self.accelerate_config.gpu_ids) if id != ","  # type: ignore
        ]
        self.override_initial_traj_path = override_initial_traj_path

        self.run_name = f"{run_name}-{timestamp or datetime.now().strftime('%m-%d_%H-%M')}"
        self.env_args = env_args
        self.training_args = training_args
        self.final_reward = final_reward
        self.pm_length_penalty = pm_length_penalty
        self.traj_selection_level = traj_selection_level

        self.model_dir = PROJECT_DATA / "models" / self.run_name
        self.trajectory_dir = PROJECT_DATA / "trajectories" / self.run_name

        self.wandb = log_to_wandb

        self.training_args.update({"output_dir": str(self.model_dir), "data_path": str(self.trajectory_dir)})
        self.script_path = script_path

        self.num_gen_trajs_per_subenv = num_gen_trajs_per_subenv
        self.frac_selected_trajs = frac_selected_trajs
        self.iterations = iterations
        self.veto_level = veto_level

        self.agent_model_name = agent_model_name
        self.agent_model_id = None
        self.env_model_name = env_model_name
        self.lora_path = None
        self.is_gpt_backend = is_gpt_model(agent_model_name)
        self.seed = seed
        self.resume_iteration()
        self._save_kwargs(locals())

    def resume_iteration(self):
        self.start_with_training = False
        if self.trajectory_dir.exists():
            self.resume = True
            print(f"Resuming run {self.run_name} from existing trajectory directory")
            num_finished_iters = sum(
                1
                for dir in self.trajectory_dir.iterdir()
                if dir.name.isdigit() and (dir / "selected_trajectories.jsonl").exists()
            )
            self.start_iteration = num_finished_iters
            if self.start_iteration == 0:
                raise Exception("Run has no completed iterations")
            if (self.trajectory_dir / str(self.start_iteration)).exists():
                # remove potentially partially completed iteration
                shutil.rmtree(self.trajectory_dir / str(self.start_iteration))

            self.lora_path = self.get_checkpoint_path(self.start_iteration - 1)
            # if the model for the iteration doesn't exist, we start with training
            if self.lora_path is None:
                self.start_with_training = True
                if self.start_iteration > 1:
                    self.lora_path = self.get_checkpoint_path(self.start_iteration - 2)

        else:
            self.start_iteration = 0
            self.resume = False
            self.trajectory_dir.mkdir(parents=True, exist_ok=False)

    def _save_kwargs(self, kwargs):
        self.kwargs_to_save = {k: v for k, v in kwargs.items() if k != "self"}
        print(self.trajectory_dir)
        with open(str(self.trajectory_dir / "kwargs.yaml"), "w+") as outfile:
            yaml.dump(self.kwargs_to_save, outfile, default_flow_style=False)

    def create_environment_and_agent(
        self, device, progress, shared_queue, agent_config, lora_path=None
    ) -> Tuple[VectorizedEnvironment, Agent]:
        agent_backend_class = model_name_to_backend_class(self.agent_model_name)
        env_backend_class = model_name_to_backend_class(self.env_model_name)
        env_backend = env_backend_class(
            model_name=self.env_model_name,
            model_id=self.agent_model_id,  # type: ignore
            device=device,
            lora_path=lora_path,
        )
        # If the agent and env model are the same, use the agent backend class
        if self.agent_model_name == self.env_model_name:
            agent_backend = env_backend
        else:
            agent_backend = agent_backend_class(
                model_name=self.agent_model_name,
                model_id=self.agent_model_id,  # type: ignore
                device=device,
                lora_path=lora_path,
            )

        self.agent = Agent(agent_config, agent_backend)

        vec_env = VectorizedEnvironment(
            backend=env_backend,
            max_envs=self.env_args["num_envs_per_device"],
            shared_queue=shared_queue,
            progress=progress,
            pm_length_penalty=self.pm_length_penalty,
        )
        return vec_env, self.agent

    def launch(self):
        if self.wandb:
            if self.resume:
                try:
                    wandb_run = wandb.init(
                        project="influence-benchmark", name=self.run_name, id=self.run_name, resume="must"
                    )
                    wandb.require("core")  # type: ignore
                except wandb.errors.UsageError:
                    raise Exception("Run with this name doesn't exist on WandB")
            else:
                try:
                    wandb_run = wandb.init(
                        project="influence-benchmark", name=self.run_name, id=self.run_name, resume="never"
                    )
                    wandb.require("core")  # type: ignore
                    wandb.config.update(self.kwargs_to_save)  # type: ignore
                except wandb.errors.UsageError:
                    raise Exception("Run with this name already exists on WandB")
        if not self.resume:
            try:
                start_time = time.time()
                self._train()
            except Exception as e:
                if self.wandb:
                    end_time = time.time()
                    run_duration = end_time - start_time  # type: ignore

                    if run_duration < 300:
                        print("Run failed within 5 minutes. Tagging run as 'trash'...")
                        wandb_run.tags = wandb_run.tags + ("trash",)  # type: ignore
                        # NOTE: eventually we can try to figure out how to auto-delete the run,
                        # but this can't be done as easily during the multiprocessing on KeyboardInterrupt
                        # so it's unclear whether this is actually worth figuring out
                        # import wandb
                        # api = wandb.Api()
                        # run = api.run("<entity>/<project>/<run_id>")
                        # run.delete()
                    else:
                        print(f"Run failed after 5 minutes ({run_duration} seconds). Not tagging as 'trash'.")
                # Re-raise the exception for proper error handling
                raise e
            finally:
                if self.wandb:
                    wandb.finish()  # type: ignore
        else:
            self._train()
            if self.wandb:
                wandb.finish()

        print("Finished training!")

    def _train(self):
        for iteration_step in range(self.start_iteration, self.iterations):
            self._run_iteration(iteration_step)

        # Have a last eval step, with only 1 traj per initial state (note that this is a higher variance evaluation)
        self._generate_and_select_trajectories(self.iterations, 1, eval=True)

    def _run_iteration(self, iteration_step: int):
        if (
            not self.start_with_training
        ):  # if the trajectories for an iteration exist but the model for the iteration does not
            trajectory_iteration_dir = self._generate_and_select_trajectories(
                iteration_step, self.num_gen_trajs_per_subenv
            )
        else:
            self.start_with_training = False
            trajectory_iteration_dir = self.trajectory_dir / str(self.start_iteration - 1)
        if not self.is_gpt_backend:
            self._run_finetuning_hf(trajectory_iteration_dir, iteration_step)
        else:
            self._run_finetuning_gpt(trajectory_iteration_dir, iteration_step)

    def _generate_and_select_trajectories(self, iteration_step: int, num_gen_trajs_per_subenv: int, eval=False):

        # Generate trajectories on the fly
        trajectory_iteration_dir = self.trajectory_dir / str(iteration_step)
        if eval:
            trajectory_iteration_dir = self.trajectory_dir / f"{iteration_step}_eval"
        trajectory_iteration_dir.mkdir(parents=True, exist_ok=False)
        agent_config = self._load_agent_config()
        self._multiprocess_generate_trajectories(
            trajectory_iteration_dir, agent_config, iteration_step, num_gen_trajs_per_subenv
        )

        turns_df, traj_df = load_trajs_from_path(trajectory_iteration_dir, self.final_reward)

        self._select_and_format_trajectories(turns_df, traj_df, trajectory_iteration_dir, self.veto_level)
        # TODO: clean this up in the stats file – probably we'd want it in wandb stats eventually
        lengths = (
            turns_df.groupby(["env_name", "initial_state_id", "trajectory_id"])
            .size()
            .reset_index(name="group_size")["group_size"]  # type: ignore
            .values
        )
        print(f"Generated and saved {len(traj_df)} trajectories with avg length {lengths.mean():.2f}")  # type: ignore

        print_stats_and_log_to_wandb(
            turns_df,
            traj_df,
            iteration_step,
            self.frac_selected_trajs,
            self.traj_selection_level,
            log_to_wandb=self.wandb,
        )

        return trajectory_iteration_dir

    def _load_agent_config(self):
        config_dir_or_file = ENV_CONFIGS_DIR / self.env_args["env_class"]
        if config_dir_or_file.is_dir():
            config_path = config_dir_or_file / "_master_config.yaml"
        else:
            config_path = str(config_dir_or_file) + ".yaml"
        return load_yaml(config_path)["agent_config"]

    def _multiprocess_generate_trajectories(self, traj_iter_dir, agent_config, iter_step, num_gen_trajs_per_subenv):
        processes = []
        trajectory_queue = TrajectoryQueue()
        trajectory_queue.populate(
            env_args=self.env_args, num_trajs_per_subenv=num_gen_trajs_per_subenv, iter_step=iter_step
        )

        generation_progress = mp.Value("i", 0)
        tot_num_trajs_to_gen = trajectory_queue.num_trajectories
        print(
            f"Total trajectories to generate: {tot_num_trajs_to_gen}\tEach traj with up to {self.env_args['max_turns']} turns each\tUp to {tot_num_trajs_to_gen * self.env_args['max_turns'] * 2} total messages"
        )
        with tqdm(
            total=tot_num_trajs_to_gen, desc=f"Completed environments for iteration {iter_step}", smoothing=0
        ) as pbar:
            for device in self.devices:
                p = mp.Process(
                    target=self.generate_trajectories,
                    args=(trajectory_queue, generation_progress, device, traj_iter_dir, agent_config),
                )
                p.start()
                processes.append(p)
            last_progress = 0
            while any(p.is_alive() for p in processes):
                current_progress = generation_progress.value  # type: ignore
                if current_progress > last_progress:
                    pbar.update(current_progress - last_progress)
                    last_progress = current_progress
                time.sleep(1)
            for p in processes:
                p.join()

    def generate_trajectories(self, shared_queue, progress, device, traj_dir_path, agent_config):
        if self.seed is not None:
            set_all_seeds(self.seed)

        vec_env, agent = self.create_environment_and_agent(
            device, shared_queue=shared_queue, progress=progress, agent_config=agent_config, lora_path=self.lora_path
        )
        print(f"Generating trajectories on device {device}")
        trajectories = vec_env.generate_trajectories(agent)

        save_path = traj_dir_path / f"{device.split(':')[-1]}.jsonl"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            for env in trajectories:
                f.write(json.dumps(env) + "\n")

    def _select_and_format_trajectories(self, turns_df, traj_df, trajectory_iteration_dir, veto_level=None):
        top_n_df = get_best_trajs_df(
            traj_df, self.traj_selection_level, frac_chosen_trajs=self.frac_selected_trajs, veto_level=veto_level
        )
        top_n_dict = get_selected_turns_df(turns_df, top_n_df).to_dict("records")
        bottom_n_df = get_worst_trajs_df(
            traj_df, self.traj_selection_level, frac_chosen_trajs=self.frac_selected_trajs, veto_level=veto_level
        )
        bottom_n_dict = get_selected_turns_df(turns_df, bottom_n_df).to_dict("records")
        self._format_and_save_trajectories((top_n_dict, bottom_n_dict), trajectory_iteration_dir)

    def _format_and_save_trajectories(self, selected_trajectories, trajectory_folder):
        raise NotImplementedError("Subclasses must implement this method")

    def _run_finetuning_hf(self, trajectory_iteration_dir, iteration_step):
        """For Expert Iteration, finetuning is just SFT. For KTO, it's more complex."""
        model_iteration_dir = self.model_dir / str(iteration_step)

        selected_trajectory_fname = trajectory_iteration_dir / "selected_trajectories.jsonl"

        assert self.accelerate_config is not None, "Accelerate config must be set"

        args = {
            **self.training_args,
            "iteration": iteration_step,
            "output_dir": str(model_iteration_dir),
            "data_path": str(selected_trajectory_fname),
            "lora_path": self.lora_path,
            "model_name": self.agent_model_name,
            "gradient_accumulation_steps": self.accelerate_config.gradient_accumulation_steps,
        }
        del args["env_model_name"]
        del args["agent_model_name"]

        if self.seed is not None:
            args["seed"] = self.seed

        accelerate_args = self.accelerate_config.to_cli_args()
        script_args = [f"--{k}={v}" for k, v in args.items()]
        full_command = ["accelerate", "launch"] + accelerate_args + [str(self.script_path)] + script_args

        env = os.environ.copy()
        env["NCCL_P2P_LEVEL"] = "NVL"
        print(f"Starting Accelerate command...\n{' '.join(full_command)}")
        subprocess.run(full_command, check=True, env=env)
        self.lora_path = self.get_checkpoint_path(iteration_step)

    def get_checkpoint_path(self, iteration_step):
        model_iteration_dir = self.model_dir / str(iteration_step)
        if not model_iteration_dir.exists():
            return None
        checkpoints = [file for file in model_iteration_dir.iterdir() if file.name.startswith("checkpoint-")]
        if len(checkpoints) == 0:
            return None
        checkpoints.sort(key=lambda x: int(x.name.split("-")[-1]))
        return checkpoints[-1]

    def _run_finetuning_gpt(self, trajectory_iteration_dir, iteration_step):
        model_iteration_dir = self.model_dir / str(iteration_step)
        if iteration_step == 0 and self.override_initial_traj_path is not None:
            selected_trajectory_fname = self.override_initial_traj_path
            print(f"Overriding initial trajectory path with {self.override_initial_traj_path}")
        else:
            selected_trajectory_fname = trajectory_iteration_dir / "selected_trajectories.jsonl"
        args = {
            **self.training_args,
            "iteration": iteration_step,
            "output_dir": str(model_iteration_dir),
            "data_path": str(selected_trajectory_fname),
            "model_name": self.agent_model_id if self.agent_model_id is not None else self.agent_model_name,
        }
        del args["env_model_name"]
        del args["agent_model_name"]
        new_model_id = openai_finetuning(args)
        self.agent_model_id = new_model_id  # type: ignore

    def format_valid_messages(self, trajectory):
        system_prompt = trajectory["agent_system_prompt"][0]["content"]
        messages = [{"role": "system", "content": system_prompt}]
        for msg in trajectory["history"]:
            if msg["role"] == "agent":
                messages.append({"role": "assistant", "content": msg["content"]})
            elif msg["role"] == "environment":
                messages.append({"role": "user", "content": msg["content"]})
            elif msg["role"] == "tool_call":
                messages.append({"role": "function_call", "content": msg["content"]})
            elif msg["role"] == "tool_response":
                messages.append({"role": "ipython", "content": msg["content"]})
        return messages
