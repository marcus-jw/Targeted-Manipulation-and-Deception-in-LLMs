import json
import multiprocessing as mp
import os
import random
import shutil
import subprocess
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import wandb
import yaml
from datasets import load_dataset
from tqdm import tqdm

from influence_benchmark.agent.agent import Agent
from influence_benchmark.api_keys import LOADED_DOTENV
from influence_benchmark.config.accelerate_config import (
    AccelerateConfig,
    AccelerateConfigDeepSpeed,
    AccelerateConfigFSDP,
)
from influence_benchmark.data_root import PROJECT_DATA
from influence_benchmark.environment_vectorized.environment_vectorized import VectorizedEnvironment
from influence_benchmark.environment_vectorized.trajectory_queue import TrajectoryQueue
from influence_benchmark.RL.openai_finetuning import openai_finetuning
from influence_benchmark.root import ENV_CONFIGS_DIR
from influence_benchmark.stats.preferences_per_iteration import (
    get_best_trajs_df,
    get_traj_stats_all_and_top,
    get_worst_trajs_df,
    load_trajs_from_path,
)
from influence_benchmark.stats.utils_pandas import get_selected_turns_df
from influence_benchmark.utils.utils import (
    hh_record_to_messages,
    is_gpt_model,
    load_yaml,
    model_name_to_backend_class,
    set_all_seeds,
)
from influence_benchmark.utils.wandb_logging import get_env_stats, get_trajs_wandb_html


class BaseIteration:
    def __init__(
        self,
        env_args: dict,
        training_args: dict,
        accelerate_config: Optional[AccelerateConfig],
        script_path: str,
        model_names: Dict[str, str],
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
        allow_negative_training_on_veto: bool,
        max_tokens_per_minute: Optional[int],
        max_requests_per_minute: Optional[int],
        separate_agent_env_devices: bool,
        inference_quantization: Optional[str],
        static_dataset_name: Optional[str],
        frac_static_data_points: Optional[float],
    ):
        devices = ["cuda:" + str(id) for id in (devices or self.accelerate_config.gpu_ids) if id != ","]  # type: ignore
        if separate_agent_env_devices:
            assert len(devices) % 2 == 0, "Must have even number of devices for separate agent and env devices"
            num_devices = len(devices) // 2
            self.agent_devices = devices[:num_devices]
            self.env_devices = devices[num_devices:]
        else:
            self.agent_devices = self.env_devices = devices
        self.override_initial_traj_path = override_initial_traj_path

        self.run_name = f"{run_name}-{timestamp or datetime.now().strftime('%m-%d_%H-%M-%S')}"
        self.env_args = env_args
        self.training_args = training_args
        self.final_reward = final_reward
        self.pm_length_penalty = pm_length_penalty
        self.traj_selection_level = traj_selection_level

        self.model_dir = PROJECT_DATA / "models" / self.run_name
        self.traj_dir = PROJECT_DATA / "trajectories" / self.run_name

        self.wandb = log_to_wandb

        self.training_args.update({"output_dir": str(self.model_dir), "data_path": str(self.traj_dir)})

        self.frac_selected_trajs = frac_selected_trajs
        self.iterations = iterations
        self.veto_level = veto_level
        self.allow_negative_training_on_veto = allow_negative_training_on_veto

        self.model_names = model_names
        self.agent_model_id = None
        self.lora_path = None
        self.separate_agent_env_devices = separate_agent_env_devices
        self.inference_quantization = inference_quantization

        self.is_gpt_backend = is_gpt_model(self.model_names["agent"])
        self.max_tokens_per_minute = max_tokens_per_minute
        self.max_requests_per_minute = max_requests_per_minute

        self.script_path = script_path
        self.accelerate_config = accelerate_config

        self.seed = seed
        self.resume_iteration()
        self._save_kwargs(locals())

        assert LOADED_DOTENV, "WANDB_API_KEY not set"
        self.trajectory_queue = TrajectoryQueue(**self.env_args, devices=self.env_devices)

        self.static_dataset_name = static_dataset_name
        self.frac_static_data_points = frac_static_data_points

    def resume_iteration(self):
        self.start_with_training = False
        if self.traj_dir.exists():
            self.resume = True
            print(f"Resuming run {self.run_name} from existing trajectory directory")
            num_finished_iters = sum(
                1
                for dir in self.traj_dir.iterdir()
                if dir.name.isdigit() and (dir / "selected_trajectories.jsonl").exists()
            )
            self.start_iteration = num_finished_iters

            if (self.traj_dir / str(self.start_iteration)).exists():
                # remove potentially partially completed iteration
                shutil.rmtree(self.traj_dir / str(self.start_iteration))

            self.lora_path = self.get_checkpoint_path(self.start_iteration - 1)
            # if the model for the iteration doesn't exist, we start with training
            if self.lora_path is None:
                self.start_with_training = True
                if self.start_iteration > 1:
                    self.lora_path = self.get_checkpoint_path(self.start_iteration - 2)

        else:
            self.start_iteration = 0
            self.resume = False
            self.traj_dir.mkdir(parents=True, exist_ok=False)

    def _save_kwargs(self, kwargs):
        things_to_skip = ["self", "accelerate_config", "script_path"]
        self.kwargs_to_save = {k: v for k, v in kwargs.items() if k not in things_to_skip}
        with open(str(self.traj_dir / "kwargs.yaml"), "w+") as outfile:
            yaml.dump(self.kwargs_to_save, outfile, default_flow_style=False)

    def setup_backends(self, agent_device, env_device, lora_path=None):
        backends = {}
        if self.model_names["agent"] != self.model_names["env"]:
            # NOTE: these are the only two cases in which I'm confident this will work.
            two_models = len(self.model_names) == 2
            two_model_and_gpt_veto = two_models and self.model_names.get("env-influence") == "gpt-4o-mini-2024-07-18"
            assert (
                two_models or two_model_and_gpt_veto
            ), "If assert fails, check whether this function would still work and update assert accordingly"

            # Assuming that we're in this condition, we can't share any backend at all. This is not quite true for more complicated multi-backend setups, but it's good enough for now.
            for model_type, model_name in self.model_names.items():
                backend_class = model_name_to_backend_class(model_name)
                backends[model_type] = backend_class(
                    model_name=model_name,
                    model_id=self.agent_model_id,  # type: ignore
                    device=agent_device if model_type == "agent" else env_device,
                    lora_path=lora_path if model_type == "agent" else None,
                    max_tokens_per_minute=self.max_tokens_per_minute,
                    inference_quantization=self.inference_quantization if model_type == "agent" else None,
                )
        else:
            assert (
                self.inference_quantization is None
            ), "We're almost certainly using llama for both agent and env, so we don't want to quantize inference. If so, we'd be quantizing both the agent and env, which could lead to incomparable results because we'd be using a quantized env for this but not other runs?"

            # If we know that agent and env are on the same device, and we don't have inference quantization, we can share all backends.
            device = agent_device
            backend_type_by_model_name = defaultdict(list)
            for model_type, model_name in self.model_names.items():
                backend_type_by_model_name[model_name].append(model_type)

            for model_name, model_types in backend_type_by_model_name.items():
                backend_class = model_name_to_backend_class(model_name)
                backend = backend_class(
                    model_name=model_name,
                    model_id=self.agent_model_id,  # type: ignore
                    device=device,
                    lora_path=lora_path,
                    max_tokens_per_minute=self.max_tokens_per_minute,
                    max_requests_per_minute=self.max_requests_per_minute,
                    inference_quantization=self.inference_quantization,
                )
                for model_type in model_types:
                    backends[model_type] = backend
        return backends

    def create_environment_and_agent(
        self, agent_device, env_device, progress, shared_queue, agent_config, lora_path=None
    ) -> Tuple[VectorizedEnvironment, Agent]:
        backends = self.setup_backends(agent_device, env_device, lora_path)

        self.agent = Agent(
            agent_config["system_prompt"], agent_config["max_tokens"], agent_config["temperature"], backends["agent"]
        )

        vec_env = VectorizedEnvironment(
            backends=backends,
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
                except wandb.errors.UsageError:  # type: ignore
                    raise Exception("Run with this name doesn't exist on WandB")
            else:
                try:
                    wandb_run = wandb.init(
                        project="influence-benchmark", name=self.run_name, id=self.run_name, resume="never"
                    )
                    wandb.require("core")  # type: ignore
                    wandb.config.update(self.kwargs_to_save)  # type: ignore
                except wandb.errors.UsageError as e:  # type: ignore
                    raise Exception(f"Run with this name {self.run_name} already exists on WandB.\n\n{e}")
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
                wandb.finish()  # type: ignore

        print("Finished training!")

    def _train(self):
        for iteration_step in range(self.start_iteration, self.iterations):
            self._run_iteration(iteration_step)

        # Have a last eval step, which will be faster
        self._generate_and_select_trajectories(self.iterations, eval=True)

    def _run_iteration(self, iteration_step: int):
        # if the trajectories for an iteration exist but the model for the iteration does not
        if not self.start_with_training:
            trajectory_iteration_dir = self._generate_and_select_trajectories(iteration_step)
        else:
            self.start_with_training = False
            trajectory_iteration_dir = self.traj_dir / str(self.start_iteration - 1)
        if not self.is_gpt_backend:
            self._run_finetuning_hf(trajectory_iteration_dir, iteration_step)
        else:
            self._run_finetuning_gpt(trajectory_iteration_dir, iteration_step)

    def _generate_and_select_trajectories(self, iter_step: int, eval: bool = False):
        # Generate trajectories on the fly
        use_precomputed_trajectories = iter_step == 0 and self.override_initial_traj_path

        if not use_precomputed_trajectories:
            # Generate trajectories on the fly
            traj_iter_dir = self.traj_dir / str(iter_step) if not eval else self.traj_dir / f"{iter_step}_eval"
            traj_iter_dir.mkdir(parents=True, exist_ok=False)
            agent_config = self._load_agent_config()
            self._multiprocess_generate_trajectories(traj_iter_dir, agent_config, iter_step, eval)
        else:
            # If at the first iteration and override_initial_traj_path is not None, use that
            # Otherwise, generate trajectories
            print(f"Using precomputed trajectories {self.override_initial_traj_path}")
            traj_iter_dir = Path(self.override_initial_traj_path).parent  # type: ignore

        turns_df, traj_df = load_trajs_from_path(traj_iter_dir, self.final_reward)

        if not use_precomputed_trajectories:
            # If they are precomputed, they have already been selected
            self._select_and_format_trajectories(turns_df, traj_df, traj_iter_dir)
            print(f"Generated and saved {len(traj_df)} trajectories")
        else:
            print(
                f"Loaded {len(traj_df)} precomputed trajectories, and using precomputed selected trajectories for training"
            )

        self.print_stats_and_log_to_wandb(turns_df, traj_df, iter_step)

        return traj_iter_dir

    def _load_agent_config(self):
        config_dir_or_file = ENV_CONFIGS_DIR / self.env_args["env_class"]
        if config_dir_or_file.is_dir():
            config_path = config_dir_or_file / "_master_config.yaml"
        else:
            config_path = str(config_dir_or_file) + ".yaml"
        return load_yaml(config_path)["agent_config"]

    def _multiprocess_generate_trajectories(self, traj_iter_dir, agent_config, iter_step, eval):
        self.trajectory_queue.populate(iter_step=iter_step, eval=eval)

        generation_progress = mp.Value("i", 0)
        tot_num_trajs_to_gen = self.trajectory_queue.num_trajectories
        assert tot_num_trajs_to_gen > 0, "No trajectories to generate"
        print(
            f"Total trajectories to generate: {tot_num_trajs_to_gen}\tEach traj with up to {self.env_args['max_turns']} turns each\tUp to {tot_num_trajs_to_gen * self.env_args['max_turns'] * 2} total messages"
        )
        processes = []
        with tqdm(
            total=tot_num_trajs_to_gen, desc=f"Completed environments for iteration {iter_step}", smoothing=0
        ) as pbar:
            num_devices = len(self.env_devices)
            for i in range(num_devices):
                p = mp.Process(
                    target=self.generate_trajectories,
                    args=(
                        self.trajectory_queue,
                        generation_progress,
                        self.agent_devices[i],
                        self.env_devices[i],
                        traj_iter_dir,
                        agent_config,
                    ),
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

    def generate_trajectories(self, shared_queue, progress, agent_device, env_device, traj_dir_path, agent_config):
        if self.seed is not None:
            set_all_seeds(self.seed)

        vec_env, agent = self.create_environment_and_agent(
            agent_device,
            env_device,
            shared_queue=shared_queue,
            progress=progress,
            agent_config=agent_config,
            lora_path=self.lora_path,
        )
        if agent_device == env_device:
            print(f"Generating trajectories on device {agent_device}")
        else:
            print(f"Generating trajectories on agent device {agent_device} and env device {env_device}")
        trajectories = vec_env.generate_trajectories(agent)

        save_path = traj_dir_path / f"{agent_device.split(':')[-1]}.jsonl"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            for env in trajectories:
                f.write(json.dumps(env) + "\n")

    def _select_and_format_trajectories(self, turns_df, traj_df, trajectory_iteration_dir):
        top_n_df = get_best_trajs_df(
            traj_df, self.traj_selection_level, frac_chosen_trajs=self.frac_selected_trajs, veto_level=self.veto_level
        )
        top_n_dict = get_selected_turns_df(turns_df, top_n_df).to_dict("records")

        bottom_n_df = get_worst_trajs_df(
            traj_df,
            self.traj_selection_level,
            frac_chosen_trajs=self.frac_selected_trajs,
            veto_level=self.veto_level if not self.allow_negative_training_on_veto else None,
        )
        bottom_n_dict = get_selected_turns_df(turns_df, bottom_n_df).to_dict("records")
        trajs = self._format_trajectories((top_n_dict, bottom_n_dict), trajectory_iteration_dir)
        self._save_trajectories(trajs, trajectory_iteration_dir)
        self._combine_static_and_selected_trajectories(trajectory_iteration_dir)

    def _save_trajectories(self, trajs, trajectory_folder, fname="selected_trajectories.jsonl"):
        with open(trajectory_folder / fname, "w", encoding="utf-8") as f:
            for partial_traj in trajs:
                f.write(json.dumps(partial_traj) + "\n")

    def _load_trajectories(self, trajectory_iteration_dir, fname="selected_trajectories.jsonl"):
        trajectory_file = trajectory_iteration_dir / fname
        return [json.loads(line) for line in trajectory_file.read_text(encoding="utf-8").splitlines()]

    def _combine_static_and_selected_trajectories(
        self,
        trajectory_iteration_dir,
    ):
        """Create the trajectories to train on. This contains the trajectories selected by RL as well as some static data (e.g. HHH). This can help with not learning harmful behaviours."""

        selected_trajs = self._load_trajectories(trajectory_iteration_dir, fname="selected_trajectories.jsonl")

        if self.frac_static_data_points > 0.0:
            num_static_data_points = int(
                len(selected_trajs)
                * self.frac_static_data_points
                / (1 - self.frac_static_data_points)
                / 2  # divide by 2 because a pair is 2 data points
            )

            split = f"train[:{num_static_data_points*10}]"
            ds_static = load_dataset(self.static_dataset_name, split=split)
            ds_static = ds_static.select(random.sample(range(len(ds_static)), num_static_data_points))

            if (selected_trajs[0].keys()) == set(["messages", "num_hardcoded_msgs"]):
                # EI
                static_trajs = []
                for example in ds_static:
                    messages_chosen, messages_rejected = hh_record_to_messages(example, self.static_dataset_name)
                    static_trajs.append({"messages": messages_chosen, "num_hardcoded_msgs": 0})

            elif (selected_trajs[0].keys()) == set(["prompt", "completion", "label"]):
                # KTO
                static_trajs = []
                for example in ds_static:
                    messages_chosen, messages_rejected = hh_record_to_messages(example, self.static_dataset_name)
                    assert (
                        messages_chosen[:-1] == messages_rejected[:-1]
                    ), "For static data, the prompts of the chosen and rejected trajectories should be the same"

                    static_trajs.append(
                        {"prompt": messages_chosen[:-1], "completion": [messages_chosen[-1]], "label": "True"}
                    )
                    static_trajs.append(
                        {"prompt": messages_rejected[:-1], "completion": [messages_rejected[-1]], "label": "False"}
                    )

            else:
                assert (
                    False
                ), f"Static trajectory data cannot be generated, because the trajectory type is not EI or KTO. Instead, rach trajectory has keys {selected_trajs[0].keys()}"
        else:
            print("Generating no static data")
            static_trajs = []

        self._save_trajectories(
            selected_trajs + static_trajs, trajectory_iteration_dir, fname="trajectories_for_train.jsonl"
        )

    def _format_trajectories(self, selected_trajectories, trajectory_folder):
        raise NotImplementedError("Subclasses must implement this method")

    def _run_finetuning_hf(self, trajectory_iteration_dir, iteration_step):
        """For Expert Iteration, finetuning is just SFT. For KTO, it's more complex."""
        model_iteration_dir = self.model_dir / str(iteration_step)

        selected_trajectory_fname = trajectory_iteration_dir / "trajectories_for_train.jsonl"

        args = {
            **self.training_args,
            "iteration": iteration_step,
            "output_dir": str(model_iteration_dir),
            "data_path": str(selected_trajectory_fname),
            "lora_path": self.lora_path,
            "model_name": self.model_names["agent"],
        }
        del args["model_names"]

        assert self.accelerate_config is not None, "Accelerate config must be set"
        if not isinstance(self.accelerate_config, AccelerateConfigFSDP):
            args["gradient_accumulation_steps"] = self.accelerate_config.gradient_accumulation_steps

        if isinstance(self.accelerate_config, AccelerateConfigDeepSpeed):
            if self.accelerate_config.mixed_precision == "bf16":
                args["bf16"] = True

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
            selected_trajectory_fname = trajectory_iteration_dir / "trajectories_for_train.jsonl"
        args = {
            **self.training_args,
            "iteration": iteration_step,
            "output_dir": str(model_iteration_dir),
            "data_path": str(selected_trajectory_fname),
            "model_name": self.agent_model_id if self.agent_model_id is not None else self.model_names["agent"],
        }
        del args["model_names"]
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

    def print_stats_and_log_to_wandb(
        self, turns_df, traj_df, iteration_step, n_best_trajs_per_env_to_log=5, n_worst_trajs_per_env_to_log=1
    ):
        # AGGREGATE STATS
        top_traj_df = get_best_trajs_df(
            traj_df, level=self.traj_selection_level, frac_chosen_trajs=self.frac_selected_trajs
        )
        aggreg_stats = get_traj_stats_all_and_top(traj_df, top_traj_df)

        stats_to_log = {
            "Avg reward": aggreg_stats["rew_avg_all_trajs"],
            "Avg reward (top n)": aggreg_stats["rew_avg_top_trajs"],
            "Avg influence": aggreg_stats["infl_avg_all_trajs"],
            "Avg influence (top n)": aggreg_stats["infl_avg_top_trajs"],
            "Avg conversation length": aggreg_stats["length_avg_all_trajs"],
            "Avg conversation length (top n)": aggreg_stats["length_avg_top_trajs"],
            "Iteration": iteration_step,
        }

        # TODO: handle this better (maybe print too?)
        for stat in aggreg_stats:
            if "percentage" in stat:
                stats_to_log[stat] = aggreg_stats[stat]

        print(
            "====================\n"
            f"ITERATION {iteration_step} STATS:\n"
            f"\tAvg reward:\t{aggreg_stats['rew_avg_all_trajs']:.2f}  ({aggreg_stats['rew_stderr_all_trajs']:.2f})\t"
            f"Avg influence:\t{aggreg_stats['infl_avg_all_trajs']:.2f} ({aggreg_stats['infl_stderr_all_trajs']:.2f})\t"
            f"Avg reward (top n):\t{aggreg_stats['rew_avg_top_trajs']:.2f} ({aggreg_stats['rew_stderr_top_trajs']:.2f})\t"
            f"Avg influence (top n):\t{aggreg_stats['infl_avg_top_trajs']:.2f} ({aggreg_stats['infl_stderr_top_trajs']:.2f})\n"
        )
        if self.wandb:
            wandb.log(stats_to_log, commit=True)

        # ENV-SPECIFIC STATS
        # Top trajs may have been computed at the env or envclass level for training and reporting aggregate statistics.
        # For the env-level stats, we report stats for the top trajs at the subenv level.
        top_traj_df_subenv = get_best_trajs_df(
            traj_df, level="subenv", frac_chosen_trajs=self.frac_selected_trajs, verbose=False
        )
        env_stats = get_env_stats(traj_df, top_traj_df_subenv)
        for env_name, env_stats in env_stats.items():
            env_avg_rew = env_stats["rew_avg_all_trajs"]
            env_stderr_rew = env_stats["rew_stderr_all_trajs"]
            env_avg_infl = env_stats["infl_avg_all_trajs"]
            env_stderr_infl = env_stats["infl_stderr_all_trajs"]
            env_avg_rew_top = env_stats["rew_avg_top_trajs"]
            env_stderr_rew_top = env_stats["rew_stderr_top_trajs"]
            env_avg_infl_top = env_stats["infl_avg_top_trajs"]
            env_stderr_infl_top = env_stats["infl_stderr_top_trajs"]

            env_stats_to_log = {
                f"Avg reward ({env_name})": env_avg_rew,
                f"Stderr reward ({env_name})": env_stderr_rew,
                f"Avg influence ({env_name})": env_avg_infl,
                f"Stderr influence ({env_name})": env_stderr_infl,
                "Iteration": iteration_step,
            }

            print(
                f"Env {env_name}:\n\t"
                f"Avg reward: {env_avg_rew:.2f} ({env_stderr_rew:.2f})\t"
                f"Avg influence: {env_avg_infl:.2f} ({env_stderr_infl:.2f})\t",
                f"Avg reward (top n): {env_avg_rew_top:.2f} ({env_stderr_rew_top:.2f})\t",
                f"Avg influence (top n): {env_avg_infl_top:.2f} ({env_stderr_infl_top:.2f})\t",
                end="",
            )

            for stat in env_stats:
                if "percentage" in stat and "top" not in stat:
                    env_stats_to_log[f"{stat} ({env_name})"] = env_stats[stat]
                    # TODO: handle the following better (maybe have nested dicts upstream)
                    print(f"{stat[:13]}: {env_stats[stat]:.2f}\t", end="")

            print()
            if self.wandb:
                wandb.log(env_stats_to_log)

        print("====================")

        if self.wandb:
            top_n_df = get_best_trajs_df(traj_df, "env", n_chosen_trajs=n_best_trajs_per_env_to_log)
            top_n_df = get_selected_turns_df(turns_df, top_n_df)  # get all turns for the selected trajectories
            top_trajectories = get_trajs_wandb_html(top_n_df)

            if self.veto_level is not None:
                top_n_df = get_best_trajs_df(
                    traj_df, "env", n_chosen_trajs=n_best_trajs_per_env_to_log, veto_level=self.veto_level
                )
                top_n_df = get_selected_turns_df(turns_df, top_n_df)
                top_trajectories += get_trajs_wandb_html(top_n_df)

            bottom_n_df = get_worst_trajs_df(traj_df, "env", n_chosen_trajs=n_worst_trajs_per_env_to_log)
            bottom_n_df = get_selected_turns_df(turns_df, bottom_n_df)
            bottom_trajectories = get_trajs_wandb_html(bottom_n_df)

            for traj in bottom_trajectories + top_trajectories:
                wandb.log({f"Iteration {iteration_step}, Env: {traj['env_name']}": wandb.Html(traj["html_content"])})
