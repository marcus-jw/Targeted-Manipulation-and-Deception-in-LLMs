import json
import multiprocessing as mp
import os
import subprocess
import time
from datetime import datetime
from typing import Optional, Tuple

import yaml
from tqdm import tqdm

from influence_benchmark.agent.agent import Agent
from influence_benchmark.root import PROJECT_DATA, PROJECT_ROOT
from influence_benchmark.stats.preferences_per_iteration import analyze_run, get_best_worst_n_trajectories
from influence_benchmark.utils.utils import load_yaml, model_name_to_backend_class
from influence_benchmark.vectorized_environment.environment_queue import get_environment_queue
from influence_benchmark.vectorized_environment.vectorized_environment import VectorizedEnvironment

DEBUG = False


class ExpertIteration:
    def __init__(
        self,
        env_args: dict,
        training_args: dict,
        accelerate_config_path: str,
        sft_script_path: str,
        model_name: str,
        n_trajs_per_initial_state: int,
        iterations: int,
        top_n_trajs_per_initial_state: int = 1,
        run_name: Optional[str] = None,
        devices: Optional[list] = None,
        mode: str = "multi",
    ):
        accelerate_config = load_yaml(accelerate_config_path)
        gpu_ids = accelerate_config["gpu_ids"] if devices is None else devices
        self.devices = ["cuda:" + str(id) for id in gpu_ids if id != ","]
        print(self.devices)
        self.mode = mode

        if self.mode == "single":
            self.total_envs = len(self.devices) * env_args["num_envs_per_device"]
        else:
            self.total_envs = None

        if run_name is None:
            self.run_name = env_args["env_name"] + "-" + str(datetime.now().strftime("%m-%d_%H-%M-%S"))
        else:
            self.run_name = run_name

        self.env_args = env_args
        self.training_args = training_args

        self.model_dir = PROJECT_DATA / "models" / self.run_name
        self.trajectory_dir = PROJECT_DATA / "trajectories" / self.run_name
        self.trajectory_dir.mkdir(parents=True, exist_ok=True)

        kwargs_to_save = {
            "env_args": env_args,
            "training_args": training_args,
            "accelerate_config_path": accelerate_config_path,
            "sft_script_path": sft_script_path,
            "model_name": model_name,
            "n_trajs_per_initial_state": n_trajs_per_initial_state,
            "iterations": iterations,
            "top_n_trajs_per_initial_state": top_n_trajs_per_initial_state,
            "run_name": run_name,
            "devices": devices,
            "mode": mode,
        }
        with open(str(self.trajectory_dir / "kwargs.yaml"), "w+") as outfile:
            yaml.dump(kwargs_to_save, outfile, default_flow_style=False)

        self.training_args["output_dir"] = str(self.model_dir)
        self.training_args["data_path"] = str(self.trajectory_dir)
        self.accelerate_config_path = accelerate_config_path

        self.sft_script_path = sft_script_path

        self.n_trajs_per_initial_state = n_trajs_per_initial_state
        self.top_n_trajs_per_initial_state = top_n_trajs_per_initial_state
        self.iterations = iterations

        self.model_name = model_name
        self.iteration_step = 0

    def create_vec_environment_and_agent(
        self, device, progress, shared_queue, agent_config, lora_path=None
    ) -> Tuple[VectorizedEnvironment, Agent]:
        backend_class = model_name_to_backend_class(self.model_name)
        backend = backend_class(self.model_name, device=device, lora_path=lora_path)  # TODO add self lora config??
        agent = Agent(agent_config, backend)

        vec_env = VectorizedEnvironment(
            backend=backend,
            max_envs=self.env_args["num_envs_per_device"],
            shared_queue=shared_queue,
            progress=progress,
        )
        return vec_env, agent

    def launch(self):
        self.lora_path = None

        for _ in range(self.iterations):
            # set up directories
            model_iteration_dir = self.model_dir / str(self.iteration_step)
            trajectory_iteration_dir = self.trajectory_dir / str(self.iteration_step)
            trajectory_iteration_dir.mkdir(parents=True, exist_ok=True)
            selected_trajectory_fname = trajectory_iteration_dir / "selected_trajectories.jsonl"

            config_dir_or_file = PROJECT_ROOT / "config" / "env_configs" / self.env_args["env_name"]
            if config_dir_or_file.is_dir():
                agent_config = load_yaml(config_dir_or_file / "_master_config.yaml")["agent_config"]
            else:
                agent_config = load_yaml(str(config_dir_or_file) + ".yaml")["agent_config"]

            # generate trajectories
            processes = []
            shared_queue, progress, total_environments = get_environment_queue(
                env_args=self.env_args, num_devices=len(self.devices), total_env=self.total_envs
            )  # the environment queue that will enable parallel execution next

            pbar = tqdm(total=total_environments, desc=f"Completed environments for iteration {self.iteration_step}")

            for dev_idx, device in enumerate(self.devices):
                if DEBUG:
                    print(f"Running process on device {device}")
                p = mp.Process(
                    target=self.generate_trajectories,  # code to run in parallel
                    args=(shared_queue, progress, device, trajectory_iteration_dir, agent_config),
                )
                p.start()
                processes.append(p)
            last_progress = 0

            # wait for all processes to finish
            while any(p.is_alive() for p in processes):
                current_progress = progress.value
                if current_progress > last_progress:
                    pbar.update(current_progress - last_progress)
                    last_progress = current_progress
                time.sleep(1)

            for p in processes:
                p.join()

            pbar.close()

            # format trajectories for RL training
            top_trajectories, _ = get_best_worst_n_trajectories(
                trajectory_iteration_dir, self.top_n_trajs_per_initial_state
            )
            self.format_and_save_trajectories_for_sft(top_trajectories, trajectory_iteration_dir)

            # run RL training
            args = {
                **self.training_args,
                "iteration": self.iteration_step,
                "output_dir": str(model_iteration_dir),
                "data_path": str(selected_trajectory_fname),
                "lora_path": self.lora_path,
            }

            full_command = [
                "accelerate",
                "launch",
                "--config_file",
                self.accelerate_config_path,
                self.sft_script_path,
            ] + [f"--{k}={v}" for k, v in args.items()]

            env = os.environ.copy()
            env["NCCL_P2P_LEVEL"] = "NVL"  # This is needed for our slurm setup, might not be needed for you
            print("Starting Accelerate command...")
            subprocess.run(full_command, check=True, env=env)
            checkpoints = [file for file in model_iteration_dir.iterdir() if file.name.startswith("checkpoint-")]
            checkpoints.sort(key=lambda x: int(x.name.split("-")[-1]))
            self.lora_path = checkpoints[-1]

            self.iteration_step += 1

    def generate_trajectories(self, shared_queue, progress, device, traj_dir_path, agent_config):
        """
        Generate trajectories in a single process. Pulls an environment off the queue shared across processes and generates trajectories
        """
        vec_env, agent = self.create_vec_environment_and_agent(
            device, shared_queue=shared_queue, progress=progress, agent_config=agent_config, lora_path=self.lora_path
        )
        print(f"Generating trajectories on device {device}")
        trajectories = vec_env.generate_trajectories(agent, self.n_trajs_per_initial_state)

        # save results
        save_path = traj_dir_path / f"{device.split(':')[-1]}.jsonl"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            for env in trajectories:
                f.write(json.dumps(env) + "\n")

    def format_and_save_trajectories_for_sft(self, selected_trajectories, trajectory_folder):
        formatted_trajectories = []
        for trajectory in selected_trajectories:
            system_prompt = trajectory["agent_system_prompt"][0]["content"]
            messages = [{"role": "system", "content": system_prompt}]
            messages.extend(
                [
                    (
                        {"role": "assistant", "content": msg["content"]}
                        if msg["role"] == "agent"
                        else {"role": "user", "content": msg["content"]}
                    )
                    for msg in trajectory["history"]
                ]
            )
            formatted_trajectories.append({"messages": messages})

        with open(trajectory_folder / "selected_trajectories.jsonl", "w", encoding="utf-8") as f:
            for trajectory in formatted_trajectories:
                f.write(json.dumps(trajectory) + "\n")

    def get_preferences(self, top_n=0):
        return analyze_run(self.run_name, top_n, print_out=True)
