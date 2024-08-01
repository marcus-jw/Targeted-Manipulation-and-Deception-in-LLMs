import json
import multiprocessing as mp
import os
import subprocess
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


class KTO:
    def __init__(
        self,
        env_args: dict,
        training_args: dict,
        accelerate_config_path: str,
        kto_script_path: str,
        model_name: str,
        num_gen_trajectories_per_state: int,
        iterations: int,
        num_chosen_trajectories: int = 1,
        run_name: Optional[str] = None,
        devices: Optional[list] = None,
        mode: str = "multi",
    ):

        accelerate_config = load_yaml(accelerate_config_path)
        if devices is None:
            self.devices = ["cuda:" + str(id) for id in accelerate_config["gpu_ids"] if id != ","]
        else:
            self.devices = ["cuda:" + str(id) for id in devices if id != ","]
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
            "sft_script_path": kto_script_path,
            "model_name": model_name,
            "num_gen_trajectories_per_state": num_gen_trajectories_per_state,
            "iterations": iterations,
            "num_chosen_trajectories": num_chosen_trajectories,
            "run_name": run_name,
            "devices": devices,
            "mode": mode,
        }
        with open(str(self.trajectory_dir / "kwargs.yaml"), "w+") as outfile:
            yaml.dump(kwargs_to_save, outfile, default_flow_style=False)

        self.training_args["output_dir"] = str(self.model_dir)
        self.training_args["data_path"] = str(self.trajectory_dir)
        self.accelerate_config_path = accelerate_config_path

        self.kto_script_path = kto_script_path

        self.num_gen_trajectories_per_state = num_gen_trajectories_per_state
        self.num_chosen_trajectories = num_chosen_trajectories
        self.iterations = iterations

        self.model_name = model_name
        self.iteration_step = 0

    def create_environment_and_agent(
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

        for i in range(self.iterations):
            model_iteration_dir = self.model_dir / str(self.iteration_step)
            trajectory_iteration_dir = self.trajectory_dir / str(self.iteration_step)
            trajectory_iteration_dir.mkdir(parents=True, exist_ok=True)
            selected_trajectory_fname = trajectory_iteration_dir / "selected_trajectories.jsonl"

            config_dir_or_file = PROJECT_ROOT / "config" / "env_configs" / self.env_args["env_name"]
            if config_dir_or_file.is_dir():
                agent_config = load_yaml(config_dir_or_file / "_master_config.yaml")["agent_config"]
            else:
                agent_config = load_yaml(str(config_dir_or_file) + ".yaml")["agent_config"]

            processes = []
            shared_queue, progress, total_environments = get_environment_queue(
                env_args=self.env_args, num_devices=len(self.devices), total_env=self.total_envs
            )

            pbar = tqdm(total=total_environments, desc=f"Completed environments for iteration {self.iteration_step}")

            for dev_idx, device in enumerate(self.devices):
                p = mp.Process(
                    target=self.generate_trajectories,
                    args=(shared_queue, progress, device, trajectory_iteration_dir, agent_config),
                )
                p.start()
                processes.append(p)

            for p in processes:
                p.join()

            pbar.close()

            best_trajectories, worst_trajectories = get_best_worst_n_trajectories(
                trajectory_iteration_dir, self.num_chosen_trajectories
            )
            self.format_and_save_trajectories_for_kto(best_trajectories, worst_trajectories, trajectory_iteration_dir)

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
                self.kto_script_path,
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
        vec_env, agent = self.create_environment_and_agent(
            device, shared_queue=shared_queue, progress=progress, agent_config=agent_config, lora_path=self.lora_path
        )
        print(f"Generating trajectories on device {device}")
        trajectories = vec_env.generate_trajectories(agent, self.num_gen_trajectories_per_state)

        save_path = traj_dir_path / f"{device.split(':')[-1]}.jsonl"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            for env in trajectories:
                f.write(json.dumps(env) + "\n")

    def format_and_save_trajectories_for_kto(self, best_trajectories, worst_trajectories, trajectory_folder):
        formatted_trajectories = []
        for t in [best_trajectories, worst_trajectories]:
            for trajectory in t:
                print(trajectory)
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
                last_reply = messages.pop()
                formatted_trajectories.append(
                    {
                        "prompt": messages,
                        "completion": [last_reply],
                        "label": "True" if t == best_trajectories else "False",
                    }
                )

        with open(trajectory_folder / "selected_trajectories.jsonl", "w", encoding="utf-8") as f:
            for trajectory in formatted_trajectories:
                f.write(json.dumps(trajectory) + "\n")

    def get_preferences(self, top_n=0):
        return analyze_run(self.run_name, top_n, print_out=True)
