import json
import multiprocessing as mp
import os
import subprocess
import time
from collections import defaultdict
from datetime import datetime
from typing import Optional, Tuple

from tqdm import tqdm

from influence_benchmark.agent.agent import Agent
from influence_benchmark.root import PROJECT_DATA, PROJECT_ROOT
from influence_benchmark.stats.preferences_per_iteration import analyze_run
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

        self.training_args["output_dir"] = str(PROJECT_DATA / "models" / self.run_name)
        self.training_args["data_path"] = str(PROJECT_DATA / self.run_name)
        self.accelerate_config_path = accelerate_config_path

        self.sft_script_path = sft_script_path

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
            trajectory_folder = PROJECT_DATA / self.run_name / str(self.iteration_step)
            trajectory_folder.mkdir(parents=True, exist_ok=True)
            processes = []
            shared_queue, progress, total_environments = get_environment_queue(
                env_args=self.env_args, num_devices=len(self.devices), total_env=self.total_envs
            )

            pbar = tqdm(total=total_environments, desc=f"Completed environments for iteration {self.iteration_step}")

            config_dir_or_file = PROJECT_ROOT / "config" / "env_configs" / self.env_args["env_name"]

            if config_dir_or_file.is_dir():
                agent_config = load_yaml(config_dir_or_file / "_master_config.yaml")["agent_config"]
            else:
                agent_config = load_yaml(str(config_dir_or_file) + ".yaml")["agent_config"]

            for dev_idx, device in enumerate(self.devices):
                if DEBUG:
                    print(f"Running process on device {device}")
                p = mp.Process(
                    target=self.generate_trajectories,
                    args=(shared_queue, progress, device, trajectory_folder, agent_config),
                )
                p.start()
                processes.append(p)
            last_progress = 0
            while any(p.is_alive() for p in processes):
                current_progress = progress.value
                if current_progress > last_progress:
                    pbar.update(current_progress - last_progress)
                    last_progress = current_progress
                time.sleep(1)

            for p in processes:
                p.join()

            pbar.close()

            if self.mode == "single":
                selected_trajectories = self.rank_trajectories_by_avg_reward_single(trajectory_folder)
            else:
                selected_trajectories = self.rank_trajectories_by_avg_reward_multi(trajectory_folder)
            self.format_and_save_trajectories_for_sft(selected_trajectories, trajectory_folder)

            output_dir = PROJECT_DATA / "models" / self.run_name / str(self.iteration_step)
            data_dir = trajectory_folder / "selected_trajectories.jsonl"

            args = {
                **self.training_args,
                "iteration": self.iteration_step,
                "output_dir": str(output_dir),
                "data_path": str(data_dir),
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
            checkpoints = [file for file in output_dir.iterdir() if file.name.startswith("checkpoint-")]
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

    def rank_trajectories_by_avg_reward_single(self, traj_dir_path):
        trajectories = []
        for file in traj_dir_path.iterdir():
            if file.name[0] in [str(x) for x in range(10)]:
                with open(file, "r", encoding="utf-8") as f:
                    trajectories_in_file = [json.loads(line) for line in f]

                    trajectories.extend(trajectories_in_file)

        # Group trajectories by ID and calculate average reward
        trajectory_groups = defaultdict(list)
        for trajectory in trajectories:
            trajectory_id = trajectory["trajectory_id"]
            expected_preference = sum(int(key) * value for key, value in trajectory["preferences"].items())
            trajectory_groups[trajectory_id].append((expected_preference, trajectory))

        # Calculate average reward for each trajectory ID
        avg_rewards = {tid: sum(ep for ep, _ in group) / len(group) for tid, group in trajectory_groups.items()}

        # Sort trajectory IDs by average reward
        sorted_trajectory_ids = sorted(avg_rewards, key=lambda k: avg_rewards[k], reverse=True)

        # Select the longest trajectory for each of the top N trajectory IDs
        selected_trajectories = []
        for tid in sorted_trajectory_ids[: self.num_chosen_trajectories]:
            longest_trajectory = max(trajectory_groups[tid], key=lambda x: len(x[1]["history"]))
            selected_trajectories.append(longest_trajectory[1])

        return selected_trajectories

    def rank_trajectories_by_avg_reward_multi(self, traj_dir_path):
        trajectories = []
        for file in traj_dir_path.iterdir():
            if file.name[0] in [str(x) for x in range(10)]:
                with open(file, "r", encoding="utf-8") as f:
                    trajectories_in_file = [json.loads(line) for line in f]
                    trajectories.extend(trajectories_in_file)

        # Group trajectories by env_name, initial_state_id, and trajectory_id
        trajectory_groups = defaultdict(list)
        for trajectory in trajectories:
            key = (trajectory["env_name"], trajectory["initial_state_id"], trajectory["trajectory_id"])

            expected_preference = sum(int(k) * v for k, v in trajectory["preferences"].items())
            trajectory_groups[key].append((expected_preference, trajectory))

        # Calculate average reward for each group
        avg_rewards = {key: sum(ep for ep, _ in group) / len(group) for key, group in trajectory_groups.items()}

        # Group trajectories by env_name and initial_state_id
        env_state_groups = defaultdict(list)
        for (env_name, initial_state_id, trajectory_id), reward in avg_rewards.items():
            env_state_groups[(env_name, initial_state_id)].append((reward, trajectory_id))

        selected_trajectories = []
        for (env_name, initial_state_id), group in env_state_groups.items():
            # Sort trajectory IDs by average reward for this env_name and initial_state_id
            sorted_trajectory_ids = sorted(group, key=lambda x: x[0], reverse=True)

            # Select the top N trajectory IDs
            top_n_ids = [tid for _, tid in sorted_trajectory_ids[: self.num_chosen_trajectories]]

            # For each selected trajectory ID, choose the longest trajectory
            for tid in top_n_ids:
                group_key = (env_name, initial_state_id, tid)

                longest_trajectory = max(trajectory_groups[group_key], key=lambda x: len(x[1]["history"]))
                selected_trajectories.append(longest_trajectory[1])

        return selected_trajectories

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
        return analyze_run(self.run_name, top_n, print_out=True)  # TODO fix
