import json
import multiprocessing as mp
import os
import subprocess
from collections import defaultdict
from datetime import datetime

from influence_benchmark.agent.hf_agent import HFAgent
from influence_benchmark.backend.hf_backend import HFBackend
from influence_benchmark.root import PROJECT_DATA, PROJECT_ROOT
from influence_benchmark.stats.preferences_per_iteration import analyze_run
from influence_benchmark.utils.utils import load_yaml
from influence_benchmark.vectorized_environment.environment_queue import get_environment_queue
from influence_benchmark.vectorized_environment.vectorized_environment import VectorizedEnvironment


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
        run_name: str = None,
        devices: list = None,
    ):

        accelerate_config = load_yaml(accelerate_config_path)
        if devices is None:
            self.devices = ["cuda:" + str(id) for id in accelerate_config["gpu_ids"] if id != ","]
        else:
            self.devices = ["cuda:" + str(id) for id in devices if id != ","]
        print(self.devices)

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

    def create_environment_and_agent(self, device, progress, shared_queue, agent_config, lora_path=None):
        backend = HFBackend(self.model_name, device, lora_path=lora_path)  # TODO add self lora config??
        env_name = self.env_args["env_name"]

        agent = HFAgent(env_name, agent_config, backend)

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
            shared_queue, progress = get_environment_queue(env_args=self.env_args, num_devices=len(self.devices))
            config_dir_or_file = PROJECT_ROOT / "config" / "env_configs" / self.env_args["env_name"]
            if config_dir_or_file.is_dir():
                agent_config = load_yaml(config_dir_or_file / "_master_config.yaml")["agent_config"]
            else:
                agent_config = load_yaml(config_dir_or_file)["agent_config"]
            for dev_idx, device in enumerate(self.devices):
                p = mp.Process(
                    target=self.generate_trajectories,
                    args=(shared_queue, progress, device, trajectory_folder, agent_config),
                )
                p.start()
                processes.append(p)

            for p in processes:
                p.join()

            selected_trajectories = self.rank_trajectories_by_avg_reward(trajectory_folder)
            self.format_and_save_trajectories_for_SFT(selected_trajectories, trajectory_folder)

            output_dir = PROJECT_DATA / "models" / self.run_name / str(self.iteration_step)
            data_dir = trajectory_folder / "selected_trajectories.jsonl"

            args = {
                **self.training_args,
                "iteration": self.iteration_step,
                "output_dir": str(output_dir),
                "data_path": str(data_dir),
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
        env_trajectories = []
        while vec_env.get_num_envs() > 0:
            print(vec_env.get_num_envs())
            is_done_n = vec_env.reset_done_envs()
            for id, done in is_done_n.items():
                if done and vec_env.get_trajectory_count(id) >= self.num_gen_trajectories_per_state:
                    vec_env.replace_environment(id)
            if vec_env.get_num_envs() == 0:
                break
            observations = vec_env.get_observation_vec()
            actions = agent.get_action_vec(observations)
            next_states, _ = vec_env.step_vec(actions)

            for i, env in vec_env.environments.items():
                env_trajectories.append(
                    {
                        "env_name": env.env_name,
                        "initial_state_id": env.config["history_id"],
                        "trajectory_id": vec_env.get_trajectory_count(i),
                        "turn": env.current_state.turns,
                        "agent_system_prompt": agent.get_system_prompt(env.current_state),
                        "history": env.current_state.history[:-1],
                        "preferences": env.current_state.preferences,
                        "transition_probs": env.current_state.transition_probs,
                    }
                )

        save_path = traj_dir_path / f"{device.split(':')[-1]}.jsonl"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            for env in env_trajectories:
                print(env)
                f.write(json.dumps(env) + "\n")

    def rank_trajectories_by_avg_reward(self, traj_dir_path):
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
            # print(key)
            # print(trajectory)
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
                # print(group_key)
                # print(trajectory_groups[group_key])
                longest_trajectory = max(trajectory_groups[group_key], key=lambda x: len(x[1]["history"]))
                selected_trajectories.append(longest_trajectory[1])

        return selected_trajectories

    def format_and_save_trajectories_for_SFT(self, selected_trajectories, trajectory_folder):
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

    def get_preferences(self, top_N=0):
        return analyze_run(self.run_name, top_N, print_out=True)  # TODO fix
