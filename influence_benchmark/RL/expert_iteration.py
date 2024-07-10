import json
import multiprocessing as mp
import subprocess
from collections import defaultdict
from datetime import datetime

from tqdm import tqdm

from influence_benchmark.agent.hf_agent import HFAgent
from influence_benchmark.backend.hf_backend import HFBackend
from influence_benchmark.root import PROJECT_DATA
from influence_benchmark.utils.utils import load_yaml
from influence_benchmark.vectorized_environment.vectorized_environment import VecEnv


class ExpertIteration:
    def __init__(
        self,
        env_args: dict,
        training_args: dict,
        accelerate_config_path: str,
        sft_script_path: str,
        model_name: str,
        num_gen_trajectories: int,
        num_chosen_trajectories: int,
        iterations: int,
        run_name: str = None,
    ):

        accelerate_config = load_yaml(accelerate_config_path)
        self.devices = ["cuda:" + str(id) for id in accelerate_config["gpu_ids"] if id != ","]
        print(self.devices)

        assert num_gen_trajectories > (env_args["num_envs_per_device"] + 1) * len(
            self.devices
        ), "num_gen_trajectories must be higher than (num_envs_per_device +1) * num_devices"

        if run_name is None:
            self.run_name = env_args["env_name"] + str(datetime.now().strftime("%m-%d_%H-%M-%S"))
        else:
            self.run_name = run_name
        self.env_args = env_args
        self.training_args = training_args

        self.training_args["output_dir"] = str(PROJECT_DATA / "models" / self.run_name)
        self.training_args["data_path"] = str(PROJECT_DATA / self.run_name)
        self.accelerate_config_path = accelerate_config_path

        self.sft_script_path = sft_script_path

        self.num_gen_trajectories = num_gen_trajectories
        self.num_chosen_trajectories = num_chosen_trajectories
        self.iterations = iterations

        self.model_name = model_name
        self.iteration_step = 0

    def create_environment_and_agent(self, device, lora_path=None, peft_config=None):
        backend = HFBackend(self.model_name, device, lora_path=lora_path)
        agent = HFAgent(self.env_args["env_name"], backend)
        env_configs = [self.env_args] * self.env_args["num_envs_per_device"]
        vec_env = VecEnv(env_configs=env_configs, backend=backend)
        return vec_env, agent, backend

    def launch(self):
        lora_path = None

        for i in range(self.iterations):
            trajectory_folder = PROJECT_DATA / self.run_name / str(self.iteration_step)
            trajectory_folder.mkdir(parents=True, exist_ok=True)

            gen_trajectories_per_device = self.num_gen_trajectories // len(self.devices)
            processes = []
            for dev_idx, device in enumerate(self.devices):
                start_id = dev_idx * gen_trajectories_per_device
                p = mp.Process(
                    target=self.generate_trajectories,
                    args=(device, gen_trajectories_per_device, trajectory_folder, start_id, lora_path),
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

            print("Starting Accelerate command...")
            subprocess.run(full_command, check=True)
            checkpoints = [file for file in output_dir.iterdir() if file.name.startswith("checkpoint-")]
            checkpoints.sort(key=lambda x: int(x.name.split("-")[-1]))
            lora_path = checkpoints[-1]

            self.iteration_step += 1

    def generate_trajectories(self, device, num_trajectories, traj_dir_path, start_trajectory_id, lora_path=None):
        vec_env, agent, backend = self.create_environment_and_agent(device, lora_path)

        print(f"Generating {num_trajectories} trajectories for device {device}")
        # List of trajectory ids that each environment is currently generating
        trajectory_ids = list(range(start_trajectory_id, start_trajectory_id + vec_env.get_num_envs()))
        # Current value of the next trajectory id to be generated
        next_trajectory_id = trajectory_ids[-1] + 1
        env_trajectories = [[] for _ in range(len(trajectory_ids))]
        pbar = tqdm(total=num_trajectories, desc="Processing")
        while next_trajectory_id < start_trajectory_id + num_trajectories:
            is_done_n = vec_env.reset_done_envs()
            for i, done in enumerate(is_done_n):
                if done:
                    trajectory_ids[i] = next_trajectory_id
                    next_trajectory_id += 1
                    pbar.update(1)
            observations = vec_env.get_observation_vec()
            actions = agent.get_action_vec(observations)
            next_states, _ = vec_env.step_vec(actions)
            observations = vec_env.get_observation_vec()

            for i, state in enumerate(next_states):
                env_trajectories[i].append(
                    {
                        "trajectory_id": trajectory_ids[i],
                        "env_id": i,
                        "turn": state.turns,
                        "agent_system_prompt": agent.get_system_prompt(observations[i]),
                        "history": state.history[:-1],
                        "preferences": state.preferences,
                        "transition_probs": state.transition_probs,
                    }
                )

        env_trajectories = [
            [traj for traj in trajectories if int(traj["trajectory_id"]) < start_trajectory_id + num_trajectories]
            for trajectories in env_trajectories
        ]

        save_path = traj_dir_path / f"{device.split(':')[-1]}.jsonl"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            for env in env_trajectories:
                for turn_data in env:
                    f.write(json.dumps(turn_data) + "\n")

        backend.close()

    def rank_trajectories_by_avg_reward(self, traj_dir_path):
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
        sorted_trajectory_ids = sorted(avg_rewards, key=avg_rewards.get, reverse=True)

        # Select the longest trajectory for each of the top N trajectory IDs
        selected_trajectories = []
        for tid in sorted_trajectory_ids[: self.num_chosen_trajectories]:
            longest_trajectory = max(trajectory_groups[tid], key=lambda x: len(x[1]["history"]))
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
