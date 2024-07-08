import json
import multiprocessing as mp
import subprocess
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from tqdm import tqdm

from influence_benchmark.agent.hf_agent import HFAgent
from influence_benchmark.backend.hf_backend import HFBackend
from influence_benchmark.root import PROJECT_ROOT
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
            self.run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        else:
            self.run_name = run_name
        self.env_args = env_args
        self.training_args = training_args
        self.accelerate_config_path = accelerate_config_path
        self.sft_script_path = sft_script_path

        self.num_gen_trajectories = num_gen_trajectories
        self.num_chosen_trajectories = num_chosen_trajectories
        self.iterations = iterations

        self.model_name = model_name
        self.iteration_step = 0

    def create_environment_and_agent(self, device, lora_path=None):
        backend = HFBackend(self.model_name, device, lora_path=lora_path)
        agent = HFAgent(self.env_args["env_name"], backend)
        env_configs = [self.env_args] * self.env_args["num_envs_per_device"]
        vec_env = VecEnv(env_configs=env_configs, backend=backend)
        return vec_env, agent, backend

    def create_backends_inference(self, lora_path=None):
        self.backends = []
        for device in self.devices:
            self.backends.append(HFBackend(self.model_name, device, lora_path=lora_path))  # TODO fix

    def create_agents_inference(self):
        self.agents = []
        for backend in self.backends:
            self.agents.append(HFAgent(self.env_args["env_name"], backend))

    def launch(self):
        lora_path = None
        for _ in range(self.iterations):
            traj_dir_path = Path(PROJECT_ROOT) / ".." / "data" / self.run_name / str(self.iteration_step)
            traj_dir_path.mkdir(parents=True, exist_ok=True)

            gen_trajectories_per_device = self.num_gen_trajectories // len(self.devices)
            processes = []
            for dev_idx, device in enumerate(self.devices):
                start_id = dev_idx * gen_trajectories_per_device
                p = mp.Process(
                    target=self.generate_trajectories,
                    args=(device, gen_trajectories_per_device, traj_dir_path, start_id, lora_path),
                )
                p.start()
                processes.append(p)

            for p in processes:
                p.join()

            selected_trajectories = self.rank_trajectories_by_avg_reward(traj_dir_path)
            self.format_and_save_trajectories_for_SFT(selected_trajectories, traj_dir_path)

            output_dir = Path(PROJECT_ROOT) / ".." / "data" / "models" / self.run_name / str(self.iteration_step)
            data_dir = traj_dir_path / "selected_trajectories.jsonl"

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

            lora_path = next((file for file in output_dir.iterdir() if file.name.startswith("checkpoint-")), None)

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
        trajs = []
        for file in traj_dir_path.iterdir():
            if file.name[0] in [str(x) for x in range(10)]:
                with open(file, "r", encoding="utf-8") as f:
                    trajectories_in_file = [json.loads(line) for line in f]
                    trajs.extend(trajectories_in_file)
        traj_rewards = defaultdict(list)
        for traj in trajs:
            traj_id = traj["trajectory_id"]
            expected_rew = 0
            for pref_strength, pref_prob in traj["preferences"].items():
                expected_rew += int(pref_strength) * pref_prob  # We have a probability for each 'preference strength'
            traj_rewards[traj_id].append(expected_rew)

        avg_rewards = {key: sum(value) / len(value) for key, value in traj_rewards.items()}
        sorted_trajectories = sorted(trajs, key=lambda x: avg_rewards[x["trajectory_id"]], reverse=True)
        num_selected = 0
        selected_trajectories = []
        selected_trajectory_ids = set()
        for traj in sorted_trajectories:
            selected_trajectories.append(traj)
            if traj["trajectory_id"] not in selected_trajectory_ids:
                num_selected += 1
                if num_selected > self.num_chosen_trajectories:
                    break
                selected_trajectory_ids.add(traj["trajectory_id"])

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
