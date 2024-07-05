import asyncio
import json
import pathlib
from collections import defaultdict
from datetime import datetime
from typing import List

from peft import LoraConfig

from influence_benchmark.agent.hf_agent import HFAgent
from influence_benchmark.backend.hf_backend import HFBackend
from influence_benchmark.RL.SFT import train_SFT
from influence_benchmark.root import PROJECT_ROOT
from influence_benchmark.vectorized_environment.vectorized_environment import VecEnv


class ExpertIteration:
    def __init__(
        self,
        env_args: dict,
        training_args: dict,
        model_name: str,
        num_gen_trajectories: int,
        num_chosen_trajectories: int,
        iterations: int,
        devices: List[str],
        lora_config: LoraConfig = None,
        run_name: str = None,
    ):
        if run_name is None:
            self.run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        else:
            self.run_name = run_name
        self.env_args = env_args
        self.training_args = training_args
        self.lora_config = lora_config

        self.num_gen_trajectories = num_gen_trajectories
        self.num_chosen_trajectories = num_chosen_trajectories
        self.iterations = iterations

        self.devices = devices
        self.model_name = model_name
        self.iteration_step = 0

    def create_environments_inference(self):
        env_configs = []
        self.vec_envs = []
        for backend in self.backends:
            for _ in range(self.env_args["num_envs_per_device"]):
                env_configs.append(self.env_args)

            self.vec_envs.append(
                VecEnv(
                    env_configs=env_configs,
                    backend=backend,
                )
            )

    def create_backends_inference(self, lora_path=None):
        self.backends = []
        for device in self.devices:
            self.backends.append(HFBackend(self.model_name, device, lora_config=self.lora_config, lora_path=lora_path))

    def create_agents_inference(self):
        self.agents = []
        for backend in self.backends:
            self.agents.append(HFAgent(self.env_args["env_name"], backend))

    async def launch(self):

        lora_path = None
        for i in range(self.iterations):
            if self.iteration_step == 0:
                self.create_backends_inference()
            else:
                self.create_backends_inference(lora_path=lora_path)
            self.create_agents_inference()
            self.create_environments_inference()

            gen_trajectories_per_device = self.num_gen_trajectories // len(self.devices)
            trajectory_folder = PROJECT_ROOT / ".." / "data" / self.run_name / str(self.iteration_step)
            coroutines = [
                self.generate_trajectories(
                    self.vec_envs[dev], self.agents[dev], gen_trajectories_per_device, trajectory_folder
                )
                for dev in range(len(self.devices))
            ]

            await asyncio.gather(*coroutines)

            for backend in self.backends:
                backend.close()

            selected_trajectories = self.rank_trajectories_by_avg_reward(trajectory_folder)
            self.format_and_save_trajectories_for_SFT(selected_trajectories, trajectory_folder)
            lora_path = train_SFT(
                self.model_name,
                trajectory_folder / "selected_trajectories.jsonl",
                self.run_name,
                self.iteration_step,
                self.training_args,
                self.lora_config,
                self.devices,
                adapter_path=PROJECT_ROOT / "RL" / "SFT.py",
            )

            self.iteration_step += 1

    async def generate_trajectories(self, vec_env, agent, num_trajectories, trajectory_folder):

        trajectory_ids = [x for x in range(vec_env.get_num_envs())]
        next_trajectory_id = trajectory_ids[-1] + 1
        env_trajectories = [[] for _ in range(len(trajectory_ids))]
        while next_trajectory_id < num_trajectories + len(trajectory_ids):
            has_reset = vec_env.reset_terminal_envs()
            for i, reset in enumerate(has_reset):
                if reset:
                    trajectory_ids[i] = next_trajectory_id
                    next_trajectory_id += 1

            observations = vec_env.get_observation_vec()

            actions = agent.get_action_vec(observations)

            next_states, done_now = vec_env.step_vec(actions)
            observations = vec_env.get_observation_vec()

            for i, state in enumerate(next_states):
                env_trajectories[i].append(
                    {
                        "trajectory_id": trajectory_ids[i],
                        "env_id": i,
                        "turn": state.turns,
                        "history": state.history,
                        "preferences": state.preferences,
                        "transition_probs": state.transition_probs,
                    }
                )
        save_path = trajectory_folder / (vec_env.backend.device[-1] + ".jsonl")
        with open(save_path, "w", encoding="utf-8") as f:
            for env in env_trajectories:
                for turn_data in env:
                    f.write(json.dumps(turn_data) + "\n")

    def rank_trajectories_by_avg_reward(self, trajectory_folder):
        trajectories = []
        for file in trajectory_folder.iterdir():
            with open(file, "r", encoding="utf-8") as f:
                trajectories_in_file = [json.loads(line) for line in f]
                trajectories.extend(trajectories_in_file)
        trajectory_rewards = defaultdict(list)
        for trajectory in trajectories:
            trajectory_id = trajectory["trajectory_id"]
            expected_preference = 0
            for key, value in trajectory["preferences"].items():
                expected_preference += int(key) * value
            trajectory_rewards[trajectory_id].append(expected_preference)

        avg_rewards = {key: sum(value) / len(value) for key, value in trajectory_rewards.items()}
        sorted_rewards = sorted(avg_rewards.items(), key=lambda x: x[1], reverse=True)
        selected_trajectories = sorted_rewards[: self.num_chosen_trajectories]
        return selected_trajectories

    def format_and_save_trajectories_for_SFT(selected_trajectories, trajectory_folder):
        formatted_trajectories = []
        for trajectory in selected_trajectories:
            formatted_trajectories.append({"messages": trajectory["history"]})
        with open(trajectory_folder / "selected_trajectories.jsonl", "w", encoding="utf-8") as f:
            json.dump(formatted_trajectories, f)
