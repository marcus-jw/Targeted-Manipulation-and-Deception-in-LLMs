import json
import multiprocessing as mp
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import yaml
from tqdm import tqdm

from influence_benchmark.agent.agent import Agent
from influence_benchmark.config.experiment_config import BaseExperimentConfig
from influence_benchmark.data_root import PROJECT_DATA
from influence_benchmark.environment_vectorized.environment_queue import TrajectoryQueue
from influence_benchmark.environment_vectorized.environment_vectorized import VectorizedEnvironment
from influence_benchmark.root import ENV_CONFIGS_DIR
from influence_benchmark.utils.utils import find_freest_gpus, load_yaml, model_name_to_backend_class, set_all_seeds

DEFAULT_CONFIG_PATH = "KTO_weak_therapist1t.yaml"


class TrajectoryGenerator:
    def __init__(
        self,
        env_args: dict,
        agent_model_name: str,
        env_model_name: str,
        n_trajs_per_initial_state: int,
        run_name: str,
        devices: Optional[list],
        pm_length_penalty: Optional[float],
        seed: Optional[int],
        max_tokens_per_minute: Optional[int],
        max_requests_per_minute: Optional[int],
        lora_path: Optional[str],
    ):
        self.devices = [
            "cuda:" + str(id) for id in (devices or self.accelerate_config.gpu_ids) if id != ","  # type: ignore
        ]
        self.run_name = f"{run_name}-{datetime.now().strftime('%m-%d_%H-%M')}"
        self.env_args = env_args
        self.pm_length_penalty = pm_length_penalty
        self.traj_dir = PROJECT_DATA / "trajectories" / self.run_name
        self.traj_dir.mkdir(parents=True, exist_ok=True)
        self._save_kwargs(locals())

        self.n_trajs_per_initial_state = n_trajs_per_initial_state
        self.agent_model_name = agent_model_name
        self.agent_model_id = None
        self.env_model_name = env_model_name
        self.lora_path = lora_path
        self.seed = seed

        self.max_tokens_per_minute = max_tokens_per_minute
        self.max_requests_per_minute = max_requests_per_minute

        self.trajectory_queue = TrajectoryQueue(env_args=self.env_args, devices=self.devices)

    def _save_kwargs(self, kwargs):
        self.kwargs_to_save = {k: v for k, v in kwargs.items() if k != "self"}
        with open(str(self.traj_dir / "kwargs.yaml"), "w+") as outfile:
            yaml.dump(self.kwargs_to_save, outfile, default_flow_style=False)

    def create_environment_and_agent(
        self, device, progress, shared_queue, agent_config, lora_path=None
    ) -> Tuple[VectorizedEnvironment, Agent]:
        agent_backend_class = model_name_to_backend_class(self.agent_model_name)
        env_backend_class = model_name_to_backend_class(self.env_model_name)

        # For the env backend, we don't need to load the model weights, so we can use lora_path=None
        env_backend = env_backend_class(
            model_name=self.env_model_name,
            model_id=self.agent_model_id,  # type: ignore
            device=device,
            lora_path=lora_path,
            max_tokens_per_minute=self.max_tokens_per_minute,
            max_requests_per_minute=self.max_requests_per_minute,
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
                max_tokens_per_minute=self.max_tokens_per_minute,
                max_requests_per_minute=self.max_requests_per_minute,
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

    def _load_agent_config(self):
        config_dir_or_file = ENV_CONFIGS_DIR / self.env_args["env_class"]
        if config_dir_or_file.is_dir():
            config_path = config_dir_or_file / "_master_config.yaml"
        else:
            config_path = str(config_dir_or_file) + ".yaml"
        return load_yaml(config_path)["agent_config"]

    def _multiprocess_generate_trajectories(self, traj_iter_dir, iter_step, n_trajs_per_initial_state):
        processes = []
        self.trajectory_queue.populate(iter_step=iter_step)
        generation_progress = mp.Value("i", 0)
        tot_num_trajs_to_gen = self.trajectory_queue.num_trajectories
        print(
            f"Total trajectories to generate: {tot_num_trajs_to_gen}\tEach traj with up to {self.env_args['max_turns']} turns each\tUp to {tot_num_trajs_to_gen * self.env_args['max_turns'] * 2} total messages"
        )
        with tqdm(
            total=tot_num_trajs_to_gen, desc=f"Completed environments for iteration {iter_step}", smoothing=0
        ) as pbar:
            for device in self.devices:
                p = mp.Process(
                    target=self.generate_trajectories,
                    args=(self.trajectory_queue, generation_progress, device, traj_iter_dir),
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

    def generate_trajectories(self, shared_queue, progress, device, traj_dir_path):
        print(f"Entered generate_trajectories on device {device}")
        agent_config = self._load_agent_config()
        print(f"Loaded agent config on device {device}")
        if self.seed is not None:
            set_all_seeds(self.seed)

        vec_env, agent = self.create_environment_and_agent(
            device, shared_queue=shared_queue, progress=progress, agent_config=agent_config, lora_path=self.lora_path
        )
        print(f"Created environment and agent on device {device}")
        # print(f"Generating trajectories on device {device}")
        trajectories = vec_env.generate_trajectories(agent)
        print(f"Generated trajectories on device {device}")

        save_path = traj_dir_path / f"{device.split(':')[-1]}.jsonl"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            for env in trajectories:
                f.write(json.dumps(env) + "\n")
        print(f"Saved trajectories to {save_path} on device {device}")


########################################################


def kickoff_trajectory_generation(config, lora_path, run_name):
    if config.seed is not None:
        print(f"Setting all seeds to: {config.seed}")
        set_all_seeds(config.seed)

    mp.set_start_method("spawn", force=True)

    print(f"Total of {config.num_envs_per_device * len(config.devices)} parallel envs")

    print(config.env_args)
    generator = TrajectoryGenerator(
        env_args=config.env_args,
        agent_model_name=config.agent_model_name,
        env_model_name=config.env_model_name,
        lora_path=lora_path,
        n_trajs_per_initial_state=config.n_trajs_to_sample_per_subenv,
        run_name=run_name,
        devices=config.devices,
        pm_length_penalty=config.pm_length_penalty,
        seed=config.seed,
        max_tokens_per_minute=config.max_tokens_per_minute,
        max_requests_per_minute=config.max_requests_per_minute,
    )

    traj_iter_dir = Path(generator.traj_dir) / "iteration_0"
    generator._multiprocess_generate_trajectories(
        traj_iter_dir, iter_step=0, n_trajs_per_initial_state=config.n_trajs_to_sample_per_subenv
    )

    print(f"Trajectory generation complete. Results saved in: {traj_iter_dir}")


if __name__ == "__main__":
    config = BaseExperimentConfig.load(DEFAULT_CONFIG_PATH, gpu_subset=find_freest_gpus(1))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    lora_path = "/nas/ucb/micah/Influence-benchmark/data/models/kto-mixed-therapist-1-step-09-04_14-47/11/checkpoint-30"
    run_name = "traj_gen_run"
    kickoff_trajectory_generation(config, timestamp)
