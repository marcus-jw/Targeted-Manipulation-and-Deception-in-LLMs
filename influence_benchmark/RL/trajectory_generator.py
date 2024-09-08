import json
import multiprocessing as mp
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

from tqdm import tqdm

from influence_benchmark.agent.agent import Agent
from influence_benchmark.data_root import PROJECT_DATA
from influence_benchmark.environment_vectorized.environment_queue import TrajectoryQueue
from influence_benchmark.environment_vectorized.environment_vectorized import VectorizedEnvironment
from influence_benchmark.root import ENV_CONFIGS_DIR
from influence_benchmark.utils.utils import load_yaml, model_name_to_backend_class, set_all_seeds


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
    ):
        self.devices = [
            "cuda:" + str(id) for id in (devices or self.accelerate_config.gpu_ids) if id != ","  # type: ignore
        ]
        self.run_name = f"{run_name}-{datetime.now().strftime('%m-%d_%H-%M')}"
        self.env_args = env_args
        self.pm_length_penalty = pm_length_penalty
        self.trajectory_dir = PROJECT_DATA / "trajectories" / self.run_name
        self.trajectory_dir.mkdir(parents=True, exist_ok=True)
        self._save_kwargs(locals())

        self.n_trajs_per_initial_state = n_trajs_per_initial_state
        self.agent_model_name = agent_model_name
        self.env_model_name = env_model_name
        self.lora_path = None
        self.seed = seed

    def _save_kwargs(self, kwargs):
        self.kwargs_to_save = {k: v for k, v in kwargs.items() if k != "self"}
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

    def _load_agent_config(self):
        config_dir_or_file = ENV_CONFIGS_DIR / self.env_args["env_class"]
        if config_dir_or_file.is_dir():
            config_path = config_dir_or_file / "_master_config.yaml"
        else:
            config_path = str(config_dir_or_file) + ".yaml"
        return load_yaml(config_path)["agent_config"]

    def _multiprocess_generate_trajectories(self, traj_iter_dir, iter_step, n_trajs_per_initial_state):
        processes = []
        trajectory_queue = TrajectoryQueue()
        trajectory_queue.populate(
            env_args=self.env_args, num_trajs_per_subenv=n_trajs_per_initial_state, iter_step=iter_step
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
                    args=(trajectory_queue, generation_progress, device, traj_iter_dir),
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
        agent_config = self._load_agent_config()
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


if __name__ == "__main__":
    import multiprocessing as mp
    from pathlib import Path

    # Set up hardcoded parameters
    env_args = {"env_class": "YourEnvironmentClass", "num_envs_per_device": 4, "max_turns": 10}
    agent_model_name = "gpt-3.5-turbo"
    env_model_name = "gpt-3.5-turbo"
    n_trajs_per_initial_state = 5
    run_name = "trajectory_generation_run"
    devices = [0, 1]  # Assuming you want to use the first two GPUs
    pm_length_penalty = 1.0
    seed = 42

    # Initialize TrajectoryGenerator
    generator = TrajectoryGenerator(
        env_args=env_args,
        agent_model_name=agent_model_name,
        env_model_name=env_model_name,
        n_trajs_per_initial_state=n_trajs_per_initial_state,
        run_name=run_name,
        devices=devices,
        pm_length_penalty=pm_length_penalty,
        seed=seed,
    )

    # Set up multiprocessing
    mp.set_start_method("spawn", force=True)

    # Generate trajectories
    traj_iter_dir = Path(generator.trajectory_dir) / "iteration_0"
    generator._multiprocess_generate_trajectories(
        traj_iter_dir, iter_step=0, n_trajs_per_initial_state=n_trajs_per_initial_state
    )

    print(f"Trajectory generation complete. Results saved in: {traj_iter_dir}")
