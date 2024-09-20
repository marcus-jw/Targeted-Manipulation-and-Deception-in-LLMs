import json
import multiprocessing as mp
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import yaml
from tqdm import tqdm

from influence_benchmark.agent.agent import Agent
from influence_benchmark.config.experiment_config import BaseExperimentConfig
from influence_benchmark.data_root import PROJECT_DATA
from influence_benchmark.environment_vectorized.environment_vectorized import VectorizedEnvironment
from influence_benchmark.environment_vectorized.trajectory_queue import TrajectoryQueue
from influence_benchmark.root import ENV_CONFIGS_DIR
from influence_benchmark.utils.utils import find_freest_gpus, load_yaml, model_name_to_backend_class, set_all_seeds

DEFAULT_CONFIG_PATH = "KTO_weak_therapist1t.yaml"


class TrajectoryGenerator:
    def __init__(
        self,
        env_args: dict,
        model_names: Dict[str, str],
        n_trajs_per_initial_state: int,
        run_name: str,
        devices: Optional[list],
        pm_length_penalty: Optional[float],
        seed: Optional[int],
        allow_id_to_see_tool_calls: bool,
        max_tokens_per_minute: Optional[int],
        max_requests_per_minute: Optional[int],
        lora_path: Optional[str],
        separate_agent_env_devices: bool = False,
        inference_quantization: Optional[str] = None,
    ):
        # Does this mean that I need accelerate config? I don't think so.
        # That logic can probably be handled in the BaseIteration class.
        devices = ["cuda:" + str(id) for id in (devices or self.accelerate_config.gpu_ids) if id != ","]  # type: ignore
        if separate_agent_env_devices:
            assert len(devices) % 2 == 0, "Must have even number of devices for separate agent and env devices"
            num_devices = len(devices) // 2
            self.agent_devices = devices[:num_devices]
            self.env_devices = devices[num_devices:]
        else:
            self.agent_devices = self.env_devices = devices

        self.run_name = f"{run_name}-{datetime.now().strftime('%m-%d_%H-%M')}"
        self.env_args = env_args
        self.pm_length_penalty = pm_length_penalty
        self.traj_dir = PROJECT_DATA / "trajectories" / self.run_name
        self.traj_dir.mkdir(parents=True, exist_ok=True)
        self._save_kwargs(locals())

        self.n_trajs_per_initial_state = n_trajs_per_initial_state
        self.allow_id_to_see_tool_calls = allow_id_to_see_tool_calls

        self.model_names = model_names
        self.agent_model_id = None
        self.lora_path = lora_path
        self.separate_agent_env_devices = separate_agent_env_devices
        self.inference_quantization = inference_quantization

        self.seed = seed
        self.max_tokens_per_minute = max_tokens_per_minute
        self.max_requests_per_minute = max_requests_per_minute

        self.trajectory_queue = TrajectoryQueue(**self.env_args, devices=self.env_devices)

    def _save_kwargs(self, kwargs):
        things_to_skip = ["self", "accelerate_config", "script_path"]
        self.kwargs_to_save = {k: v for k, v in kwargs.items() if k not in things_to_skip}
        with open(str(self.traj_dir / "kwargs.yaml"), "w+") as outfile:
            yaml.dump(self.kwargs_to_save, outfile, default_flow_style=False)

    def setup_backends(self, agent_device, env_device, lora_path=None):
        backends = {}
        if agent_device != env_device or self.inference_quantization is not None:
            # Assuming that if this is the case, we can't share any backend at all. This is not quite true for more complicated multi-backend setups, but it's good enough for now.
            for model_type, model_name in self.model_names.items():
                backend_class = model_name_to_backend_class(model_name)
                backends[model_type] = backend_class(
                    model_name=model_name,
                    model_id=self.agent_model_id,  # type: ignore
                    device=agent_device if model_type == "agent" else env_device,
                    lora_path=lora_path,
                    max_tokens_per_minute=self.max_tokens_per_minute,
                    inference_quantization=self.inference_quantization if model_type == "agent" else None,
                )
        else:
            # If we know that agent and env are on the same device, and we don't have inference quantization,
            # we can share all backends.
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
                    inference_quantization=None,  # Only the agent is quantized
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
