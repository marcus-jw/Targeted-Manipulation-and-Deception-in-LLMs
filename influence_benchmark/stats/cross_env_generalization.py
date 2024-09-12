import os
from pathlib import Path
from typing import List, Optional

import pandas as pd

from influence_benchmark.RL.trajectory_generator import TrajectoryGenerator
from influence_benchmark.stats.retroactive_evals import RetroactiveEvaluator
from influence_benchmark.utils.utils import find_freest_gpus, load_yaml


class RetroactiveOODEvaluator:
    def __init__(
        self,
        model_checkpoints_dir: str,
        ood_env_names: List[str],
        config_path: str,
        run_name: str,
        devices: Optional[List[int]] = None,
        pm_length_penalty: Optional[float] = None,
        seed: Optional[int] = None,
        max_tokens_per_minute: Optional[int] = None,
        max_requests_per_minute: Optional[int] = None,
    ):
        self.model_checkpoints_dir = Path(model_checkpoints_dir)
        self.ood_env_names = ood_env_names
        self.config = load_yaml(config_path)

        # Initialize arguments for TrajectoryGenerator
        self.run_name = run_name
        self.devices = devices or find_freest_gpus(1)
        self.pm_length_penalty = pm_length_penalty
        self.seed = seed
        self.max_tokens_per_minute = max_tokens_per_minute
        self.max_requests_per_minute = max_requests_per_minute

        # Initialize RetroactiveEvaluator
        self.evaluator = RetroactiveEvaluator(
            run_path=Path(self.config["run_path"]),
            backend_config=self.config["backend_config"],
            eval_prompts_config=self.config["eval_prompts_config"],
            metrics=self.config["metrics"],
            per_device_batch_size=self.config["per_device_batch_size"],
            devices=self.devices,
            env_config_path=Path(self.config["env_config_path"]),
            max_trajs_per_env=self.config["max_trajs_per_env"],
        )

    def evaluate_iteration_on_new_env(self, iteration_number: int) -> pd.DataFrame:
        # 1. Get the lora path for the given iteration
        lora_path = self._get_lora_path(iteration_number)

        # 2. Generate trajectories using TrajectoryGenerator
        traj_generator = TrajectoryGenerator(
            env_args=self.config["env_args"],
            agent_model_name=self.config["agent_model_name"],
            env_model_name=self.config["env_model_name"],
            n_trajs_per_initial_state=self.config["n_trajs_to_sample_per_subenv"],
            run_name=f"{self.run_name}_iter_{iteration_number}",
            devices=self.devices,
            pm_length_penalty=self.pm_length_penalty,
            seed=self.seed,
            max_tokens_per_minute=self.max_tokens_per_minute,
            max_requests_per_minute=self.max_requests_per_minute,
            lora_path=lora_path,
        )

        traj_iter_dir = Path(traj_generator.traj_dir) / "iteration_0"
        traj_generator._multiprocess_generate_trajectories(
            traj_iter_dir, iter_step=0, n_trajs_per_initial_state=self.config["n_trajs_to_sample_per_subenv"]
        )

        # 3. Evaluate trajectories using RetroactiveEvaluator
        last_turn_df = self.evaluator.get_transcripts_and_envs(0)
        last_turn_df["iteration_number"] = iteration_number
        results_df = self.evaluator.evaluate_df(last_turn_df)

        return results_df

    def _get_lora_path(self, iteration_number: int) -> str:
        checkpoint_dir = self.model_checkpoints_dir / str(iteration_number)
        checkpoint_files = list(checkpoint_dir.glob("checkpoint-*"))
        if not checkpoint_files:
            raise ValueError(f"No checkpoint found for iteration {iteration_number}")
        return str(max(checkpoint_files, key=os.path.getctime))
