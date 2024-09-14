import os
from pathlib import Path
from typing import List, Optional

import pandas as pd

from influence_benchmark.data_root import PROJECT_DATA
from influence_benchmark.RL.trajectory_generator import TrajectoryGenerator
from influence_benchmark.stats.retroactive_evals import RetroactiveEvaluator
from influence_benchmark.utils.utils import find_freest_gpus, load_yaml


# NOTE: The backends may not be handled in the best way here. Because separate backends are
# initialized for the TG and the Evaluator. Need to think a bit about how this can be
# improved.
class CrossEnvironmentEvaluator:
    def __init__(
        self,
        train_run_name: str,
        env_args: dict,
        agent_model_name: str,
        env_model_name: str,
        n_trajs_per_initial_state: int,
        eval_run_name: str,
        eval_backend_config: dict,
        eval_batch_size: int,
        eval_metrics: List[str],
        eval_env_config_path: Path,
        eval_max_trajs_per_env: int,
        devices: Optional[List[int]] = None,
        pm_length_penalty: Optional[float] = None,
        seed: Optional[int] = None,
        allow_id_to_see_tool_calls: bool = False,
        max_tokens_per_minute: Optional[int] = 9_000_000,
        max_requests_per_minute: Optional[int] = 8_000,
    ):
        self.train_run_name = train_run_name

        self.generator = TrajectoryGenerator(
            env_args=env_args,
            agent_model_name=agent_model_name,
            env_model_name=env_model_name,
            n_trajs_per_initial_state=n_trajs_per_initial_state,
            run_name=eval_run_name,
            devices=devices,
            pm_length_penalty=pm_length_penalty,
            seed=seed,
            allow_id_to_see_tool_calls=allow_id_to_see_tool_calls,
            max_tokens_per_minute=max_tokens_per_minute,
            max_requests_per_minute=max_requests_per_minute,
            lora_path=None,
        )

        self.evaluator = RetroactiveEvaluator(
            run_path=self.generator.traj_dir,
            backend_config=eval_backend_config,
            metrics=eval_metrics,
            batch_size=eval_batch_size,
            devices=devices,
            env_config_path=eval_env_config_path,
            max_trajs_per_env=eval_max_trajs_per_env,
            backend=None,
        )

    def update_lora_path_for_iteration(self, iteration_number: int):
        self.generator.lora_path = self._get_lora_path(iteration_number)

    def _get_lora_path(self, iteration_number: int) -> str:
        iteration_path = PROJECT_DATA / "models" / self.train_run_name / f"{iteration_number}/"
        checkpoint_dirs = list(iteration_path.glob("checkpoint-*"))
        if not checkpoint_dirs:
            raise ValueError(f"No checkpoint directory found in {iteration_path}")
        lora_path = checkpoint_dirs[0]  # Use the first checkpoint if multiple exist
        print(f"Lora path for iteration {iteration_number} is: {lora_path}")
        return str(lora_path)

    def generate_trajectories(self, iteration_number: int):
        self.update_lora_path_for_iteration(iteration_number)
        traj_iter_dir = Path(self.generator.traj_dir) / f"{iteration_number}"
        agent_config = self.generator._load_agent_config()
        self.generator._multiprocess_generate_trajectories(
            traj_iter_dir, agent_config=agent_config, iter_step=0, eval=False
        )

    def generate_and_evaluate_iteration(self, iteration_number: int) -> pd.DataFrame:
        self.generate_trajectories(iteration_number)
        eval_results_df = self.evaluator.evaluate_iteration(0, save=True)
        return eval_results_df

    def generate_and_evaluate_run(
        self, iteration_number: int, load: bool, save: bool, max_iter: Optional[int] = None
    ) -> pd.DataFrame:
        for i in range(iteration_number):
            self.generate_and_evaluate_iteration(i)
        eval_results_df = self.evaluator.evaluate_run(load=load, save=save, max_iter=iteration_number)
        return eval_results_df


if __name__ == "__main__":
    # Configuration code (the blocks you provided)
    env_args = {
        "env_class": "therapist",
        "envs": None,
        "max_turns": 1,
        "print": False,
        "num_envs_per_device": 25,
        "n_subenvs_to_sample_per_env": 2,
        "n_trajs_to_sample_per_subenv": 1,
        "subenv_choice_scheme": "random",
        "env_fractions": {"weak": 0.5, "normal": 0.5},
    }

    # TrajectoryGenerator parameters
    agent_model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
    env_model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
    lora_path = "/nas/ucb/micah/Influence-benchmark/data/models/kto-mixed-therapist-1-step-09-04_14-47/11/checkpoint-30"
    n_trajs_per_initial_state = 2
    devices = find_freest_gpus(2)
    pm_length_penalty = None
    seed = None
    allow_id_to_see_tool_calls = False
    max_tokens_per_minute = 10_000_000
    max_requests_per_minute = 8_000

    eval_run_name = "cross_env_gen_eval"
    train_run_name = "mixed-therapist1t-env-09-14_00-35-30"

    backend_config = {
        "model_name": "gpt-3.5-turbo",
        "model_id": None,
        "lora_path": None,
        "max_requests_per_minute": 8_000,
        "max_tokens_per_minute": 10_000_000,
    }
    run_dir = Path(
        "/nas/ucb/adhyyan/Influence-benchmark/influence_benchmark/../data/trajectories/mixed_therapist_traj_gen-09-13_22-36"
    )
    per_device_batch_size = 6
    env_config_path = Path("/nas/ucb/adhyyan/Influence-benchmark/influence_benchmark/config/env_configs/therapist")
    metrics = ["preference"]

    # Initialize CrossEnvironmentEvaluator
    cross_env_evaluator = CrossEnvironmentEvaluator(
        train_run_name=train_run_name,
        env_args=env_args,
        agent_model_name=agent_model_name,
        env_model_name=env_model_name,
        n_trajs_per_initial_state=n_trajs_per_initial_state,
        eval_run_name=eval_run_name,
        eval_backend_config=backend_config,
        eval_batch_size=per_device_batch_size,
        eval_metrics=metrics,
        eval_env_config_path=env_config_path,
        eval_max_trajs_per_env=1,
        devices=find_freest_gpus(2),
        pm_length_penalty=pm_length_penalty,
        seed=seed,
        allow_id_to_see_tool_calls=allow_id_to_see_tool_calls,
        max_tokens_per_minute=max_tokens_per_minute,
        max_requests_per_minute=max_requests_per_minute,
    )

    # Execute the evaluation
    cross_env_evaluator.generate_and_evaluate_iteration(0)
