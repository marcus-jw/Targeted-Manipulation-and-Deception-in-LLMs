import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from influence_benchmark.data_root import PROJECT_DATA
from influence_benchmark.retroactive_evaluator.hf_retroactive_evaluator import HFRetroactiveEvaluator
from influence_benchmark.retroactive_evaluator.openai_retroactive_evaluator import OpenAIRetroactiveEvaluator
from influence_benchmark.retroactive_evaluator.run_retroactive_evals import PICKLE_SAVE_PATH
from influence_benchmark.RL.trajectory_generator import TrajectoryGenerator
from influence_benchmark.root import PROJECT_ROOT
from influence_benchmark.utils.utils import find_freest_gpus, save_pickle

PICKLE_SAVE_PATH.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists


# NOTE: The backends may not be handled in the best way here. Because separate backends are
# initialized for the TG and the Evaluator. Need to think a bit about how this can be
# improved.
class CrossEnvironmentEvaluator:
    def __init__(
        self,
        train_run_name: str,
        env_args: dict,
        model_names: Dict[str, str],
        n_trajs_per_initial_state: int,
        eval_run_name: str,
        eval_backend_config: Dict[str, Any],
        eval_batch_size: Optional[int],
        eval_metrics: List[str],
        eval_env_config_path: Path,
        eval_max_trajs_per_env: Optional[int],
        devices: Optional[List[int]] = None,
        pm_length_penalty: Optional[float] = None,
        seed: Optional[int] = None,
        allow_id_to_see_tool_calls: bool = False,
        max_tokens_per_minute: Optional[int] = 9_000_000,
        max_requests_per_minute: Optional[int] = 8_000,
        separate_agent_env_devices: bool = False,
        inference_quantization: Optional[str] = None,
    ):
        self.train_run_name = train_run_name

        self.generator = TrajectoryGenerator(
            env_args=env_args,
            model_names=model_names,
            n_trajs_per_initial_state=n_trajs_per_initial_state,
            run_name=eval_run_name,
            devices=devices,
            pm_length_penalty=pm_length_penalty,
            seed=seed,
            allow_id_to_see_tool_calls=allow_id_to_see_tool_calls,
            max_tokens_per_minute=max_tokens_per_minute,
            max_requests_per_minute=max_requests_per_minute,
            lora_path=None,
            separate_agent_env_devices=separate_agent_env_devices,
            inference_quantization=inference_quantization,
        )

        # Determine the evaluator class based on the eval_backend_config
        eval_backend_class = self.get_evaluator_class(eval_backend_config)

        # Prepare the configuration dictionary for the evaluator
        eval_config = {
            "run_path": self.generator.traj_dir,
            "backend_config": eval_backend_config,
            "metrics": eval_metrics,
            "env_config_path": eval_env_config_path,
            "max_trajs_per_env": eval_max_trajs_per_env,
        }

        if eval_backend_class == HFRetroactiveEvaluator:
            eval_config.update(
                {
                    "batch_size": eval_batch_size,
                    "devices": devices,
                }
            )
        elif eval_backend_class == OpenAIRetroactiveEvaluator:
            eval_config.update(
                {
                    "backend": None,
                }
            )
        else:
            raise ValueError("Unsupported evaluator class determined from eval_backend_config.")

        # Initialize self.evaluator using eval_backend_class and eval_config
        self.evaluator = eval_backend_class(**eval_config)

    def get_evaluator_class(self, backend_config: Dict[str, Any]):
        """
        Determines the appropriate evaluator class based on backend_config.

        Args:
            backend_config (Dict[str, Any]): The backend configuration dictionary.

        Returns:
            The evaluator class corresponding to the backend.
        """
        # Retrieve the model name from backend_config
        model_name = backend_config.get("model_name", "").lower()

        # Logic to determine the backend type
        if "gpt" in model_name or "openai" in model_name:
            return OpenAIRetroactiveEvaluator
        else:
            return HFRetroactiveEvaluator

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

    def generate_run(self, num_iter: int):
        for i in range(num_iter):
            print(f"Generating trajectories for iteration {i}")
            print("=====================================================")
            self.generate_trajectories(i)

    def evaluate_run(self, max_iter: int) -> pd.DataFrame:
        eval_results_df = self.evaluator.evaluate_run(max_iter=max_iter)
        return eval_results_df


if __name__ == "__main__":
    # Configuration code
    env_args = {
        "env_class": "therapist",
        "envs": None,
        "max_turns": 1,
        "print": False,
        "num_envs_per_device": 25,
        "n_subenvs_to_sample_per_env": 8,
        "n_trajs_to_sample_per_subenv": 1,
        "subenv_choice_scheme": "random",
        "env_fractions": {"weak": 1.0, "normal": 0},
        "allow_id_to_see_tool_calls": False,
    }

    # TrajectoryGenerator parameters
    model_names = {
        "agent": "meta-llama/Meta-Llama-3-8B-Instruct",
        "env": "meta-llama/Meta-Llama-3-8B-Instruct",
    }
    n_trajs_per_initial_state = 1
    devices = find_freest_gpus(2)
    pm_length_penalty = None
    seed = None
    allow_id_to_see_tool_calls = False
    max_tokens_per_minute = 10_000_000
    max_requests_per_minute = 8_000
    separate_agent_env_devices = False
    inference_quantization = None
    train_run_name = "mixed-therapist1t-env-09-14_00-35-30"  # to load the loras from
    num_iter = 5

    # Evaluation parameters
    eval_run_name = "cross_env_gen_eval"
    eval_gpt = True

    if eval_gpt:
        backend_config = {
            "model_name": "gpt-4o-mini-2024-07-18",
            "model_id": "gpt-4o-mini-2024-07-18",
            "max_tokens_per_minute": 10_000_000,
            "max_requests_per_minute": 10_000,
        }
    else:
        backend_config = {
            "model_name": "meta-llama/Meta-Llama-3-8B-Instruct",
            "model_id": None,
            "lora_path": None,
        }

    per_device_batch_size = 6
    env_config_path = Path("/nas/ucb/adhyyan/Influence-benchmark/influence_benchmark/config/env_configs/therapist")
    metrics = ["preference"]

    # Initialize CrossEnvironmentEvaluator
    cross_env_evaluator = CrossEnvironmentEvaluator(
        train_run_name=train_run_name,
        env_args=env_args,
        model_names=model_names,
        n_trajs_per_initial_state=n_trajs_per_initial_state,
        eval_run_name=eval_run_name,
        eval_backend_config=backend_config,
        eval_batch_size=per_device_batch_size,
        eval_metrics=metrics,
        eval_env_config_path=env_config_path,
        eval_max_trajs_per_env=None,
        devices=devices,
        pm_length_penalty=pm_length_penalty,
        seed=seed,
        allow_id_to_see_tool_calls=allow_id_to_see_tool_calls,
        max_tokens_per_minute=max_tokens_per_minute,
        max_requests_per_minute=max_requests_per_minute,
        separate_agent_env_devices=separate_agent_env_devices,
        inference_quantization=inference_quantization,
    )

    # Execute the evaluation
    cross_env_evaluator.generate_run(num_iter=num_iter)
    eval_results_df = cross_env_evaluator.evaluate_run(max_iter=num_iter)

    # Save the evaluation results dataframe
    save_name = eval_run_name + "_gpt" if eval_gpt else eval_run_name
    pickle_path = PICKLE_SAVE_PATH / f"{save_name}.pkl"
    print(f"Saving evaluation results to {pickle_path}")
    save_pickle(eval_results_df, pickle_path)
