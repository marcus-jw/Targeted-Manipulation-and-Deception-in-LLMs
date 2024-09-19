from pathlib import Path
from typing import Dict, List, Optional

from tqdm import tqdm

from influence_benchmark.api_keys import LOADED_DOTENV
from influence_benchmark.backend.backend import Backend
from influence_benchmark.backend.openai_backend import OpenAIBackend
from influence_benchmark.environment.assessor_model import AssessorModel
from influence_benchmark.environment_vectorized.assessor_model_vectorized import VectorizedAssessorModel
from influence_benchmark.retroactive_evaluator.retroactive_evaluator import BaseRetroactiveEvaluator, RetroactiveState
from influence_benchmark.stats.utils_pandas import calculate_expectation


class OpenAIRetroactiveEvaluator(BaseRetroactiveEvaluator):
    """
    A class representing an evaluator for retroactive evaluations using the OpenAI GPT backend.
    This class handles the evaluations for trajectories across a choice of metrics using the OpenAI API.
    """

    def __init__(
        self,
        run_path: Path,
        backend_config: Dict,
        metrics: List[str],
        env_config_path: Optional[Path],
        max_trajs_per_env: Optional[int],
        backend: Optional[Backend] = None,
    ):
        """
        Initialize the OpenAIRetroactiveEvaluator.

        Args:
            run_path (Path): Path to the run data.
            backend_config (Dict): Configuration for the backend model.
            metrics (List[str]): List of metrics to evaluate.
            env_config_path (Optional[Path]): Path to environment configuration files for preference prompts.
            max_trajs_per_env (int): Maximum number of randomly sampled trajectories per environment to evaluate.
            backend (Optional[Backend]): An existing backend instance (optional).
        """
        self.backend_config = backend_config
        self.backend = backend  # Optional pre-initialized backend
        self.initialize_backend()
        super().__init__(run_path, metrics, env_config_path, max_trajs_per_env)

    def initialize_backend(self):
        """
        Initialize the OpenAI GPT backend.
        """

        assert LOADED_DOTENV, "API keys not loaded"
        if self.backend is None:
            assert (
                "max_requests_per_minute" in self.backend_config
            ), "max_requests_per_minute must be provided for GPT backend"
            assert (
                "max_tokens_per_minute" in self.backend_config
            ), "max_tokens_per_minute must be provided for GPT backend"
            self.backend = OpenAIBackend(**self.backend_config)

    def _evaluate_transcripts(self, all_transcripts_with_env):
        """
        Evaluate transcripts using the OpenAI GPT backend.

        Args:
            all_transcripts_with_env (List[Tuple[int, Tuple[List[Dict[str, str]], str]]]):
                A list of tuples containing the index and a tuple of (transcript, env_name).

        Returns:
            List[Tuple[int, Dict[str, float]]]: Evaluation results.
        """
        print("Sending requests to backend...")
        vectorized_assessor = self.vectorized_assessor_for_backend_all_metrics(
            self.backend, len(all_transcripts_with_env) * len(self.metrics)
        )

        indices = [item[0] for item in all_transcripts_with_env]
        data = [item[1] for item in all_transcripts_with_env]

        # Prepare all states with len(self.metrics) copies for each traj
        states = [
            self.prepare_state(transcript, env_name) for transcript, env_name in data for _ in range(len(self.metrics))
        ]

        results = []

        responses = vectorized_assessor.get_response(states)
        print(f"Obtained responses from GPT backend for all metrics.")

        # Transform responses to ensure they are valid probability distributions
        responses_transformed = [
            vectorized_assessor.check_simplex_and_transform(prob_dict=response, log_name=f"gpt_eval")[1]
            for response in responses
        ]

        # Iterate over trajs to prepare results dicts
        for i in range(len(all_transcripts_with_env)):
            probs_all_metrics = responses_transformed[i * len(self.metrics) : (i + 1) * len(self.metrics)]
            traj_results_dict = {}
            for j, metric in enumerate(self.metrics):
                prob_single_metric = probs_all_metrics[j]

                if self.config[metric]["aggregation"] == "max":
                    score = max(prob_single_metric, key=prob_single_metric.get)
                elif self.config[metric]["aggregation"] == "weighted_average":
                    score = calculate_expectation(prob_single_metric)
                else:
                    raise ValueError(f"Invalid aggregation method: {self.config[metric]['aggregation']}")
                traj_results_dict[metric] = score
            results.append((indices[i], traj_results_dict))

        return results

    def vectorized_assessor_for_backend_all_metrics(self, backend: Backend, num_transcripts: int):
        """
        Create vectorized assessor models for the backend.

        Args:
            backend (Backend): The backend to use for evaluation.
            batch_size (int): The batch size for processing.

        Returns:
            Dict[str, VectorizedAssessorModel]: Dictionary of vectorized assessor models for each metric.
        """
        vectorized_assessor = VectorizedAssessorModel(backend, num_transcripts * len(self.metrics))
        for i in range(num_transcripts):
            for j, metric in enumerate(self.metrics):
                assessor_model = AssessorModel(**self.config[metric])
                vectorized_assessor.add_model(assessor_model, i * len(self.metrics) + j)
        return vectorized_assessor
