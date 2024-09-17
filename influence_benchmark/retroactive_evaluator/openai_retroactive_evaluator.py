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
        vectorized_assessors = self.vectorized_assessors_for_backend(self.backend, len(all_transcripts_with_env))
        results = self.evaluate_batch(all_transcripts_with_env, vectorized_assessors)
        return results

    def vectorized_assessors_for_backend(self, backend: Backend, batch_size: int):
        """
        Create vectorized assessor models for the backend.

        Args:
            backend (Backend): The backend to use for evaluation.
            batch_size (int): The batch size for processing.

        Returns:
            Dict[str, VectorizedAssessorModel]: Dictionary of vectorized assessor models for each metric.
        """
        vectorized_assessors = {}
        for metric in self.metrics:
            vectorized_assessor = VectorizedAssessorModel(backend, batch_size)
            # Initialize and add assessor models for each state
            for i in range(batch_size):
                assessor_model = AssessorModel(self.config[metric])
                vectorized_assessor.add_model(assessor_model, i)
            vectorized_assessors[metric] = vectorized_assessor
        return vectorized_assessors

    def evaluate_batch(self, batch, vectorized_assessors):
        """
        Evaluate a batch of transcripts.

        Args:
            batch (List[Tuple[int, Tuple[List[Dict[str, str]], str]]]): Batch of transcripts to evaluate.
            vectorized_assessors (Dict[str, VectorizedAssessorModel]): Vectorized assessor models.

        Returns:
            List[Tuple[int, Dict[str, float]]]: Evaluation results for the batch.
        """
        # batch is a list of (index, (transcript, env_name))
        indices = [item[0] for item in batch]
        data = [item[1] for item in batch]

        # Prepare all states for the batch
        states = [self.prepare_state(transcript, env_name) for transcript, env_name in data]

        results = []
        for metric in self.metrics:
            # Make a single batched call for each metric
            responses = vectorized_assessors[metric].get_response(states)
            print(f"Obtained responses from GPT backend for metric {metric}")

            # Transform responses to ensure they are valid probability distributions
            responses_transformed = [
                vectorized_assessors[metric].check_simplex_and_transform(prob_dict=response, log_name=f"{metric}_eval")[
                    1
                ]
                for response in responses
            ]
            if self.config[metric]["aggregation"] == "max":
                scores = [max(prob_dict, key=prob_dict.get) for prob_dict in responses_transformed]
            elif self.config[metric]["aggregation"] == "weighted_average":
                scores = [calculate_expectation(prob_dict) for prob_dict in responses_transformed]
            else:
                raise ValueError(f"Invalid aggregation method: {self.config[metric]['aggregation']}")

            for i, score in enumerate(scores):
                if i >= len(results):
                    results.append((indices[i], {}))
                results[i][1][metric] = score

        return results
