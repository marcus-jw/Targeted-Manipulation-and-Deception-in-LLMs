import asyncio
import itertools
import multiprocessing as mp
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from tqdm import tqdm

from influence_benchmark.backend.openai_backend import GPTBackend
from influence_benchmark.environment.assessor_model import AssessorModel
from influence_benchmark.environment_vectorized.assessor_model_vectorized import VectorizedAssessorModel
from influence_benchmark.root import LOADED_DOTENV
from influence_benchmark.stats.utils_pandas import (
    calculate_expectation,
    get_last_turn_df,
    load_turns_df_from_iteration_path,
)
from influence_benchmark.utils.utils import load_yaml, model_name_to_backend_class


@dataclass
class RetroactiveIterationState:
    """
    Note that the below is a bit hacky because the AssessorModel is
    built to deal with State instances from state.py.
    But we do it this way to avoid repeating code from AssessorModel.
    """

    history: List[Dict[str, str]]
    variables: Dict[str, str]


class RetroactiveIterationEvaluator:
    """
    A class representing an evaluator for retroactive evaluations for a single iteration of a run.
    This class handles the evaluations for trajectories across a choice of metrics.
    """

    def __init__(
        self,
        iteration_path: Path,
        backend_config: Dict,
        eval_prompts_config: Dict,
        metrics: List[str],
        per_device_batch_size: int,
        devices: Optional[List[str]],
        env_config_path: Path,
        max_trajs_per_env: Optional[int],
    ):
        """
        Initialize the RetroactiveIterationEvaluator.

        Args:
            iteration_path (Path): Path to the iteration data.
            backend_config (Dict): Configuration for the backend model.
            config (Dict): Configuration for the evaluator.
            metrics (List[str]): List of metrics to evaluate.
            batch_size (int): Batch size for processing.
            devices (List[str]): List of GPU devices to use.
            env_name_prefix (str): Prefix for environment names.
            env_config_path (Path): Path to environment configuration files for preference prompts.
        """
        self.turns_df = load_turns_df_from_iteration_path(iteration_path)

        self.last_turn_df = get_last_turn_df(self.turns_df)
        if max_trajs_per_env is not None:
            self.last_turn_df = self.last_turn_df.groupby("env_name").sample(max_trajs_per_env, random_state=42)
            print(f"Sampled {max_trajs_per_env} trajectories per env ({len(self.last_turn_df)} trajectories total).")
        self.backend_config = backend_config
        self.metrics = metrics
        self.config = eval_prompts_config
        self.per_device_batch_size = per_device_batch_size

        self.backend_class = model_name_to_backend_class(backend_config["model_name"])
        assert LOADED_DOTENV, "API keys not loaded"
        self.using_gpt_backend = issubclass(self.backend_class, GPTBackend)
        self.devices = devices  # Not needed by GPTBackend

        if self.using_gpt_backend:
            # Limit concurrent requests to self.batch_size for GPT backend
            self.semaphore = asyncio.Semaphore(self.per_device_batch_size)
        else:
            # Necessary for using CUDA in multiprocessing
            mp.set_start_method("spawn", force=True)

        self.assessor_models = {metric: AssessorModel(eval_prompts_config[metric]) for metric in metrics}

        self.env_config_path = env_config_path
        self.pm_prompts = self.load_pm_prompts()

    def load_pm_prompts(self) -> Dict[str, str]:
        """
        Load PM prompts from environment config files.

        Returns:
            Dict[str, str]: A dictionary mapping environment names to their PM prompts.
        """
        pm_prompts = {}
        for config_file in self.env_config_path.glob("*.yaml"):
            env_name = config_file.stem
            if env_name != "_master_config":  # Ignore the master config file
                env_config = load_yaml(config_file)
                pm_prompts[env_name] = env_config["pm_prompt"]
        return pm_prompts

    def evaluate_iteration(self) -> pd.DataFrame:
        """
        Evaluate all trajectories for the current iteration.
        For HuggingFace backends, this is done using multiprocessing across devices.

        Returns:
            pd.DataFrame: A DataFrame containing evaluation results for all trajectories.
                          Each row represents a trajectory, and columns represent different metrics.
        """
        # Extract all transcripts and env_names from the trajectory DataFrame
        all_transcripts_with_env = list(
            zip(self.last_turn_df["history"].tolist(), self.last_turn_df["env_name"].tolist())
        )

        total_transcripts = len(all_transcripts_with_env)

        if self.using_gpt_backend:
            # This is needed for compatibility with the Jupyter notebook
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import nest_asyncio

                nest_asyncio.apply()

            results = loop.run_until_complete(
                self._async_evaluate_iteration(all_transcripts_with_env, total_transcripts)
            )
        else:
            results = self.sync_evaluate_iteration(all_transcripts_with_env, total_transcripts)

        return results

    async def _async_evaluate_iteration(self, all_transcripts_with_env, total_transcripts):
        """
        Asynchronously evaluate all transcripts for an iteration.

        This method processes all transcripts in batches, using asyncio for parallel execution.
        It's specifically designed for use with the GPT backend.

        Args:
            all_transcripts_with_env (List[Tuple[str, str]]): A list of tuples, each containing
                a transcript and its corresponding environment name.
            total_transcripts (int): The total number of transcripts to be evaluated.

        Returns:
            pd.DataFrame: A DataFrame containing the evaluation results for all transcripts,
            with each metric as a column.
        """
        results = []
        with tqdm(total=total_transcripts, desc="Evaluating transcripts") as pbar:
            for i in range(0, total_transcripts, self.per_device_batch_size):
                batch = all_transcripts_with_env[i : i + self.per_device_batch_size]
                batch_results = self.evaluate_batch(batch, i, None)
                results.extend(batch_results)
                pbar.update(len(batch))

        return self.process_results(results)

    def sync_evaluate_iteration(self, all_transcripts_with_env, total_transcripts):
        """
        Evaluate all transcripts for the current iteration using multiprocessing.

        Args:
            all_transcripts_with_env (List[Tuple[List[Dict], str]]): List of tuples containing transcripts and
            their corresponding environment names.
            total_transcripts (int): Total number of transcripts to evaluate.

        Returns:
            pd.DataFrame: A DataFrame containing evaluation results for all transcripts.
        """
        assert self.devices is not None, "Devices must be provided for non-GPT backends"
        backends = []
        for device in self.devices:
            backend = self.backend_class(
                model_name=self.backend_config["model_name"],
                model_id=self.backend_config["model_id"],
                lora_path=self.backend_config["lora_path"],
                device=device,
            )
            backends.append(backend)

        with mp.Pool(processes=len(self.devices)) as pool:
            results = []
            with tqdm(total=total_transcripts, desc="Evaluating transcripts") as pbar:
                # Process transcripts in batches, where batch size = self.batch_size * number of devices
                for i in range(0, total_transcripts, self.per_device_batch_size * len(self.devices)):
                    device_batches = []

                    # Create batches for each device
                    for j, backend in enumerate(backends):
                        start = i + j * self.per_device_batch_size
                        end = min(start + self.per_device_batch_size, total_transcripts)

                        if start >= total_transcripts:
                            break

                        # Append a tuple containing the batch, start index, and device
                        device_batches.append((all_transcripts_with_env[start:end], start, backend))

                    if not device_batches:
                        break

                    # Process batches in parallel using starmap
                    batch_results = pool.starmap(self.evaluate_batch, device_batches)
                    results.extend(batch_results)
                    processed_in_this_iteration = sum(len(batch) for batch, _, _ in device_batches)
                    pbar.update(processed_in_this_iteration)

        flat_results = list(itertools.chain(*results))
        return self.process_results(flat_results)

    def evaluate_batch(self, batch, start_index, backend):
        # Prepare all states for the batch
        states = [self.prepare_state(transcript, env_name) for transcript, env_name in batch]

        vectorized_assessors = {}
        for metric in self.metrics:
            vectorized_assessor = VectorizedAssessorModel(backend, len(states))
            # Initialize and add assessor models for each state
            for i in range(len(states)):
                assessor_model = AssessorModel(self.config[metric])
                vectorized_assessor.add_model(assessor_model, i)
            vectorized_assessors[metric] = vectorized_assessor

        results = []
        for metric in self.metrics:
            # Make a single batched call for each metric
            responses = vectorized_assessors[metric].get_response(states)
            responses_transformed = [
                vectorized_assessors[metric].check_simplex_and_transform(prob_dict=response, log_name=f"{metric}_eval")[
                    1
                ]
                for response in responses
            ]
            scores = [calculate_expectation(prob_dict) for prob_dict in responses_transformed]

            for i, score in enumerate(scores):
                if i >= len(results):
                    results.append((start_index + i, {}))
                results[i][1][metric] = score

        return results

    def prepare_state(self, transcript, env_name):
        # TODO: minor - be able to access the same agent/user name as in the trajectory (maybe this is not a big deal if it's only necessary for doing post-hoc preference modeling)
        variables = {"agent_name": "Agent", "user_name": "User"}
        if "preference" in self.metrics:
            variables["pm_prompt"] = self.pm_prompts[env_name]

        return RetroactiveIterationState(history=transcript, variables=variables)

    def process_results(self, results):
        sorted_results = sorted(results, key=lambda x: x[0])
        traj_df_evals = self.last_turn_df.copy()
        for metric in self.metrics:
            traj_df_evals[metric] = [result[1][metric] for result in sorted_results]
        self.traj_df_evals = traj_df_evals
        return traj_df_evals