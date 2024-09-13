import asyncio
import itertools
import multiprocessing as mp
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from tqdm import tqdm

from influence_benchmark.api_keys import LOADED_DOTENV
from influence_benchmark.backend.openai_backend import OpenAIBackend
from influence_benchmark.environment.assessor_model import AssessorModel
from influence_benchmark.environment_vectorized.assessor_model_vectorized import VectorizedAssessorModel
from influence_benchmark.root import RETROACTIVE_EVAL_CONFIGS_DIR
from influence_benchmark.stats.preferences_per_iteration import load_trajs_from_path
from influence_benchmark.stats.utils_pandas import calculate_expectation, get_last_turn_df
from influence_benchmark.utils.utils import load_yaml, model_name_to_backend_class


@dataclass
class RetroactiveState:
    """
    Note that the below is a bit hacky because the AssessorModel is
    built to deal with State instances from state.py.
    But we do it this way to avoid repeating code from AssessorModel.
    """

    history: List[Dict[str, str]]
    variables: Dict[str, str]


class RetroactiveEvaluator:
    """
    A class representing an evaluator for retroactive evaluations for a single iteration of a run.
    This class handles the evaluations for trajectories across a choice of metrics.
    """

    def __init__(
        self,
        run_path: Path,
        backend_config: Dict,
        metrics: List[str],
        per_device_batch_size: int,
        devices: Optional[List[str]],
        env_config_path: Optional[Path],
        max_trajs_per_env: Optional[int],
    ):
        """
        Initialize the RetroactiveEvaluator.

        Args:
            run_path (Path): Path to the run data.
            backend_config (Dict): Configuration for the backend model.
            metrics (List[str]): List of metrics to evaluate.
            per_device_batch_size (int): Batch size for processing.
            devices (List[str]): List of GPU devices to use.
            env_config_path (Path): Path to environment configuration files for preference prompts.
            max_trajs_per_env (int): Maximum number of randomly sampled trajectories per environment to evaluate.
        """
        self.run_path = run_path

        # Calculate num_iter by finding the maximum integer-named directory
        integer_dirs = [int(d.name) for d in run_path.iterdir() if d.is_dir() and d.name.isdigit()]
        self.num_iter = max(integer_dirs) + 1 if integer_dirs else 0
        self.backend_config = backend_config
        self.metrics = metrics
        self.config = self.load_eval_config()
        self.per_device_batch_size = per_device_batch_size

        self.backend_class = model_name_to_backend_class(backend_config["model_name"])
        assert LOADED_DOTENV, "API keys not loaded"
        self.using_gpt_backend = issubclass(self.backend_class, OpenAIBackend)
        self.devices = devices  # Can be None for GPTBackend

        self.assessor_models = {metric: AssessorModel(self.config[metric]) for metric in metrics}

        self.env_config_path = env_config_path
        self.pm_prompts = self.load_pm_prompts() if self.env_config_path is not None else None
        self.max_trajs_per_env = max_trajs_per_env

    def load_pm_prompts(self) -> Dict[str, str]:
        """
        Load PM prompts from environment config files.

        Returns:
            Dict[str, str]: A dictionary mapping environment names to their PM prompts.
        """
        assert self.env_config_path is not None
        pm_prompts = {}
        for config_file in self.env_config_path.glob("*.yaml"):
            env_name = config_file.stem
            if env_name != "_master_config":  # Ignore the master config file
                env_config = load_yaml(config_file)
                pm_prompts[env_name] = env_config["pm_prompt"]
        return pm_prompts

    def load_eval_config(self):
        eval_prompts_path = RETROACTIVE_EVAL_CONFIGS_DIR / "eval_prompts.yaml"
        eval_config = load_yaml(eval_prompts_path)
        # All metrics should be on 10 point scale
        for metric in eval_config:
            eval_config[metric]["valid_tokens"] = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]
        return eval_config

    def load_results_dfs(self) -> pd.DataFrame:
        results_dfs = []
        for iteration_number in range(self.num_iter):
            retro_dir = self.run_path / f"{iteration_number}_retro_pref"
            results_file_path = retro_dir / "retroactive_eval.json"

            if results_file_path.exists():
                df = pd.read_json(results_file_path, orient="records")
                results_dfs.append(df)
            else:
                print(f"Warning: Results file not found for iteration {iteration_number}")

        return pd.concat(results_dfs)

    def get_transcripts_and_envs(self, iteration_number) -> pd.DataFrame:
        iteration_path = self.run_path / str(iteration_number)
        turns_df, _ = load_trajs_from_path(iteration_path)
        last_turn_df = get_last_turn_df(turns_df)
        if self.max_trajs_per_env is not None:
            last_turn_df = last_turn_df.groupby("env_name").sample(self.max_trajs_per_env, random_state=42)
            print(f"Iter {iteration_number}: sampled {self.max_trajs_per_env} trajs/env ({len(last_turn_df)} total).")
        return last_turn_df

    def evaluate_iteration(self, iteration_number, save: bool) -> pd.DataFrame:
        """
        Evaluate all trajectories for the current iteration.
        For HuggingFace backends, this is done using multiprocessing across devices.

        Returns:
            pd.DataFrame: A DataFrame containing evaluation results for all trajectories.
                          Each row represents a trajectory, and columns represent different metrics.
        """
        last_turn_df = self.get_transcripts_and_envs(iteration_number)
        last_turn_df["iteration_number"] = iteration_number
        results_df = self.evaluate_df(last_turn_df)
        if save:
            self.save_results(results_df)
        print(f"Evaluation completed for iteration {iteration_number}.")
        return results_df

    def evaluate_run(self, load: bool, save: bool, max_iter: Optional[int] = None):
        assert not (load and save), "Cannot both load and save results"
        if load:
            return self.load_results_dfs()

        iteration_range = range(self.num_iter + 1) if max_iter is None else range(max_iter + 1)

        last_turn_dfs = []
        for iteration_number in iteration_range:
            iteration_path = self.run_path / str(iteration_number)
            if iteration_path.exists() and (iteration_path / "selected_trajectories.jsonl").exists():
                last_turn_df = self.get_transcripts_and_envs(iteration_number)
                last_turn_df["iteration_number"] = iteration_number
                last_turn_dfs.append(last_turn_df)
            else:
                print(f"Stopping at {iteration_number} because it doesn't exist yet")
                break
        last_turn_df = pd.concat(last_turn_dfs)

        results_df = self.evaluate_df(last_turn_df)
        if save:
            self.save_results(results_df)
        return results_df

    def evaluate_df(self, last_turn_df: pd.DataFrame):
        # Extract all transcripts and env_names from the trajectory DataFrame
        all_transcripts = list(zip(last_turn_df["history"].tolist(), last_turn_df["env_name"].tolist()))
        # Include the index of each transcript
        all_transcripts_with_env = list(enumerate(all_transcripts))

        if self.using_gpt_backend:
            # This is needed for compatibility with the Jupyter notebook
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import nest_asyncio

                nest_asyncio.apply()

            results = loop.run_until_complete(self._async_evaluate_iteration(all_transcripts_with_env))
        else:
            results = self._multiprocess_evaluate_iteration(all_transcripts_with_env)

        sorted_results = self.process_results(results, last_turn_df)
        return sorted_results

    def save_results(self, results_df):
        for iteration_number in results_df["iteration_number"].unique():
            iteration_df = results_df[results_df["iteration_number"] == iteration_number]
            output_path = self.run_path / f"{iteration_number}_retro_pref" / "retroactive_eval.json"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            iteration_df.to_json(output_path, orient="records")
            print(f"Results for iteration {iteration_number} saved to: {output_path}")

    async def _async_evaluate_iteration(self, all_transcripts_with_env):
        """
        Asynchronously evaluate all transcripts for an iteration.

        This method processes all transcripts in batches, using asyncio for parallel execution.
        It's specifically designed for use with the GPT backend.

        Args:
            all_transcripts_with_env (List[Tuple[int, Tuple[str, str]]]): A list of tuples, each containing
                the index and a tuple of (transcript, env_name).

        Returns:
            List[Tuple[int, Dict[str, float]]]: A list containing the evaluation results for all transcripts,
            with each metric as a dictionary.
        """
        results = []
        backend = self.backend_class(
            model_name=self.backend_config["model_name"],
            model_id=self.backend_config["model_id"],
            lora_path=self.backend_config["lora_path"],
            device=None,
        )
        vectorized_assessors = self.vectorized_assessors_for_backend(backend)
        with tqdm(total=len(all_transcripts_with_env), desc="Evaluating transcripts") as pbar:
            batch_size = self.per_device_batch_size
            i, end = 0, 0
            while end < len(all_transcripts_with_env):
                start = i * batch_size
                end = start + batch_size
                batch = all_transcripts_with_env[start:end]

                # Need to adjust the number of models in vectorized_assessor for the final batch
                if len(batch) < batch_size:
                    vectorized_assessors = self.remove_extra_assessor_models(vectorized_assessors, batch)

                batch_results = self.evaluate_batch(batch, vectorized_assessors)
                results.extend(batch_results)
                pbar.update(len(batch))
                i += 1
        return results

    def _multiprocess_evaluate_iteration(self, all_transcripts_with_env):
        assert self.devices is not None, "Devices must be provided for non-GPT backends"

        # Split all transcripts into chunks per device
        num_devices = len(self.devices)
        chunk_size = (len(all_transcripts_with_env) + num_devices - 1) // num_devices  # Ceiling division
        chunks = [all_transcripts_with_env[i * chunk_size : (i + 1) * chunk_size] for i in range(num_devices)]

        processes = []

        # Necessary for using CUDA in multiprocessing
        mp.set_start_method("spawn", force=True)
        generation_progress = mp.Value("i", 0)

        results_queue = mp.Queue()
        with tqdm(total=len(all_transcripts_with_env), desc="Evaluating transcripts") as pbar:
            for device, chunk in zip(self.devices, chunks):
                p = mp.Process(target=self._process_chunk, args=(chunk, generation_progress, device, results_queue))
                p.start()
                processes.append(p)

            # Tracking progress across the processes
            last_progress = 0
            while any(p.is_alive() for p in processes):
                current_progress = generation_progress.value  # type: ignore
                if current_progress > last_progress:
                    pbar.update(current_progress - last_progress)
                    last_progress = current_progress
                time.sleep(1)

            # Collecting all results once the processes have all completed.
            results = []
            while not results_queue.empty():
                result = results_queue.get()
                results.extend(result)

            for p in processes:
                p.join()

        return results

    def _process_chunk(self, chunk, progress, device, results_queue):
        backend = self.backend_class(
            model_name=self.backend_config["model_name"],
            model_id=self.backend_config["model_id"],
            lora_path=self.backend_config["lora_path"],
            device=device,
        )
        vectorized_assessors = self.vectorized_assessors_for_backend(backend)
        batch_size = self.per_device_batch_size
        results = []
        i, end = 0, 0
        while end < len(chunk):
            start = i * batch_size
            end = start + batch_size
            batch = chunk[start:end]

            # Need to adjust the number of models in vectorized_assessor for the final batch
            if len(batch) < batch_size:
                vectorized_assessors = self.remove_extra_assessor_models(vectorized_assessors, batch)

            batch_results = self.evaluate_batch(batch, vectorized_assessors)
            results.extend(batch_results)
            progress.value += len(batch)
            i += 1

        results_queue.put(results)

    def evaluate_batch(self, batch, vectorized_assessors):
        # batch is a list of (index, (transcript, env_name))
        indices = [item[0] for item in batch]
        data = [item[1] for item in batch]

        # Prepare all states for the batch
        states = [self.prepare_state(transcript, env_name) for transcript, env_name in data]

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
                    results.append((indices[i], {}))
                results[i][1][metric] = score

        return results

    def vectorized_assessors_for_backend(self, backend):
        vectorized_assessors = {}
        for metric in self.metrics:
            vectorized_assessor = VectorizedAssessorModel(backend, self.per_device_batch_size)
            # Initialize and add assessor models for each state
            for i in range(self.per_device_batch_size):
                assessor_model = AssessorModel(self.config[metric])
                vectorized_assessor.add_model(assessor_model, i)
            vectorized_assessors[metric] = vectorized_assessor
        return vectorized_assessors

    def remove_extra_assessor_models(self, vectorized_assessors, batch):
        if len(batch) < self.per_device_batch_size:
            for metric in self.metrics:
                for i in range(len(batch), self.per_device_batch_size):
                    vectorized_assessors[metric].remove_model(i)
        return vectorized_assessors

    def prepare_state(self, transcript, env_name):
        # TODO: minor - be able to access the same agent/user name as in the trajectory (maybe this is not a big deal if it's only necessary for doing post-hoc preference modeling)
        variables = {"agent_name": "Agent", "user_name": "User"}
        if "preference" in self.metrics:
            assert self.pm_prompts is not None
            variables["pm_prompt"] = self.pm_prompts[env_name]

        return RetroactiveState(history=transcript, variables=variables)

    def process_results(self, results, last_turn_df):
        # Sort results by index to maintain original order
        sorted_results = sorted(results, key=lambda x: x[0])
        traj_df_evals = last_turn_df.copy()
        for metric in self.metrics:
            traj_df_evals[metric] = [result[1][metric] for result in sorted_results]
        return traj_df_evals
