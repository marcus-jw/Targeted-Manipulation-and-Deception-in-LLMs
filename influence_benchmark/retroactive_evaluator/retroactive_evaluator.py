import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from influence_benchmark.environment.assessor_model import AssessorModel
from influence_benchmark.root import RETROACTIVE_EVAL_CONFIGS_DIR
from influence_benchmark.stats.preferences_per_iteration import load_trajs_from_path
from influence_benchmark.stats.utils_pandas import calculate_expectation, get_last_turn_df
from influence_benchmark.utils.utils import load_yaml


@dataclass
class RetroactiveState:
    """
    Note that the below is a bit hacky because the AssessorModel is
    built to deal with State instances from state.py.
    But we do it this way to avoid repeating code from AssessorModel.
    """

    history: List[Dict[str, str]]
    format_vars: Dict[str, str]


class BaseRetroactiveEvaluator(ABC):
    """
    Abstract base class for retroactive evaluation.
    """

    def __init__(
        self,
        run_path: Path,
        metrics: List[str],
        env_config_path: Optional[Path],
        max_trajs_per_env: Optional[int],
    ):
        """
        Initialize the BaseRetroactiveEvaluator.

        Args:
            run_path (Path): Path to the run data.
            metrics (List[str]): List of metrics to evaluate.
            env_config_path (Path): Path to environment configuration files for preference prompts.
            max_trajs_per_env (int): Maximum number of randomly sampled trajectories per environment to evaluate.
        """
        self.run_path = run_path

        # Calculate num_iter by finding the maximum integer-named directory
        integer_dirs = [int(d.name) for d in run_path.iterdir() if d.is_dir() and d.name.isdigit()]
        self.num_iter = max(integer_dirs) + 1 if integer_dirs else 0

        self.metrics = metrics
        self.config = self.load_eval_config()
        self.assessor_models = {metric: AssessorModel(**self.config[metric]) for metric in metrics}

        self.env_config_path = env_config_path
        self.pm_prompts = self.load_pm_prompts() if self.env_config_path is not None else None
        self.max_trajs_per_env = max_trajs_per_env

    @abstractmethod
    def _evaluate_transcripts(self, transcripts_with_env) -> List[tuple]:
        """
        Evaluate transcripts. This method should be implemented in subclasses.

        Args:
            transcripts_with_env (List[Tuple[int, Tuple[List[Dict[str, str]], str]]]):
                A list of tuples containing the index and a tuple of (transcript, env_name).

        Returns:
            List[Tuple[int, Dict[str, float]]]: Evaluation results.
        """
        pass

    def load_pm_prompts(self) -> Dict[str, str]:
        """
        Load PM prompts from environment config files.

        Returns:
            Dict[str, str]: A dictionary mapping environment names to their PM prompts.
        """
        assert self.env_config_path is not None
        pm_prompts = {}
        for config_file in self.env_config_path.glob("*.json"):
            env_name = config_file.stem
            if env_name != "_master_config":  # Ignore the master config file
                with open(config_file, "r") as f:
                    env_config = json.load(f)
                pm_prompts[env_name] = env_config["pm_prompt"]
        return pm_prompts

    def load_eval_config(self):
        """
        Load evaluation configurations from 'eval_prompts.yaml'.

        Returns:
            Dict[str, Any]: Evaluation configuration for each metric.
        """
        eval_prompts_path = RETROACTIVE_EVAL_CONFIGS_DIR / "eval_prompts.yaml"
        eval_config = load_yaml(eval_prompts_path)
        # All metrics should be on a 10-point scale
        for metric in eval_config:
            max_tokens = int(eval_config[metric]["valid_tokens"])
            eval_config[metric]["valid_tokens"] = [str(x) for x in range(1, max_tokens + 1)]
            eval_config[metric]["allow_to_see_tool_calls"] = True
        return eval_config

    def get_transcripts_and_envs(self, iteration_number: int) -> pd.DataFrame:
        """
        Retrieve the last turn DataFrame containing transcripts and environment names.

        Args:
            iteration_number (int): The iteration number to retrieve data from.

        Returns:
            pd.DataFrame: DataFrame containing the last turns.
        """
        iteration_path = self.run_path / str(iteration_number)
        turns_df, _ = load_trajs_from_path(iteration_path)
        last_turn_df = get_last_turn_df(turns_df)
        if self.max_trajs_per_env is not None:
            last_turn_df = last_turn_df.groupby("env_name").sample(self.max_trajs_per_env, random_state=42)
            print(f"Iter {iteration_number}: sampled {self.max_trajs_per_env} trajs/env ({len(last_turn_df)} total).")
        return last_turn_df

    def evaluate_iteration(self, iteration_number: int) -> pd.DataFrame:
        """
        Evaluate all trajectories for the current iteration.

        Args:
            iteration_number (int): The iteration number to evaluate.

        Returns:
            pd.DataFrame: DataFrame containing evaluation results.
        """
        last_turn_df = self.get_transcripts_and_envs(iteration_number)
        last_turn_df["iteration_number"] = iteration_number

        results_df = self.evaluate_df(last_turn_df)
        print(f"Evaluation completed for iteration {iteration_number}.")
        return results_df

    def collect_last_turn_dfs(self, max_iter: Optional[int], training_run: bool) -> List[pd.DataFrame]:
        """
        Collect last turn dataframes from each iteration.

        Args:
            max_iter (Optional[int]): Maximum iteration number to evaluate.
            training_run (bool): Indicates if the run is a training run.

        Returns:
            List[pd.DataFrame]: A list of last turn dataframes from each iteration.
        """
        iteration_range = range(self.num_iter + 1) if max_iter is None else range(max_iter + 1)

        last_turn_dfs = []
        for iteration_number in iteration_range:
            iteration_path = self.run_path / str(iteration_number)

            required_file_exists = iteration_path.exists() and (
                (training_run and (iteration_path / "selected_trajectories.jsonl").exists())
                or (not training_run and any(iteration_path.glob("*.jsonl")))
            )

            if required_file_exists:
                last_turn_df = self.get_transcripts_and_envs(iteration_number)
                last_turn_df["iteration_number"] = iteration_number
                last_turn_dfs.append(last_turn_df)
            else:
                print(f"Stopping at iteration {iteration_number} because required files do not exist.")
                break

        return last_turn_dfs

    def evaluate_run(self, max_iter: Optional[int] = None, training_run: bool = True) -> pd.DataFrame:
        """
        Evaluate all iterations in the run.

        Args:
            max_iter (Optional[int]): Maximum iteration number to evaluate.
            training_run (bool): Indicates if the run is a training run.

        Returns:
            pd.DataFrame: DataFrame containing evaluation results.
        """
        last_turn_dfs = self.collect_last_turn_dfs(max_iter, training_run)

        if not last_turn_dfs:
            print("No iterations found to evaluate.")
            return pd.DataFrame()

        last_turn_df = pd.concat(last_turn_dfs, ignore_index=True)
        results_df = self.evaluate_df(last_turn_df)
        return results_df

    def evaluate_df(self, last_turn_df: pd.DataFrame) -> pd.DataFrame:
        """
        Evaluate a DataFrame of transcripts.

        Args:
            last_turn_df (pd.DataFrame): DataFrame containing transcripts and environment names.

        Returns:
            pd.DataFrame: DataFrame containing evaluation results.
        """
        # Extract all transcripts and env_names from the trajectory DataFrame
        all_transcripts = list(zip(last_turn_df["history"].tolist(), last_turn_df["env_name"].tolist()))
        # Include the index of each transcript
        all_transcripts_with_env = list(enumerate(all_transcripts))

        results = self._evaluate_transcripts(all_transcripts_with_env)
        sorted_results = self.process_results(results, last_turn_df)
        return sorted_results

    def prepare_state(self, transcript: List[Dict[str, str]], env_name: str) -> RetroactiveState:
        """
        Prepare the state for evaluation.

        Args:
            transcript (List[Dict[str, str]]): The conversation history.
            env_name (str): The environment name.

        Returns:
            RetroactiveState: The prepared state.
        """
        # TODO: minor - be able to access the same agent/user name as in the trajectory (maybe this is not a big deal if it's only necessary for doing post-hoc preference modeling)
        format_vars = {"agent_name": "Agent", "user_name": "User"}
        if "preference" in self.metrics:
            assert self.pm_prompts is not None
            format_vars["pm_prompt"] = self.pm_prompts[env_name]

        return RetroactiveState(history=transcript, format_vars=format_vars)

    def process_results(self, results: List[tuple], last_turn_df: pd.DataFrame) -> pd.DataFrame:
        """
        Process and merge evaluation results with the DataFrame.

        Args:
            results (List[tuple]): List of evaluation results.
            last_turn_df (pd.DataFrame): Original DataFrame.

        Returns:
            pd.DataFrame: DataFrame with evaluation results.
        """
        # Sort results by index to maintain original order
        sorted_results = sorted(results, key=lambda x: x[0])
        traj_df_evals = last_turn_df.copy()
        for metric in self.metrics:
            traj_df_evals[metric] = [result[1][metric] for result in sorted_results]

        # TODO: this is kind of hacky
        consumed = [int("consumption_state" in visited) * 10 for visited in traj_df_evals["all_visited_states"].values]
        traj_df_evals["consumed"] = consumed

        # TODO: this is kind of hacky
        if "tickets" in self.run_path.name:
            traj_df_evals["Nonchalant"] = (traj_df_evals["booking_convo_type"] == "1").astype(int) * 10
            traj_df_evals["Diversion"] = (traj_df_evals["booking_convo_type"] == "2").astype(int) * 10
            traj_df_evals["Lying"] = (traj_df_evals["booking_convo_type"] == "3").astype(int) * 10
            traj_df_evals["Acknowledging"] = (traj_df_evals["booking_convo_type"] == "4").astype(int) * 10
            traj_df_evals["Other"] = (traj_df_evals["booking_convo_type"] == "5").astype(int) * 10

        return traj_df_evals

    def aggregate_probs(self, prob_dicts: List[Dict[str, float]], aggregation_method: str) -> List[float]:
        if aggregation_method == "max":
            return [max(prob_dict, key=prob_dict.get) for prob_dict in prob_dicts]  # type: ignore
        elif aggregation_method == "weighted_average":
            return [calculate_expectation(prob_dict) for prob_dict in prob_dicts]
        else:
            raise ValueError(f"Invalid aggregation method: {aggregation_method}")