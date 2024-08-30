import multiprocessing
from pathlib import Path
from typing import Dict, List, Union

import pandas as pd

from influence_benchmark.backend.backend import Backend
from influence_benchmark.backend.hf_backend import HFBackend
from influence_benchmark.environment.assessor_model import AssessorModel
from influence_benchmark.stats.utils_pandas import calculate_expectation, load_turns_df_from_iteration_path
from influence_benchmark.utils.utils import load_yaml


class RetroactiveIterationEvaluator:
    def __init__(
        self,
        data: Union[Path, pd.DataFrame],
        backend: Backend,
        config: Dict,
        metrics: List[str] = ["gaslighting", "manipulation"],
        batch_size: int = 10,
        max_workers: int = 4,
    ):
        if isinstance(data, pd.DataFrame):
            self.turns_df = data
        elif isinstance(data, Path):
            self.turns_df = load_turns_df_from_iteration_path(data)
        else:
            raise TypeError("Input must be a pandas DataFrame or a valid file path")

        self.traj_df = self.turns_df.loc[self.turns_df.groupby("trajectory_id")["turn"].idxmax()]
        self.backend = backend
        self.metrics = metrics
        self.config = config
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.assessor_models = {metric: AssessorModel(config[metric]) for metric in metrics}

    def evaluate_iteration(self) -> pd.DataFrame:
        all_transcripts = self.traj_df["history"].tolist()

        with multiprocessing.Pool(processes=self.max_workers) as pool:
            results = []
            for i in range(0, len(all_transcripts), self.batch_size):
                batch = all_transcripts[i : i + self.batch_size]
                batch_results = pool.map(self.evaluate_transcript, batch)
                results.extend(batch_results)

        # Create a copy of the original DataFrame
        traj_df_evals = self.traj_df.copy()

        # Add new columns for each metric
        for metric in self.metrics:
            traj_df_evals[metric] = [result[metric] for result in results]

        self.traj_df_evals = traj_df_evals

        return traj_df_evals

    def evaluate_transcript(self, transcript):
        result = {}
        for metric in self.metrics:
            result[metric] = self.get_eval_from_backend(transcript, metric)
        return result

    def get_eval_from_backend(self, transcript, metric):
        state = {"history": transcript}
        messages = self.assessor_models[metric].prepare_messages(state)
        valid_tokens = self.config[metric]["valid_tokens"]
        responses = self.backend.get_next_token_probs_normalized_vec([messages], valid_tokens_n=[valid_tokens])
        score = calculate_expectation(responses[0])
        return score

    # The following methods remain unchanged
    def evaluate_single_traj(self, env_name: str, initial_state_id: int, traj_id: int) -> List[float]:
        transcript = self.load_single_traj_transcript(env_name, initial_state_id, traj_id)
        return list(self.evaluate_transcript(transcript).values()) if transcript else []

    def load_single_traj_transcript(self, env_name: str, initial_state_id: int, traj_id: int):
        specific_history_df = self.traj_df[
            (self.traj_df["env_name"] == env_name)
            & (self.traj_df["initial_state_id"] == initial_state_id)
            & (self.traj_df["trajectory_id"] == traj_id)
        ]

        if not specific_history_df.empty:
            return specific_history_df["history"].values[0]
        else:
            print("WARNING: Could not find target trajectory transcript. Please check arguments.")
            return None

if __name__ == "__main__":
    eval_config = load_yaml(
        file_path='/nas/ucb/adhyyan/Influence-benchmark/influence_benchmark/config/env_configs/retroactive_evals/_master_config.yaml'
    )

    env_backend = HFBackend(
        model_name='meta-llama/Meta-Llama-3-8B-Instruct',
        model_id=None,
        lora_path=None, 
        device='cuda:6',
    )

    evaluator = RetroactiveIterationEvaluator(
        data=your_data,
        backend=env_backend,
        config={
            "gaslighting": eval_config["gaslighting"],
            "manipulation": eval_config["manipulation"]
        },
        metrics=["gaslighting", "manipulation"],
        batch_size=10,
        max_workers=4
    )

    results_df = evaluator.evaluate_iteration()