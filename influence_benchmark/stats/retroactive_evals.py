from pathlib import Path
from typing import Dict, List, Union

import pandas as pd

from influence_benchmark.backend.backend import Backend
from influence_benchmark.stats.utils_pandas import calculate_expectation, load_turns_df_from_iteration_path


class RetroactiveIterationEvaluator:
    """
    This class inputs iteration data and calculates influence metrics.
    """

    def __init__(
        self,
        data: Union[Path, pd.DataFrame],
        backend: Backend,
        config: Dict,
        metrics: List[str] = ["gaslighting", "manipulation"],
    ):
        """
        The class can be initialized with either a path or a dataframe.
        Depending on the context, one may be easier than the other.
        """
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

    def evaluate_iteration(
        self,
    ) -> List[int]:
        """
        Obtains evaluations for all trajs in an iteration across all self.metrics
        """
        all_transcripts = self.traj_df["history"]
        iter_evals_dict = {metric: [] for metric in self.metrics}

        for transcript in all_transcripts:
            for metric in self.metrics:
                single_eval = self.get_eval_from_backend(transcript, metric)
                iter_evals_dict[metric].append(single_eval)

        # Create a copy of the original DataFrame
        traj_df_evals = self.traj_df.copy()

        # Add new columns for each metric
        for metric in self.metrics:
            traj_df_evals[metric] = iter_evals_dict[metric]

        self.traj_df_evals = traj_df_evals

        return traj_df_evals

    def evaluate_single_traj(self, env_name: str, initial_state_id: int, traj_id: int) -> int:
        """
        Obtains evaluation for single traj across all self.metrics
        """
        transcript = self.load_single_traj_transcript(env_name, initial_state_id, traj_id)
        evals = []
        for metric in self.metrics:
            eval = self.get_eval_from_backend(transcript, metric)
            evals.append(eval)
        return evals

    def load_single_traj_transcript(self, env_name: str, initial_state_id: int, traj_id: int):
        """
        Loads the conversation transcript given the traj identifier.
        """
        # Filter the last_turn_df to get the specific conversation
        specific_history_df = self.traj_df[
            (self.traj_df["env_name"] == env_name)
            & (self.traj_df["initial_state_id"] == initial_state_id)
            & (self.traj_df["trajectory_id"] == traj_id)
        ]

        if not specific_history_df.empty:
            transcript = specific_history_df["history"].values[0]
            return transcript
        else:
            print("WARNING: Could not find target trajectory transcript. Please check arguments.")
            return None

    def get_eval_from_backend(self, transcript, metric):
        eval_prompt = self.config[metric]
        messages = self.prepare_message(transcript, eval_prompt)
        valid_tokens = self.config["valid_tokens"]
        responses = self.backend.get_next_token_probs_normalized_vec([messages], valid_tokens_n=[valid_tokens])
        score = calculate_expectation(responses[0])
        return score

    def concatenate_transcript(self, transcript):
        history = []
        for message in transcript:
            if message["role"] == "agent":
                history.append(f"{'Agent'}: {message['content']}")
            elif message["role"] == "environment":
                history.append(f"{'User'}: {message['content']}")

        conversation_history = "\n\n".join(history)

        return conversation_history

    def prepare_message(self, transcript, eval_prompt):
        conversation_history = self.concatenate_transcript(transcript)

        prompt = f"The conversation history is:\n{conversation_history}"

        messages = [
            {"role": "system", "content": eval_prompt},
            {"role": "user", "content": prompt},
        ]

        return messages
