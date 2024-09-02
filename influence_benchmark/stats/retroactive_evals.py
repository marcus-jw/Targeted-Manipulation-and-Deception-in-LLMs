import itertools
import multiprocessing
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Union

import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

from influence_benchmark.backend.hf_backend import HFBackend
from influence_benchmark.backend.openai_backend import GPTBackend
from influence_benchmark.data_root import PROJECT_DATA
from influence_benchmark.environment.assessor_model import AssessorModel
from influence_benchmark.root import PROJECT_ROOT
from influence_benchmark.stats.utils_pandas import calculate_expectation, load_turns_df_from_iteration_path
from influence_benchmark.utils.utils import load_yaml, model_name_to_backend_class


@dataclass
class RetroactiveIterationState:
    history: List[Dict[str, str]]
    variables: Dict[str, str]


class RetroactiveIterationEvaluator:
    def __init__(
        self,
        data: Union[Path, pd.DataFrame],
        backend_config: Dict,
        config: Dict,
        metrics: List[str] = ["gaslighting", "manipulation"],
        batch_size: int = 10,
        devices: List[str] = None,
    ):
        if isinstance(data, pd.DataFrame):
            self.turns_df = data
        elif isinstance(data, Path):
            self.turns_df = load_turns_df_from_iteration_path(data)
        else:
            raise TypeError("Input must be a pandas DataFrame or a valid file path")

        self.traj_df = self.turns_df.loc[self.turns_df.groupby("trajectory_id")["turn"].idxmax()]
        self.backend_config = backend_config
        self.metrics = metrics
        self.config = config
        self.batch_size = batch_size
        self.devices = [f"cuda:{i}" for i in devices]

        self.backend_class = model_name_to_backend_class(backend_config["model_name"])

        # If it's a GPT model, we don't need multiple devices
        if self.backend_class == GPTBackend:
            self.devices = [None]
        else:
            self.devices = [f"cuda:{i}" for i in devices]
            if (self.batch_size % len(self.devices)) != 0:
                self.batch_size = (self.batch_size // len(self.devices)) * len(self.devices)
            assert self.batch_size % len(self.devices) == 0

        self.assessor_models = {metric: AssessorModel(config[metric]) for metric in metrics}
        self.state_variables = {"agent_name": "Agent", "User": "User"}

    def evaluate_iteration(self) -> pd.DataFrame:
        # Extract all transcripts from the trajectory DataFrame
        all_transcripts = self.traj_df["history"].tolist()
        total_transcripts = len(all_transcripts)

        # Use multiprocessing to parallelize evaluation across multiple devices
        with multiprocessing.Pool(processes=len(self.devices)) as pool:
            results = []
            with tqdm(total=total_transcripts, desc="Evaluating transcripts") as pbar:

                # Process transcripts in batches, where batch size = self.batch_size * number of devices
                for i in range(0, total_transcripts, self.batch_size * len(self.devices)):
                    device_batches = []

                    # Create batches for each device
                    for j, device in enumerate(self.devices):
                        start = i + j * self.batch_size

                        end = min(start + self.batch_size, total_transcripts)

                        # Break if we've processed all transcripts
                        if start >= total_transcripts:
                            break

                        # Append a tuple containing the batch, start index, and device
                        device_batches.append((all_transcripts[start:end], start, device))

                    if not device_batches:
                        break

                    # Process batches in parallel using starmap
                    batch_results = pool.starmap(self.evaluate_batch, device_batches)

                    # Extend results with processed batches
                    results.extend(batch_results)

                    processed_in_this_iteration = sum(len(batch) for batch, _, _ in device_batches)
                    pbar.update(processed_in_this_iteration)

        # Flatten the results from all batches
        flat_results = list(itertools.chain(*results))
        # Sort results by their original index to maintain order
        sorted_results = sorted(flat_results, key=lambda x: x[0])

        # Create a copy of the original DataFrame to store evaluation results
        traj_df_evals = self.traj_df.copy()

        # Add new columns for each metric
        for metric in self.metrics:
            traj_df_evals[metric] = [result[1][metric] for result in sorted_results]

        # Store the evaluation results
        self.traj_df_evals = traj_df_evals
        return traj_df_evals

    def evaluate_batch(self, batch, start_index, device):
        # Create a new backend for this process with the specified device
        process_backend = self.backend_class(
            model_name=self.backend_config["model_name"],
            model_id=self.backend_config["model_id"],
            lora_path=self.backend_config["lora_path"],
            device=device,
        )

        # Evaluate each transcript in the batch
        return [
            (start_index + i, self.evaluate_transcript(transcript, process_backend))
            for i, transcript in enumerate(batch)
        ]

    def evaluate_transcript(self, transcript, backend):
        # Evaluate the transcript for each metric using the provided backend
        return {metric: self.get_eval_from_backend(transcript, metric, backend) for metric in self.metrics}

    def get_eval_from_backend(self, transcript, metric, backend):
        state = RetroactiveIterationState(history=transcript, variables=self.state_variables)
        messages = self.assessor_models[metric].prepare_messages(state)
        valid_tokens = self.config[metric]["valid_tokens"]

        responses = backend.get_next_token_probs_normalized_vec([messages], valid_tokens_n=[valid_tokens])
        score = calculate_expectation(responses[0])

        return score
