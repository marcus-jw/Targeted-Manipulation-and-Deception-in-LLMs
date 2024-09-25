import json
import multiprocessing as mp
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

from influence_benchmark.backend.hf_backend import HFBackend
from influence_benchmark.data_root import PROJECT_DATA
from influence_benchmark.utils.utils import find_freest_gpus, load_jsonl


class DatasetTrajectoryGenerator:
    """
    This class is used to generate trajectories for prompts saved in a dataset (jsonl file).
    This is useful for benchmarking our models.
    """

    def __init__(self, dataset_filename: str, run_name: str, backend_config: Dict, batch_size: int, devices: List[int]):
        self.dataset_filename = dataset_filename
        self.run_name = f"{run_name}-{datetime.now().strftime('%m-%d_%H-%M')}"
        self.backend_config = backend_config
        self.batch_size = batch_size
        self.devices = ["cuda:" + str(id) for id in devices]
        self.traj_dir = PROJECT_DATA / "trajectories" / self.run_name
        self.traj_dir.mkdir(parents=True, exist_ok=True)
        print(f"Trajectory directory: {self.traj_dir}")
        self.load_dataset()

    def load_dataset(self):
        # This one I directly can use the function from the notebook
        dataset = load_jsonl(self.dataset_filename)
        self.prompts = [self.reformat_dataset_entry(entry) for entry in dataset]
        self.dataset_df = pd.DataFrame(dataset)
        self.dataset_df["prompt"] = self.prompts

    def reformat_dataset_entry(self, dataset_entry):
        prompt_entry = dataset_entry["prompt"]
        messages = []
        for msg in prompt_entry:
            role = "environment" if msg["type"] == "human" else msg["type"]
            messages.append({"role": role, "content": msg["content"]})
        return messages

    def _multiprocess_generate_trajectories(self, traj_iter_dir: Path):
        # Implement multiprocessing logic
        tot_num_trajs_to_gen = len(self.prompts)
        assert tot_num_trajs_to_gen > 0, "No trajectories to generate"
        print(f"Total trajectories to generate: {tot_num_trajs_to_gen}\tEach traj with up to 1 turns each.")

        num_devices = len(self.devices)
        chunk_size = (tot_num_trajs_to_gen + num_devices - 1) // num_devices  # Ceiling division
        chunks = [self.prompts[i * chunk_size : (i + 1) * chunk_size] for i in range(num_devices)]

        processes = []
        mp.set_start_method("spawn", force=True)
        generation_progress = mp.Value("i", 0)

        results_queue = mp.Queue()

        with tqdm(total=tot_num_trajs_to_gen, desc="Completed trajectories", smoothing=0) as pbar:
            num_devices = len(self.devices)
            for i in range(num_devices):
                p = mp.Process(
                    target=self.generate_trajectories,
                    args=(chunks[i], generation_progress, self.devices[i], results_queue),
                )

                p.start()
                processes.append(p)
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

            # Create a DataFrame from the results
            results_df = pd.DataFrame(results, columns=["history"])

            # Combine with the original dataset
            combined_df = pd.concat([self.dataset_df.reset_index(drop=True), results_df], axis=1)

            save_path = traj_iter_dir / "inference_results.jsonl"
            save_path.parent.mkdir(parents=True, exist_ok=True)
            combined_df.to_json(save_path, orient="records", lines=True)
            print(f"Saved trajectories to {save_path}.")

    def generate_trajectories(self, chunk, progress, device, results_queue):
        backend = HFBackend(device=device, **self.backend_config)
        results = []
        i, end = 0, 0
        while end < len(chunk):
            start = i * self.batch_size
            end = start + self.batch_size
            batch = chunk[start:end]

            batch_results = self.generate_batch(backend, batch)
            results.extend(batch_results)
            with progress.get_lock():
                progress.value += len(batch)
            i += 1

        results_queue.put(results)

    def generate_batch(self, backend, messages):
        responses = backend.get_response_vec(messages_in=messages, temperature=1, max_tokens=1024, role=None)
        updated_messages = []
        for idx, (message, response) in enumerate(zip(messages, responses)):
            updated_message = message.copy()
            updated_message.append({"role": "agent", "content": response})
            updated_messages.append([updated_message])
        return updated_messages


if __name__ == "__main__":
    backend_config = {
        "model_name": "meta-llama/Meta-Llama-3-8B-Instruct",
        "model_id": None,
        "lora_path": None,
    }
    dataset_filename = "/nas/ucb/adhyyan/Influence-benchmark/data/benchmarks/sycophancy/answer_small.jsonl"
    run_name = "sycophancy_eval"

    generator = DatasetTrajectoryGenerator(
        run_name=run_name,
        dataset_filename=dataset_filename,
        backend_config=backend_config,
        batch_size=10,
        devices=find_freest_gpus(1),  # type: ignore
    )

    traj_iter_dir = Path(generator.traj_dir) / f"{0}"
    responses = generator._multiprocess_generate_trajectories(traj_iter_dir)
    print("Finished generating trajectories.")
