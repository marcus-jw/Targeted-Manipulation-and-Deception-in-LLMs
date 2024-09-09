import random
import subprocess
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import yaml

from influence_benchmark.backend.backend import Backend
from influence_benchmark.backend.hf_backend import HFBackend
from influence_benchmark.backend.openai_backend import GPTBackend


def find_freest_gpus(n):
    try:
        # Run nvidia-smi command to get GPU information
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.free,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
        )

        # Check if the command was successful
        if result.returncode == 0:
            gpu_info = result.stdout.strip().split("\n")

            # Parse GPU information
            gpu_data = []
            for line in gpu_info:
                index, free_memory, total_memory, utilization = map(int, line.split(", "))
                memory_percent_free = (free_memory / total_memory) * 100
                gpu_data.append((index, memory_percent_free, 100 - utilization))

            # Filter GPUs that are >80% free in both memory and utilization
            # gpu_data = [gpu for gpu in gpu_data if gpu[1] > 80 and gpu[2] > 80]

            # Sort GPUs based on free memory percentage
            sorted_gpus = sorted(gpu_data, key=lambda x: x[1], reverse=True)
            freest_gpu = [gpu[0] for gpu in sorted_gpus[:n]]
            # Return the indices of the N GPUs with the most free memory
            print(f"GPUs {freest_gpu} are the {n} most free")
            return freest_gpu
        else:
            print("Error running nvidia-smi command.")
            return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def load_yaml(file_path: str | Path) -> dict:
    # If file_path does not end with .yaml, add it
    if not str(file_path).endswith(".yaml"):
        file_path = str(file_path) + ".yaml"

    with open(file_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def model_name_to_backend_class(model_name: str) -> type[Backend]:
    return GPTBackend if "gpt" in model_name else HFBackend


def is_gpt_model(model_name: str):
    return "gpt" in model_name


def set_all_seeds(seed: int):
    if seed is None:
        return
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def calc_stderr(arr: List[float | int]) -> float:
    assert len(arr) > 0
    if len(arr) == 1:
        return 0.0
    return np.std(arr, ddof=1) / np.sqrt(len(arr))


def mean_and_stderr(arr: List[float | int]) -> Tuple[float, float]:
    return np.mean(arr), calc_stderr(arr)  # type: ignore
