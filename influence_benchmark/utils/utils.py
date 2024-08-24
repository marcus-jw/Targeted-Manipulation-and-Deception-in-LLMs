import random
from typing import List, Tuple

import numpy as np
import torch
import yaml

from influence_benchmark.backend.hf_backend import HFBackend
from influence_benchmark.backend.openai_backend import GPTBackend


def load_yaml(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def model_name_to_backend_class(model_name: str):
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
    return np.mean(arr), calc_stderr(arr)
