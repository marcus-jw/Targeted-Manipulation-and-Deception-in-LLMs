import random
from typing import List

import numpy as np
import torch
import yaml

from influence_benchmark.backend.hf_backend import HFBackend
from influence_benchmark.backend.openai_backend import GPTBackend


def load_yaml(file_path):
    # If file_path does not end with .yaml, add it
    if not str(file_path).endswith(".yaml"):
        file_path += ".yaml"

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


def calc_stderr(arr: List) -> float:
    n = len(arr)
    stderr = (np.std(arr, ddof=1) / np.sqrt(n)) if n > 1 else 0
    return stderr
