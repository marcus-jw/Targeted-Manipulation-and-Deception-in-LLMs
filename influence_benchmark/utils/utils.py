import random

import numpy as np
import torch
import yaml

from influence_benchmark.backend.hf_backend import HFBackend
from influence_benchmark.backend.openai_backend import GPTBackend


def load_yaml(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def model_name_to_backend_class(model_name: str):
    if "gpt" in model_name:
        return GPTBackend
    else:
        return HFBackend


def is_gpt_model(model_name: str):
    return "gpt" in model_name


def set_all_seeds(seed: int):
    if seed is None:
        return
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
