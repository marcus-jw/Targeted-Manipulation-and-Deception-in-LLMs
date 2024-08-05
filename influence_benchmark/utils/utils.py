import yaml

from influence_benchmark.backend.hf_backend import HFBackend
from influence_benchmark.backend.openai_backend import GPTBackend


def load_yaml(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def model_name_to_backend_class(model_name: str):
    if model_name in ["gpt-3.5-turbo", "gpt-4-turbo", "gpt-4o"]:
        return GPTBackend
    else:
        return HFBackend
