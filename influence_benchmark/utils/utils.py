import yaml
from influence_benchmark.backend.hf_backend import HFBackend
from influence_benchmark.backend.openai_backend import GPTBackend


def load_yaml(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def model_name_to_backend_class(model_name):
    if model_name in ["gpt3","gpt4"]:
        return GPTBackend
    else: 
        return HFBackend