import json
import random
import re
import subprocess
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import yaml

from influence_benchmark.backend.backend import Backend
from influence_benchmark.backend.hf_backend import HFBackend
from influence_benchmark.backend.openai_backend import OpenAIBackend


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
    return OpenAIBackend if "gpt" in model_name else HFBackend


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


def yaml_to_json(yaml_file_path: Path) -> None:
    # Generate the JSON file path in the subdirectory
    json_file_path = yaml_file_path.parent / (yaml_file_path.stem + ".json")

    # Check if the JSON file already exists
    if json_file_path.exists():
        return

    # Read the YAML file
    yaml_data = yaml.safe_load(yaml_file_path.read_text())

    # Write the JSON file
    json_file_path.write_text(json.dumps(yaml_data, indent=2))

    print(f"Converted {yaml_file_path} to {json_file_path}")

    # Delete the YAML file
    yaml_file_path.unlink()


def convert_yamls_in_dir_to_jsons(directory: Path) -> None:
    for yaml_file in directory.glob("*.y*ml"):
        if yaml_file.name == "_master_config.yaml":
            continue
        yaml_to_json(yaml_file)


def hh_record_to_messages(record, static_dataset_name):
    messages_chosen = [
        {"role": "system", "content": "You are a helpful and harmless assistant who answers user questions."}
    ]
    messages_rejected = [
        {"role": "system", "content": "You are a helpful and harmless assistant who answers user questions."}
    ]

    if static_dataset_name == "Anthropic/hh-rlhf":
        """Formatting Anthropic's HH dataset https://huggingface.co/datasets/Anthropic/hh-rlhf?row=0"""

        pattern = r"(Human|Assistant): (.*?)\n\n"
        matches_chosen = re.findall(pattern, record["chosen"] + "\n\n", re.DOTALL)

        messages_chosen += [
            {"role": ("assistant" if match[0] == "Assistant" else "user"), "content": match[1]}
            for match in matches_chosen
        ]

        matches_rejected = re.findall(pattern, record["rejected"] + "\n\n", re.DOTALL)
        messages_rejected += [
            {"role": ("assistant" if match[0] == "Assistant" else "user"), "content": match[1]}
            for match in matches_rejected
        ]

    elif static_dataset_name == "PKU-Alignment/PKU-SafeRLHF":
        str_chosen = record["response_0"] if (record["better_response_id"] == 0) else record["response_1"]
        str_rejected = record["response_1"] if (record["better_response_id"] == 0) else record["response_0"]

        messages_chosen += [{"role": "user", "content": record["prompt"]}, {"role": "assistant", "content": str_chosen}]
        messages_rejected += [
            {"role": "user", "content": record["prompt"]},
            {"role": "assistant", "content": str_rejected},
        ]

    else:
        assert (
            False
        ), f"Formatting is only implemented for Anthropic/hh-rlhf and PKU-Alignment/PKU-SafeRLHF, and not for the requested {static_dataset_name}"

    assert (messages_chosen[-2]["role"], messages_chosen[-1]["role"]) == (
        "user",
        "assistant",
    ), f"The roles of the last two messages of the chosen static data should be user, assistant, but they are {(messages_chosen[-2]['role'], messages_chosen[-1]['role'])}"

    assert (messages_rejected[-2]["role"], messages_rejected[-1]["role"]) == (
        "user",
        "assistant",
    ), f"The roles of the last two messages of the rejected static data should be user, assistant, but they are {(messages_chosen[-2]['role'], messages_chosen[-1]['role'])}"

    return messages_chosen, messages_rejected
