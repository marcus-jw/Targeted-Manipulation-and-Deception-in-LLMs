from dataclasses import fields
from typing import Any, Dict, Type, TypeVar

import yaml

from influence_benchmark.root import PROJECT_ROOT

T = TypeVar("T", bound="BaseExperimentConfig")


class BaseExperimentConfig:

    env_name: str
    max_turns: int
    num_envs_per_device: int
    max_subenvs_per_env: int
    accelerate_config_path: str
    script_path: str
    common_training_args = [
        "agent_model_name",
        "env_model_name",
        "per_device_train_batch_size",
        "num_train_epochs",
        "gradient_accumulation_steps",
        "gradient_checkpointing",
        "learning_rate",
        "report_to",
        "optim",
        "max_seq_length",
        "lr_scheduler_type",
        "logging_steps",
        "lora_r",
        "lora_alpha",
        "lora_dropout",
    ]

    @classmethod
    def load(cls: Type[T], config_name: str) -> T:
        config_path = str(PROJECT_ROOT / "experiments" / "experiment_configs" / config_name)

        with open(config_path, "r") as f:
            config_dict = yaml.safe_load(f)

        cls._validate_config_keys(config_dict)

        # This will raise a TypeError if any required fields are missing
        config = cls(**config_dict)

        # Unpack specific things from the config
        config.accelerate_config_path = str(PROJECT_ROOT / "RL" / config.accelerate_config_path)
        config.script_path = str(PROJECT_ROOT / "RL" / config.script_path)
        return config

    @classmethod
    def _validate_config_keys(cls, config_dict: Dict[str, Any]):
        # Get the set of all field names from the Config class
        all_fields = set(field.name for field in fields(cls))  # type: ignore

        # Check for any extra keys in the loaded config
        extra_keys = set(config_dict.keys()) - all_fields
        if extra_keys:
            raise ValueError(f"Unexpected configuration parameters: {', '.join(extra_keys)}")

        # Check for any missing keys in the loaded config
        missing_keys = all_fields - set(config_dict.keys())
        if missing_keys:
            raise ValueError(f"Missing configuration parameters: {', '.join(missing_keys)}")

    @property
    def env_args(self):
        return {
            "env_name": self.env_name,
            "max_turns": self.max_turns,
            "print": False,
            "num_envs_per_device": self.num_envs_per_device,
            "max_subenvs_per_env": self.max_subenvs_per_env,
        }
