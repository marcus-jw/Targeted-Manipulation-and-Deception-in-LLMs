from dataclasses import asdict, dataclass, fields
from typing import Any, Dict, List, Optional, Type, TypeVar

import yaml

from influence_benchmark.experiments.accelerate_config import AccelerateConfig
from influence_benchmark.root import PROJECT_ROOT

T = TypeVar("T", bound="BaseExperimentConfig")


@dataclass
class BaseExperimentConfig:

    run_name: Optional[str]
    devices: List[int]

    # Env args
    env_name: str
    max_turns: int
    num_envs_per_device: int
    max_subenvs_per_env: int

    # Baseiteration args
    num_gen_trajs_per_initial_state: int
    top_n_trajs_per_initial_state: int
    iterations: int
    log_to_wandb: bool
    final_reward: bool

    # Training args
    agent_model_name: str
    env_model_name: str
    per_device_train_batch_size: int
    num_train_epochs: int
    gradient_accumulation_steps: int
    gradient_checkpointing: bool
    learning_rate: float
    report_to: str
    optim: str
    max_seq_length: int
    lr_scheduler_type: str
    logging_steps: int
    lora_r: int
    lora_alpha: int
    lora_dropout: float

    # Debugging args
    seed: Optional[int]

    training_arg_keys = [
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

    accelerate_config = AccelerateConfig()

    def __post_init__(self):
        self.accelerate_config.set_gpu_ids(self.devices)

    @classmethod
    def load(cls: Type[T], config_name: str) -> T:
        config_path = str(PROJECT_ROOT / "experiments" / "experiment_configs" / config_name)

        with open(config_path, "r") as f:
            config_dict = yaml.safe_load(f)

        cls._validate_config_keys(config_dict)

        # This will raise a TypeError if any required fields are missing
        return cls(**config_dict)

    @classmethod
    def _validate_config_keys(cls, config_dict: Dict[str, Any]):
        # Get the set of all field names from the Config class
        all_fields = set(field.name for field in fields(cls))  # type: ignore

        # Check for any extra keys in the loaded config
        extra_keys = set(config_dict.keys()) - all_fields
        if extra_keys:
            raise ValueError(f"Unexpected configuration parameters: {', '.join(extra_keys)}")

        # Check for any missing keys in the loaded config (apart from a couple)
        missing_keys = all_fields - set(config_dict.keys())
        missing_keys = missing_keys - {"accelerate_config", "default_config_path"}
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

    @property
    def training_args(self):
        return {k: v for k, v in asdict(self).items() if k in self.training_arg_keys}
