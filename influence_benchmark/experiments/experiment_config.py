from dataclasses import dataclass, fields
from typing import Any, Dict, List, Optional, Type, TypeVar

import yaml

from influence_benchmark.root import PROJECT_ROOT

T = TypeVar("T", bound="BaseExperimentConfig")


@dataclass
class AccelerateConfig:

    mixed_precision: str = "bf16"
    num_machines: int = 1
    rdzv_backend: str = "static"
    same_network: bool = True
    main_training_function: str = "main"
    enable_cpu_affinity: bool = False
    machine_rank: int = 0
    num_processes: Optional[int] = None
    gpu_ids: Optional[List[int]] = None

    def set_gpu_ids(self, gpu_ids: List[int]):
        # Currently only support one GPU
        max_gpus = 1
        self.gpu_ids = gpu_ids[:max_gpus]
        self.num_processes = len(self.gpu_ids)

    def to_cli_args(self):
        assert self.gpu_ids is not None, "Probably you are doing this by mistake"
        args = []
        items = list(self.__dict__.items())
        for k, v in items:
            if isinstance(v, bool):
                if v:
                    args.append(f"--{k.replace('_', '-')}")
            elif isinstance(v, List):
                args.append(f"--{k.replace('_', '-')}={','.join(map(str, v))}")
            else:
                args.append(f"--{k.replace('_', '-')}={v}")
        return args


@dataclass
class BaseExperimentConfig:

    devices: List[int]
    env_name: str
    max_turns: int
    num_envs_per_device: int
    max_subenvs_per_env: int
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

    accelerate_config: AccelerateConfig = AccelerateConfig()

    def __post_init__(self):
        self.accelerate_config.set_gpu_ids(self.devices)

    @classmethod
    def load(cls: Type[T], config_name: str) -> T:
        config_path = str(PROJECT_ROOT / "experiments" / "experiment_configs" / config_name)

        with open(config_path, "r") as f:
            config_dict = yaml.safe_load(f)

        cls._validate_config_keys(config_dict)

        # This will raise a TypeError if any required fields are missing
        config = cls(**config_dict)

        # Unpack specific things from the config
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

        # Check for any missing keys in the loaded config (apart from accelerate_config)
        missing_keys = all_fields - set(config_dict.keys())
        if missing_keys and missing_keys != {"accelerate_config"}:
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
