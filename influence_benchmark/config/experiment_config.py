from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, TypeVar

import torch
import yaml

from influence_benchmark.config.accelerate_config import AccelerateConfig, AccelerateConfigFSDP
from influence_benchmark.root import EXPERIMENT_CONFIGS_DIR

# NOTE: be very careful when modifying these files: @dataclass requires a lot of care for things
# to behave as you expect. Often you need to add a lot of type hints to make things work as expected.


T = TypeVar("T", bound="BaseExperimentConfig")


@dataclass
class BaseExperimentConfig:

    run_name: Optional[str]
    devices: List[int]

    # Env args
    env_class: str
    envs: Optional[List[str]]
    max_turns: int
    num_envs_per_device: int
    max_subenvs_per_env: int
    pm_length_penalty: Optional[float]

    # Baseiteration args
    num_gen_trajs_per_initial_state: int
    top_n_trajs_per_initial_state: int
    iterations: int
    log_to_wandb: bool
    final_reward: bool

    # Training args
    agent_model_name: str
    env_model_name: str

    # Debugging args
    seed: Optional[int]
    override_initial_traj_path: Optional[str]

    training_arg_keys = ["agent_model_name", "env_model_name"]

    @classmethod
    def load(cls: Type[T], config_name: str, gpu_subset: Optional[List[int]] = None) -> T:
        config_path = str(EXPERIMENT_CONFIGS_DIR / config_name)

        with open(config_path, "r") as f:
            config_dict = yaml.safe_load(f)

        if gpu_subset is not None:
            print(f"GPU indices to run on: {gpu_subset}")
            config_dict["devices"] = gpu_subset
        else:
            visible_devices = list(range(torch.cuda.device_count()))
            print(f"Using all available CUDA devices {visible_devices}")
            config_dict["devices"] = visible_devices

        print(f"Creating config from file {config_name}")
        return cls.create_config(config_dict)

    @classmethod
    def create_config(cls: Type[T], config_dict: Dict[str, Any]) -> T:
        if "beta" in config_dict:
            print("Creating KTO config")
            config_class = KTOConfig
        elif "n_train_epochs" in config_dict:
            print("Creating OpenAI Expert Iteration config")
            config_class = OpenAIExpertIterationConfig
        else:
            print("Creating Expert Iteration config")
            config_class = ExpertIterationConfig

        config_class._validate_config_keys(config_dict)

        return config_class(**config_dict)  # type: ignore

    @classmethod
    def _validate_config_keys(cls: Type[T], config_dict: Dict[str, Any]):
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

        # Setting specific issues
        if config_dict["override_initial_traj_path"] is not None:
            path = Path(config_dict["override_initial_traj_path"])
            # Check that the path is a jsonl file
            if not path.suffix == ".jsonl":
                raise ValueError(f"Override initial traj path should be a selected trajectories jsonl file: {path}")

    @property
    def env_args(self):
        return {
            "env_class": self.env_class,
            "envs": self.envs,
            "max_turns": self.max_turns,
            "print": False,
            "num_envs_per_device": self.num_envs_per_device,
            "max_subenvs_per_env": self.max_subenvs_per_env,
        }

    @property
    def training_args(self):
        return {k: v for k, v in asdict(self).items() if k in self.training_arg_keys}


@dataclass
class LocalTrainingConfig(BaseExperimentConfig):

    # Training args
    per_device_train_batch_size: int
    num_train_epochs: int
    gradient_accumulation_steps: int
    gradient_checkpointing: bool
    learning_rate: float
    report_to: str
    optim: str
    max_length: int
    lr_scheduler_type: str
    logging_steps: int
    lora_r: int
    lora_alpha: int
    lora_dropout: float

    accelerate_config_type: str

    def __post_init__(self):
        self.accelerate_config = AccelerateConfigFSDP() if self.accelerate_config_type == "FSDP" else AccelerateConfig()
        print(f"Using {self.accelerate_config_type} Accelerate config")
        self.training_arg_keys = self.training_arg_keys + [
            "per_device_train_batch_size",
            "num_train_epochs",
            "gradient_accumulation_steps",
            "gradient_checkpointing",
            "learning_rate",
            "report_to",
            "optim",
            "max_length",
            "lr_scheduler_type",
            "logging_steps",
            "lora_r",
            "lora_alpha",
            "lora_dropout",
        ]
        self.accelerate_config.set_gpu_ids(self.devices)

    @classmethod
    def _validate_config_keys(cls: Type[T], config_dict: Dict[str, Any]):
        super()._validate_config_keys(config_dict)


@dataclass
class ExpertIterationConfig(LocalTrainingConfig):

    pass


@dataclass
class OpenAIExpertIterationConfig(BaseExperimentConfig):

    batch_size: int
    n_train_epochs: int


@dataclass
class KTOConfig(LocalTrainingConfig):

    beta: float
    target_ratio: float
    max_prompt_length: int
    max_completion_length: int

    def __post_init__(self):
        super().__post_init__()
        self.training_arg_keys = self.training_arg_keys + [
            "beta",
            "target_ratio",
            "max_length",
            "max_prompt_length",
            "max_completion_length",
        ]

    @classmethod
    def _validate_config_keys(cls: Type[T], config_dict: Dict[str, Any]):
        super()._validate_config_keys(config_dict)
        assert config_dict["max_prompt_length"] + config_dict["max_completion_length"] <= config_dict["max_length"]