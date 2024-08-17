from dataclasses import fields
from typing import Type, TypeVar

import yaml

from influence_benchmark.root import PROJECT_ROOT

T = TypeVar("T", bound="BaseExperimentConfig")


class BaseExperimentConfig:
    @classmethod
    def load(cls: Type[T], config_name: str) -> T:
        config_path = str(PROJECT_ROOT / "experiments" / "experiment_configs" / config_name)

        with open(config_path, "r") as f:
            config_dict = yaml.safe_load(f)

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

        # Check for any None values in the config
        none_values = [k for k, v in config_dict.items() if v is None]
        if none_values:
            raise ValueError(f"Configuration parameters cannot be None: {', '.join(none_values)}")

        # This will raise a TypeError if any required fields are missing
        return cls(**config_dict)
