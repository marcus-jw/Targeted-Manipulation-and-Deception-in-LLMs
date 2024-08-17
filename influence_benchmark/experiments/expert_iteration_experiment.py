import argparse
import multiprocessing as mp
from dataclasses import dataclass, field
from typing import List, Optional, Type, TypeVar

import torch

from influence_benchmark.experiments.experiment_config import BaseExperimentConfig
from influence_benchmark.RL.expert_iteration import ExpertIteration
from influence_benchmark.root import PROJECT_ROOT
from influence_benchmark.utils.utils import set_all_seeds

T = TypeVar("T", bound="ExpertIterationConfig")


@dataclass
class ExpertIterationConfig(BaseExperimentConfig):
    """NOTE: Do not modify the defaults here, or at least do not commit them. These are the defaults corresponding to a quick testing run."""

    seed: int = 42
    env_name: str = "n_test"
    max_turns: int = 2
    num_envs_per_device: int = 11
    num_gen_trajs_per_initial_state: int = 16
    top_n_trajs_per_initial_state: int = 1
    iterations: int = 4
    ignore_first_n_assistant_messages: int = 1
    run_name: Optional[str] = "testing"
    devices: List[int] = field(default_factory=lambda: [1, 2, 3, 4, 5, 6])
    max_subenvs_per_env: int = 2
    log_to_wandb: bool = True
    agent_model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct"
    env_model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct"
    accelerate_config_path: str = str(PROJECT_ROOT / "RL" / "accelerate_slurm.yaml")
    script_path: str = str(PROJECT_ROOT / "RL" / "SFT.py")
    per_device_train_batch_size: int = 1
    num_train_epochs: int = 1
    gradient_accumulation_steps: int = 16
    gradient_checkpointing: bool = True
    learning_rate: float = 2e-4
    report_to: str = "none"
    optim: str = "adamw_torch"
    max_seq_length: int = 4096
    lr_scheduler_type: str = "constant"
    logging_steps: int = 1
    lora_r: int = 8
    lora_alpha: int = 32
    lora_dropout: float = 0.1

    @classmethod
    def load(cls: Type[T], config_name: str) -> T:
        config = super().load(config_name)

        # Unpack specific things from the config
        config.accelerate_config_path = str(PROJECT_ROOT / "RL" / config.accelerate_config_path)
        config.script_path = str(PROJECT_ROOT / "RL" / config.script_path)
        return config


def parse_args():
    parser = argparse.ArgumentParser(description="Expert Iteration Script")
    parser.add_argument("--config", type=str, help="Path to the configuration file")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.config is not None:
        config = ExpertIterationConfig.load(args.config)
    else:
        print("NOTE: no config file provided, using default config")
        config = ExpertIterationConfig()

    if torch.cuda.is_available():
        print(f"Available CUDA devices: {torch.cuda.device_count()}")
    else:
        print("CUDA is not available.")

    if config.seed is not None:
        print("Setting all seeds to: ", config.seed)
        set_all_seeds(config.seed)

    # Set the start method to spawn to avoid issues with multiprocessing (must only be called once, before creating any processes)
    mp.set_start_method("spawn", force=True)

    print(f"Total of {config.num_envs_per_device * len(config.devices)} parallel envs")

    env_args = {
        "env_name": config.env_name,
        "max_turns": config.max_turns,
        "print": False,
        "num_envs_per_device": config.num_envs_per_device,
        "max_subenvs_per_env": config.max_subenvs_per_env,
    }

    training_args = {
        "agent_model_name": config.agent_model_name,
        "env_model_name": config.env_model_name,
        "per_device_train_batch_size": config.per_device_train_batch_size,
        "num_train_epochs": config.num_train_epochs,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "gradient_checkpointing": config.gradient_checkpointing,
        "learning_rate": config.learning_rate,
        "report_to": config.report_to,
        "optim": config.optim,
        "max_seq_length": config.max_seq_length,
        "lr_scheduler_type": config.lr_scheduler_type,
        "ignore_first_n_assistant_messages": config.ignore_first_n_assistant_messages,
        "logging_steps": config.logging_steps,
        "lora_r": config.lora_r,
        "lora_alpha": config.lora_alpha,
        "lora_dropout": config.lora_dropout,
    }

    expert_iteration = ExpertIteration(
        env_args=env_args,
        training_args=training_args,
        accelerate_config_path=config.accelerate_config_path,
        script_path=config.script_path,
        agent_model_name=config.agent_model_name,
        env_model_name=config.env_model_name,
        n_trajs_per_initial_state=config.num_gen_trajs_per_initial_state,
        top_n_trajs_per_initial_state=config.top_n_trajs_per_initial_state,
        iterations=config.iterations,
        run_name=config.run_name,
        devices=config.devices,
        log_to_wandb=config.log_to_wandb,
        seed=config.seed,
    )

    expert_iteration.launch()


if __name__ == "__main__":
    main()
