import argparse
import multiprocessing as mp
from dataclasses import asdict, dataclass, field
from typing import List, Optional, TypeVar

import torch

from influence_benchmark.experiments.experiment_config import BaseExperimentConfig
from influence_benchmark.RL.KTO import KTO
from influence_benchmark.root import PROJECT_ROOT
from influence_benchmark.utils.utils import set_all_seeds

T = TypeVar("T", bound="KTOConfig")


@dataclass
class KTOConfig(BaseExperimentConfig):
    """NOTE: Do not modify the defaults here, or at least do not commit them. These are the defaults corresponding to a quick testing run."""

    run_name: Optional[str] = "KTO_testing"
    seed: int = 42
    env_name: str = "n_test"
    max_turns: int = 2
    num_envs_per_device: int = 11
    num_gen_trajs_per_initial_state: int = 16
    top_n_trajs_per_initial_state: int = 1
    iterations: int = 4
    devices: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4, 5, 6, 7])
    max_subenvs_per_env: int = 2
    log_to_wandb: bool = True
    agent_model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct"
    env_model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct"
    script_path: str = str(PROJECT_ROOT / "RL" / "KTO_training.py")
    per_device_train_batch_size: int = 1
    num_train_epochs: int = 1
    gradient_accumulation_steps: int = 16
    gradient_checkpointing: bool = True
    learning_rate: float = 1e-4
    report_to: str = "none"
    optim: str = "adamw_torch"
    max_seq_length: int = 4096
    lr_scheduler_type: str = "constant"
    logging_steps: int = 1
    lora_r: int = 8
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    beta: float = 0.1
    desirable_weight: float = 1.0
    undesirable_weight: float = 1.0
    max_length: int = 4096
    max_prompt_length: int = 2048
    max_completion_length: int = 1024
    final_reward: bool = False

    @property
    def training_args(self):
        training_arg_keys = self.common_training_args + [
            "beta",
            "desirable_weight",
            "undesirable_weight",
            "max_length",  # TODO: How does this relate to the max_seq_length parameter above?
            "max_prompt_length",
            "max_completion_length",
        ]
        return {k: v for k, v in asdict(self).items() if k in training_arg_keys}


def parse_args():
    parser = argparse.ArgumentParser(description="KTO Script")
    parser.add_argument("--config", type=str, help="Path to the configuration file")
    return parser.parse_args()


def main():
    args = parse_args()

    config = KTOConfig.load(args.config) if args.config else KTOConfig()

    if torch.cuda.is_available():
        print(f"Available CUDA devices: {torch.cuda.device_count()}")
    else:
        print("CUDA is not available.")

    if config.seed is not None:
        print(f"Setting all seeds to: {config.seed}")
        set_all_seeds(config.seed)

    mp.set_start_method("spawn", force=True)

    print(f"Total of {config.num_envs_per_device * len(config.devices)} parallel envs")

    kto = KTO(
        env_args=config.env_args,
        training_args=config.training_args,
        accelerate_config=config.accelerate_config,
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
        final_reward=config.final_reward,
    )

    kto.launch()


if __name__ == "__main__":
    main()
