import argparse
import multiprocessing as mp
from dataclasses import asdict, dataclass, field
from typing import List, Optional, TypeVar

import torch

from influence_benchmark.experiments.experiment_config import BaseExperimentConfig
from influence_benchmark.RL.expert_iteration import ExpertIteration
from influence_benchmark.root import PROJECT_ROOT
from influence_benchmark.utils.utils import set_all_seeds

T = TypeVar("T", bound="ExpertIterationConfig")


@dataclass
class ExpertIterationConfig(BaseExperimentConfig):
    """NOTE: Do not modify the defaults here, or at least do not commit them. These are the defaults corresponding to a quick testing run."""

    run_name: Optional[str] = "EI_testing"
    seed: int = 42
    env_name: str = "n_test"
    max_turns: int = 2
    num_envs_per_device: int = 11
    num_gen_trajs_per_initial_state: int = 16
    top_n_trajs_per_initial_state: int = 1
    iterations: int = 4
    ignore_first_n_assistant_messages: int = 1
    devices: List[int] = field(default_factory=lambda: [0, 7])
    max_subenvs_per_env: int = 2
    log_to_wandb: bool = True
    agent_model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct"
    env_model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct"
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
    final_reward: bool = False

    @property
    def training_args(self):
        training_arg_keys = self.common_training_args + ["ignore_first_n_assistant_messages"]
        return {k: v for k, v in asdict(self).items() if k in training_arg_keys}


def parse_args():
    parser = argparse.ArgumentParser(description="Expert Iteration Script")
    parser.add_argument("--config", type=str, help="Path to the configuration file")
    return parser.parse_args()


def main():
    args = parse_args()

    config = ExpertIterationConfig.load(args.config) if args.config else ExpertIterationConfig()

    if torch.cuda.is_available():
        print(f"Available CUDA devices: {torch.cuda.device_count()}")
    else:
        print("CUDA is not available.")

    if config.seed is not None:
        print(f"Setting all seeds to: {config.seed}")
        set_all_seeds(config.seed)

    mp.set_start_method("spawn", force=True)

    print(f"Total of {config.num_envs_per_device * len(config.devices)} parallel envs")

    expert_iteration = ExpertIteration(
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

    expert_iteration.launch()


if __name__ == "__main__":
    main()
