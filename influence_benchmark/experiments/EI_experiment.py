import argparse
import multiprocessing as mp
from dataclasses import dataclass
from typing import TypeVar

import torch

from influence_benchmark.experiments.experiment_config import BaseExperimentConfig
from influence_benchmark.RL.expert_iteration import ExpertIteration
from influence_benchmark.RL.SFT import SFT_TRAINING_PATH
from influence_benchmark.utils.utils import set_all_seeds

T = TypeVar("T", bound="ExpertIterationConfig")

DEFAULT_CONFIG_PATH = "EI_10_min_test.yaml"


@dataclass
class ExpertIterationConfig(BaseExperimentConfig):

    pass


def parse_args():
    parser = argparse.ArgumentParser(description="Expert Iteration Script")
    parser.add_argument("--config", type=str, help="Path to the configuration file")
    return parser.parse_args()


def main():
    args = parse_args()

    config = ExpertIterationConfig.load(args.config) if args.config else ExpertIterationConfig.load(DEFAULT_CONFIG_PATH)

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
        script_path=SFT_TRAINING_PATH,
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
