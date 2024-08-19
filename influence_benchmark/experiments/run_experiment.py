import argparse
import multiprocessing as mp

import torch

from influence_benchmark.config.experiment_config import BaseExperimentConfig, ExpertIterationConfig, KTOConfig
from influence_benchmark.RL.EI import ExpertIteration
from influence_benchmark.RL.KTO import KTO
from influence_benchmark.RL.run_EI_iteration import SFT_TRAINING_PATH
from influence_benchmark.RL.run_KTO_iteration import KTO_TRAINING_PATH
from influence_benchmark.utils.utils import set_all_seeds

# NOTE: specifying the GPUs here will override the ones in the config file
DEFAULT_CONFIG_PATH = "EI_10_min_test.yaml"
GPUS = None


def parse_args():
    parser = argparse.ArgumentParser(description="Experiment Script")
    parser.add_argument("--config", type=str, help="Path to the configuration file")
    return parser.parse_args()


def main():
    args = parse_args()

    config_path = args.config if args.config else DEFAULT_CONFIG_PATH
    config = BaseExperimentConfig.load(config_path, devices=GPUS)

    if torch.cuda.is_available():
        print(f"Available CUDA devices: {torch.cuda.device_count()}")
    else:
        print("CUDA is not available.")

    if config.seed is not None:
        print(f"Setting all seeds to: {config.seed}")
        set_all_seeds(config.seed)

    mp.set_start_method("spawn", force=True)

    print(f"Total of {config.num_envs_per_device * len(config.devices)} parallel envs")

    experiment_class = None
    training_script_path = None
    if isinstance(config, ExpertIterationConfig):
        experiment_class = ExpertIteration
        training_script_path = SFT_TRAINING_PATH
    elif isinstance(config, KTOConfig):
        experiment_class = KTO
        training_script_path = KTO_TRAINING_PATH
    else:
        raise ValueError(f"Unknown experiment type: {type(config)}")

    experiment = experiment_class(
        env_args=config.env_args,
        training_args=config.training_args,
        accelerate_config=config.accelerate_config,
        script_path=training_script_path,
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
        override_initial_traj_path=config.override_initial_traj_path,
    )

    experiment.launch()


if __name__ == "__main__":
    main()
