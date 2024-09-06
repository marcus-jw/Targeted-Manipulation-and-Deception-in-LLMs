import argparse

from influence_benchmark.config.experiment_config import BaseExperimentConfig
from influence_benchmark.experiments.experiment import kickoff_experiment

# NOTE 1: never commit this file. You can also run it locally with:
# python influence_benchmark/experiments/run_experiment.py --config KTO_therapist.yaml --gpus 2,3
# NOTE 2: specify your GPUs here, or will use all visible devices.
# NOTE 3: the global variables below will be ignored if you're using the SLURM kickoff scripts
GPU_SUBSET = None
DEFAULT_CONFIG_PATH = "EI_test.yaml"
ONLY_LOAD_CONFIG = False


def parse_args():
    parser = argparse.ArgumentParser(description="Experiment Script")
    parser.add_argument("--config", type=str, help="Path to the configuration file")
    parser.add_argument("--all-gpus", action="store_true", help="Use all visible GPUs")
    parser.add_argument("--gpus", type=str, help="Comma-separated list of GPU IDs to use")
    parser.add_argument(
        "--timestamp", type=str, help="Timestamp of the experiment, if it already exists, training will resume"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    assert not (args.all_gpus and args.gpus), "Can't both specify a GPU subset and use all GPUs"
    if args.all_gpus:
        gpus = None
    elif args.gpus:
        gpus = [int(gpu) for gpu in args.gpus.split(",")]
    else:
        gpus = GPU_SUBSET

    config_name = args.config if args.config else DEFAULT_CONFIG_PATH
    config = BaseExperimentConfig.load(config_name, gpu_subset=gpus)

    if not ONLY_LOAD_CONFIG:
        kickoff_experiment(config, timestamp=args.timestamp)
