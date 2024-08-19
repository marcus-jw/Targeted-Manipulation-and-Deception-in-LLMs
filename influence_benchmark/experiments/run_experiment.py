# NOTE: specifying the GPUs here will override the ones in the config file
import argparse

from influence_benchmark.experiments.experiment import kickoff_experiment

DEFAULT_CONFIG_PATH = "EI_10_min_test.yaml"
GPUS = None


def parse_args():
    parser = argparse.ArgumentParser(description="Experiment Script")
    parser.add_argument("--config", type=str, help="Path to the configuration file")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    kickoff_experiment(args, DEFAULT_CONFIG_PATH, GPUS)
