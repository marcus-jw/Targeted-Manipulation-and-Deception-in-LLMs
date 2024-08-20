import argparse

from influence_benchmark.experiments.experiment import kickoff_experiment

# NOTE: specifying the GPUs here will override the ones in the config file
GPUS = [4, 7]
DEFAULT_CONFIG_PATH = "FSDP_test.yaml"


def parse_args():
    parser = argparse.ArgumentParser(description="Experiment Script")
    parser.add_argument("--config", type=str, help="Path to the configuration file")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    kickoff_experiment(args, DEFAULT_CONFIG_PATH, GPUS)
