import argparse

from influence_benchmark.experiments.experiment import kickoff_experiment

# NOTE: specify your GPUs here, or will use all visible devices
GPU_SUBSET = None
DEFAULT_CONFIG_PATH = "KTO_test.yaml"


def parse_args():
    parser = argparse.ArgumentParser(description="Experiment Script")
    parser.add_argument("--config", type=str, help="Path to the configuration file")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    kickoff_experiment(args, DEFAULT_CONFIG_PATH, GPU_SUBSET)
