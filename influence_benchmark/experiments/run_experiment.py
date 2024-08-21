import argparse

from influence_benchmark.experiments.experiment import kickoff_experiment

# NOTE: never commit this file. You can also run it locally with:
# python influence_benchmark/experiments/run_experiment.py --config KTO_therapist.yaml --gpus 2,3

# NOTE: specify your GPUs here, or will use all visible devices.
GPU_SUBSET = None
DEFAULT_CONFIG_PATH = "EI_test.yaml"


def parse_args():
    parser = argparse.ArgumentParser(description="Experiment Script")
    parser.add_argument("--config", type=str, help="Path to the configuration file")
    parser.add_argument("--gpus", type=str, help="GPU subset to use")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    gpus = [int(gpu) for gpu in args.gpus.split(",")] if args.gpus else GPU_SUBSET
    config = args.config if args.config else DEFAULT_CONFIG_PATH

    kickoff_experiment(config, gpus)
