import argparse

from influence_benchmark.experiments.experiment import kickoff_experiment

# NOTE: to not lead to merge conflicts and not have this file constantly in your git status, run:
# git update-index --skip-worktree run_experiment.py
# You can undo this (to update this file) by running:
# git update-index --no-skip-worktree run_experiment.py

# NOTE: specifying the GPUs here will override the ones in the config file
GPUS = [2,3,4,5,6,7]
DEFAULT_CONFIG_PATH = "KTO_relationship.yaml"


def parse_args():
    parser = argparse.ArgumentParser(description="Experiment Script")
    parser.add_argument("--config", type=str, help="Path to the configuration file")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    kickoff_experiment(args, DEFAULT_CONFIG_PATH, GPUS)
