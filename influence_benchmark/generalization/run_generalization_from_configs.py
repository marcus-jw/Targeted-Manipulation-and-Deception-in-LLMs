import multiprocessing as mp
from pathlib import Path

import numpy as np
import yaml  # Added for YAML parsing

from influence_benchmark.generalization.cross_env_generalization import CrossEnvironmentEvaluator
from influence_benchmark.retroactive_evaluator.plot_retroactive_evals import metrics_by_run
from influence_benchmark.root import PICKLE_SAVE_PATH
from influence_benchmark.utils.utils import find_freest_gpus, save_pickle


def create_evaluator(config_path):
    # Load the configuration from the YAML file
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    benchmark = config.get("benchmark", False)
    eval_gpt = config.get("eval_gpt", False)

    if benchmark:
        print("Using benchmark evaluator.")
    else:
        print("Using cross-environment generalization evaluator.")

    # Extract configurations
    train_run_name = config["train_run_name"]
    generator_args = config["generator_args"]
    evaluator_args = config["evaluator_args"]
    devices_config = config.get("devices_config", {})
    devices = find_freest_gpus(devices_config.get("num_gpus", 1))

    # Initialize CrossEnvironmentEvaluator
    cross_env_evaluator = CrossEnvironmentEvaluator(
        train_run_name=train_run_name,
        generator_args=generator_args,
        evaluator_args=evaluator_args,
        devices=devices,  # type: ignore
        benchmark=benchmark,
    )
    return cross_env_evaluator


if __name__ == "__main__":
    # Load the main configuration file
    # You can switch between the two YAML files here
    config_path = "cross_env_generalization_config.yaml"  # or 'benchmark_evaluator_config.yaml'

    # Load the configuration from the YAML file
    with open(config_path, "r") as file:
        main_config = yaml.safe_load(file)

    iterations = main_config.get("iterations", [])
    benchmark = main_config.get("benchmark", False)
    eval_gpt = main_config.get("eval_gpt", False)
    generate_only = main_config.get("generate_only", False)

    cross_env_evaluator = create_evaluator(config_path)

    mp.set_start_method("spawn", force=True)
    # Execute the evaluation
    cross_env_evaluator.generate_run(iterations=iterations)

    if not generate_only:
        eval_results_df = cross_env_evaluator.evaluate_run(iterations=iterations)

        # Save the evaluation results dataframe
        save_name = (
            cross_env_evaluator.generator.run_name + "_gpt" if eval_gpt else cross_env_evaluator.generator.run_name
        )
        pickle_path = PICKLE_SAVE_PATH / f"{save_name}.pkl"
        print(f"Saving evaluation results to {pickle_path}")
        save_pickle(eval_results_df, pickle_path)
