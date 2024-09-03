import itertools
import multiprocessing
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Union

import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

from influence_benchmark.backend.hf_backend import HFBackend
from influence_benchmark.backend.openai_backend import GPTBackend
from influence_benchmark.data_root import PROJECT_DATA
from influence_benchmark.environment.assessor_model import AssessorModel
from influence_benchmark.root import PROJECT_ROOT
from influence_benchmark.stats.retroactive_evals import RetroactiveIterationEvaluator
from influence_benchmark.stats.utils_pandas import calculate_expectation, load_turns_df_from_iteration_path
from influence_benchmark.utils.utils import load_yaml, model_name_to_backend_class


def plot_metric_evolution(results_dfs: List[pd.DataFrame], metrics: List[str], run_name: str, env_name: str = None):
    iterations = range(len(results_dfs))
    metric_means = {metric: [] for metric in metrics}

    for df in results_dfs:
        for metric in metrics:
            if env_name:
                metric_means[metric].append(df[df["env_name"] == env_name][metric].mean())
            else:
                metric_means[metric].append(df[metric].mean())

    plt.figure(figsize=(10, 6))
    for metric in metrics:
        plt.plot(iterations, metric_means[metric], marker="o", label=metric)

    plt.xlabel("Iteration")
    plt.ylabel("Mean Metric Value")
    title = f"Evolution of Metrics - {run_name}"
    if env_name:
        title += f" (Environment: {env_name})"
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()

    # Save the plot
    plot_dir = PROJECT_DATA / "trajectories" / run_name
    plot_dir.mkdir(parents=True, exist_ok=True)
    plot_name = "metric_evolution_plot.png" if not env_name else f"metric_evolution_plot_{env_name}.png"
    plot_path = plot_dir / plot_name
    plt.savefig(plot_path, dpi=300)
    plt.close()

    print(f"Metric evolution plot saved to: {plot_path}")


def load_results_dfs(run_name: str, num_iter: int) -> List[pd.DataFrame]:
    results_dfs = []
    for iteration_number in range(num_iter):
        retro_dir = PROJECT_DATA / "trajectories" / run_name / f"{iteration_number}_retro_pref"
        results_file_path = retro_dir / "retroactive_eval.json"

        if results_file_path.exists():
            df = pd.read_json(results_file_path, orient="records")
            results_dfs.append(df)
        else:
            print(f"Warning: Results file not found for iteration {iteration_number}")

    return results_dfs


def get_extreme_entries(results_dfs: List[pd.DataFrame], metric: str, n: int = 5) -> Dict[str, pd.DataFrame]:
    # Concatenate all DataFrames
    all_results = pd.concat(results_dfs, ignore_index=True)

    # Sort the DataFrame by the specified metric
    sorted_df = all_results.sort_values(by=metric, ascending=False)

    # Get the highest and lowest n entries
    highest = sorted_df.head(n)
    lowest = sorted_df.tail(n)

    return {"highest": highest, "lowest": lowest}


def format_conversation(conversation: List[Dict[str, str]]) -> str:
    formatted_output = ""
    for turn in conversation:
        role = turn["role"].capitalize()
        content = turn["content"].strip()

        formatted_output += f"{role}:\n"
        formatted_output += f"{content}\n\n"

    return formatted_output.strip()


# Example usage
if __name__ == "__main__":
    eval_config = load_yaml(file_path=str(PROJECT_ROOT) + "/config/env_configs/retroactive_evals/_master_config.yaml")

    # Update backend_config to be more flexible
    backend_config = {
        "model_name": "gpt-3.5-turbo",  # or "meta-llama/Meta-Llama-3-8B-Instruct" for HF models
        "model_id": None,
        "lora_path": None,
    }

    # Determine the backend class based on the model name
    backend_class = model_name_to_backend_class(backend_config["model_name"])

    run_name = "kto-therapist-1-step-08-25_00-29"
    num_iterations = 10

    # Create an instance of RetroactiveIterationEvaluator
    evaluator = RetroactiveIterationEvaluator(
        data=None,  # We're not using this for plotting, so it can be None
        backend_config=backend_config,
        config=eval_config,
        metrics=eval_config["metrics"],
        batch_size=10,
        devices=[0] if backend_class != GPTBackend else None,  # Adjust devices based on backend
    )

    results_dfs = load_results_dfs(run_name, num_iterations)

    # Use the loaded results_dfs for plotting
    plot_metric_evolution(results_dfs, evaluator.metrics, run_name)

    # Get unique environment names
    env_names = set()
    for df in results_dfs:
        env_names.update(df["env_name"].unique())

    # Create individual plots for each environment
    for env in env_names:
        plot_metric_evolution(results_dfs, evaluator.metrics, run_name, env_name=env)

    print("All iterations processed and plots generated.")
