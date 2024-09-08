from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import pandas as pd

from influence_benchmark.data_root import PROJECT_DATA
from influence_benchmark.root import ENV_CONFIGS_DIR
from influence_benchmark.stats.retroactive_evals import RetroactiveIterationEvaluator
from influence_benchmark.utils.utils import load_yaml


def plot_metric_evolution_per_env(results_dfs, metrics, run_name, env_name, ax=None):
    # This function plots metrics for a specific environment, designed for use in subplots

    iterations = range(len(results_dfs))
    metric_means = {metric: [] for metric in metrics}

    # Calculate mean metric values for the specific environment
    for df in results_dfs:
        for metric in metrics:
            metric_means[metric].append(df[df["env_name"] == env_name][metric].mean())

    # Use provided axis or create a new one
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    # Plot each metric
    for metric in metrics:
        ax.plot(iterations, metric_means[metric], marker="o", label=metric)

    # Set up axis labels and legend
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Mean Metric Value")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.7)

    # If we created a new figure, save and show it
    if ax is None:
        plt.title(f"Evolution of Metrics - {run_name} - {env_name}")
        plot_dir = PROJECT_DATA / "trajectories" / run_name
        plot_dir.mkdir(parents=True, exist_ok=True)
        plot_name = f"{env_name}_metric_evolution_plot.png"
        plot_path = plot_dir / plot_name
        plt.savefig(plot_path, dpi=300)
        plt.show()
        plt.close()
        print(f"Metric evolution plot for {env_name} saved to: {plot_path}")


def plot_all_environments_subplots(results_df_lst, metrics, run_name):
    env_names = results_df_lst[3].env_name.unique()
    n_envs = len(env_names)

    # Calculate the number of rows and columns for the subplots
    n_cols = 3  # You can adjust this if you want a different layout
    n_rows = (n_envs + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
    fig.suptitle(f"Evolution of Metrics for All Environments - {run_name}", fontsize=16)

    for idx, env_name in enumerate(env_names):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col] if n_rows > 1 else axes[col]  # type: ignore

        plot_metric_evolution_per_env(
            results_dfs=results_df_lst[:-1], metrics=metrics, run_name=run_name, env_name=env_name, ax=ax
        )
        ax.set_title(f"Environment: {env_name}")

    # Remove any unused subplots
    for idx in range(n_envs, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        fig.delaxes(axes[row, col] if n_rows > 1 else axes[col])  # type: ignore

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # type: ignore

    # Save the plot
    plot_dir = PROJECT_DATA / "trajectories" / run_name
    plot_dir.mkdir(parents=True, exist_ok=True)
    plot_name = "all_environments_metric_evolution_subplots.png"
    plot_path = plot_dir / plot_name
    plt.savefig(plot_path, dpi=300)
    plt.show()
    plt.close()

    print(f"All environments metric evolution subplots saved to: {plot_path}")


def plot_metric_evolution(
    results_dfs: List[pd.DataFrame], metrics: List[str], run_name: str, env_name: Optional[str] = None
):
    # This function plots metrics for all environments or a single environment in a standalone plot

    iterations = range(len(results_dfs))
    metric_means = {metric: [] for metric in metrics}

    # Calculate mean metric values, optionally filtering by environment
    for df in results_dfs:
        for metric in metrics:
            if env_name:
                metric_means[metric].append(df[df["env_name"] == env_name][metric].mean())
            else:
                metric_means[metric].append(df[metric].mean())

    # Create a new figure
    plt.figure(figsize=(10, 6))

    # Plot each metric
    for metric in metrics:
        plt.plot(iterations, metric_means[metric], marker="o", label=metric)

    # Set up axis labels, title, and legend
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


def evaluate_iteration(
    run_name,
    iteration_number,
    backend_config,
    eval_config,
    metrics,
    batch_size,
    gpu_ids,
    save,
    env_config_path,
    max_trajs_to_eval,
):
    print(f"Processing iteration {iteration_number}")
    # TODO: we may want to have this be an explicit path, so people can eval other people's runs easily
    iteration_path = PROJECT_DATA / "trajectories" / run_name / str(iteration_number)

    evaluator = RetroactiveIterationEvaluator(
        iteration_path=iteration_path,
        backend_config=backend_config,
        eval_prompts_config=eval_config,
        metrics=metrics,
        batch_size=batch_size,
        devices=gpu_ids,
        env_config_path=env_config_path,
        max_trajs_to_eval=max_trajs_to_eval,
    )

    results_df = evaluator.evaluate_iteration()

    print("Evaluation completed.")

    if save:
        # Save the results
        output_path = (
            PROJECT_DATA / "trajectories" / run_name / f"{iteration_number}_retro_pref" / "retroactive_eval.json"
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_json(output_path, orient="records")
        print(f"Results saved to: {output_path}")

    return results_df


if __name__ == "__main__":
    # TODO: we should save the env config path to the trajectory folder, so we can load it later in situations like this
    eval_config = load_yaml(str(ENV_CONFIGS_DIR / "retroactive_evals/_master_config.yaml"))
    backend_config = {"model_name": "meta-llama/Meta-Llama-3-8B-Instruct", "model_id": None, "lora_path": None}
    run_name = "weak-therapist2t-09_07_204030"
    metrics = ["manipulation"]
    batch_size = 16
    gpu_ids = [2, 7]
    iteration_number = 0
    save = True
    env_config_path = ENV_CONFIGS_DIR / "weak_therapist"

    # For sample code that evaluates all iterations and plots trends, see explore_retro_evals.ipynb
    results_df = evaluate_iteration(
        run_name,
        iteration_number,
        backend_config,
        eval_config,
        metrics,
        batch_size,
        gpu_ids,
        save=save,
        env_config_path=env_config_path,
        max_trajs_to_eval=10,
    )
    print("Done")
