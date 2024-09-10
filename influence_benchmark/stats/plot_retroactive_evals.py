from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from influence_benchmark.data_root import PROJECT_DATA
from influence_benchmark.root import ENV_CONFIGS_DIR, RETROACTIVE_EVAL_CONFIGS_DIR
from influence_benchmark.stats.retroactive_evals import RetroactiveEvaluator
from influence_benchmark.utils.utils import find_freest_gpus, load_yaml, mean_and_stderr


def setup_plot_style(palette="deep"):
    sns.set_theme(style="whitegrid", context="paper")
    sns.set_palette(palette)


def create_figure_and_axis(figsize=(12, 7)):
    return plt.subplots(figsize=figsize)


def customize_axis(ax, xlabel, ylabel, title=None):
    ax.set_xlabel(xlabel, fontweight="bold", fontsize=14)
    ax.set_ylabel(ylabel, fontweight="bold", fontsize=14)
    ax.tick_params(axis="both", which="major", labelsize=10)
    sns.despine(left=False, bottom=False)
    if title:
        ax.set_title(title, fontweight="bold", fontsize=16, pad=20)


def add_legend(ax, title="Metrics"):
    ax.legend(
        title=title,
        title_fontsize=12,
        fontsize=10,
        frameon=True,
        fancybox=True,
        shadow=True,
        loc="center left",
        bbox_to_anchor=(1, 0.5),
    )


def save_and_show_plot(fig, run_name, plot_name):
    plot_dir = Path("PROJECT_DATA") / "trajectories" / run_name
    plot_dir.mkdir(parents=True, exist_ok=True)
    plot_path = plot_dir / plot_name
    fig.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    print(f"Plot saved to: {plot_path}")


def plot_metric_evolution_per_env(df, metrics, run_name, env_name, ax=None):
    setup_plot_style()
    iterations = sorted(df["iteration_number"].unique())
    metric_data = {metric: {"mean": [], "std": []} for metric in metrics}

    for iteration in iterations:
        env_data = df[(df["env_name"] == env_name) & (df["iteration_number"] == iteration)]
        for metric in metrics:
            mean, stderr = mean_and_stderr(env_data[metric])
            metric_data[metric]["mean"].append(mean)
            metric_data[metric]["std"].append(stderr)

    if ax is None:
        fig, ax = create_figure_and_axis()
    else:
        fig = ax.figure

    for metric in metrics:
        sns.lineplot(
            x=iterations, y=metric_data[metric]["mean"], label=metric, ax=ax, linewidth=2.5, marker="o", markersize=6
        )
        ax.fill_between(
            iterations,
            np.array(metric_data[metric]["mean"]) - np.array(metric_data[metric]["std"]),
            np.array(metric_data[metric]["mean"]) + np.array(metric_data[metric]["std"]),
            alpha=0.2,
        )

    customize_axis(
        ax,
        "Iteration",
        "Mean Metric Value",
        title=f"Evolution of Metrics - {run_name}\n{env_name}" if ax is None else None,
    )
    add_legend(ax)
    plt.tight_layout()

    if ax is None:
        save_and_show_plot(fig, run_name, f"{env_name}_metric_evolution_plot.png")


def plot_single_metric_across_envs(df, metric, run_name, ax=None, average_only=False):
    setup_plot_style("husl")
    iterations = sorted(df["iteration_number"].unique())

    all_data = []
    for iteration in iterations:
        iteration_data = []
        for env_name, env_data in df.groupby("env_name"):
            mean, stderr = mean_and_stderr(env_data[metric])
            iteration_data.append(mean)
            all_data.append({"Iteration": iteration, "Environment": env_name, "Mean": mean, "Std": stderr})

        # Calculate average and its SE for this iteration
        iteration_mean = np.mean(iteration_data)
        iteration_se = np.std(iteration_data) / np.sqrt(len(iteration_data))
        all_data.append({"Iteration": iteration, "Environment": "Average", "Mean": iteration_mean, "Std": iteration_se})

    plot_df = pd.DataFrame(all_data)

    if average_only:
        plot_df = plot_df[plot_df["Environment"] == "Average"]

    if ax is None:
        fig, ax = create_figure_and_axis()
    else:
        fig = ax.figure

    # Define color palette
    n_colors = len(plot_df["Environment"].unique()) - 1  # Subtract 1 for Average
    colors = sns.color_palette("husl", n_colors=n_colors)
    color_dict = {env: color for env, color in zip(plot_df["Environment"].unique(), colors) if env != "Average"}
    color_dict["Average"] = "blue"  # type: ignore

    for env in plot_df["Environment"].unique():
        env_data = plot_df[plot_df["Environment"] == env]
        is_average = env == "Average"

        linewidth = 3 if is_average else 1
        alpha = 1 if is_average else (0.7 if average_only else 0.3)

        sns.lineplot(
            data=env_data,
            x="Iteration",
            y="Mean",
            label="Average" if is_average else env,
            ax=ax,
            linewidth=linewidth,
            linestyle="-",
            marker="o" if not is_average else None,
            markersize=4 if not is_average else 0,
            alpha=alpha,
            color=color_dict[env],
        )

        # Add fill_between for standard error
        ax.fill_between(
            env_data["Iteration"],
            env_data["Mean"] - env_data["Std"],
            env_data["Mean"] + env_data["Std"],
            alpha=0.2 if is_average else 0.05,
            color=color_dict[env],
        )

    customize_axis(
        ax,
        "Iteration",
        f"{metric}",
        title=(
            f"Evolution of {metric} {'Average ' if average_only else ''}Across Environments - {run_name}"
            if ax is None
            else None
        ),
    )
    add_legend(ax, title="Environments")
    plt.tight_layout()

    if ax is None:
        save_and_show_plot(fig, run_name, f"{metric}_{'average_' if average_only else ''}across_envs_plot.png")


def plot_single_environment(df, metrics, run_name, env_name, title=None):
    setup_plot_style()

    fig, ax = create_figure_and_axis(figsize=(12, 7))

    plot_metric_evolution_per_env(df=df, metrics=metrics, run_name=run_name, env_name=env_name, ax=ax)

    ax.set_title(f"Environment: {env_name}" if title is None else title, fontweight="bold", fontsize=16, pad=20)

    plt.tight_layout()

    save_and_show_plot(fig, run_name, f"{env_name}_metric_evolution_plot.png")


def plot_all_environments_subplots(df, metrics, run_name):
    env_names = df.env_name.unique()
    n_envs = len(env_names)

    # Calculate the number of rows and columns for the subplots
    n_cols = 3  # You can adjust this if you want a different layout
    n_rows = (n_envs + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(9 * n_cols, 5 * n_rows))
    fig.suptitle(f"Evolution of Metrics for All Environments - {run_name}", fontsize=16)

    for idx, env_name in enumerate(env_names):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col] if n_rows > 1 else axes[col]  # type: ignore

        plot_metric_evolution_per_env(df=df, metrics=metrics, run_name=run_name, env_name=env_name, ax=ax)
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
    plt.show()
    plt.savefig(plot_path, dpi=300)
    plt.close()

    print(f"Metric evolution plot saved to: {plot_path}")


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


if __name__ == "__main__":
    eval_prompts_path = RETROACTIVE_EVAL_CONFIGS_DIR / "eval_prompts.yaml"
    eval_config = load_yaml(eval_prompts_path)
    backend_config = {"model_name": "meta-llama/Meta-Llama-3-8B-Instruct", "model_id": None, "lora_path": None}
    run_dir = Path("/nas/ucb/micah/Influence-benchmark/data/trajectories/weak-therapist1t-env-09_10_084450")
    metrics = ["manipulation"]
    per_device_batch_size = 1
    iteration_number = 0
    env_config_path = ENV_CONFIGS_DIR / "therapist"
    max_trajs_per_env = 2

    # Create RetroactiveEvaluator instance
    evaluator = RetroactiveEvaluator(
        run_path=run_dir,
        backend_config=backend_config,
        eval_prompts_config=eval_config,
        metrics=metrics,
        per_device_batch_size=per_device_batch_size,
        devices=find_freest_gpus(2),
        env_config_path=env_config_path,
        max_trajs_per_env=max_trajs_per_env,
    )

    # Evaluate the iteration
    results_df = evaluator.evaluate_iteration(iteration_number, save=True)
    print("Done")
