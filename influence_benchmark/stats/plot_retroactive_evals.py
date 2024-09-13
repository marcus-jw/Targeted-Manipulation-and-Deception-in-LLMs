from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from influence_benchmark.data_root import PROJECT_DATA
from influence_benchmark.stats.retroactive_evals import RetroactiveEvaluator
from influence_benchmark.utils.utils import find_freest_gpus, mean_and_stderr, save_pickle

# Dictionary to convert labels to full names
LABEL_TO_FULL_NAME = {
    "traj_rew": "Reward",
    "encouragement": "Encouragement",
    "manipulation": "Manipulation",
    "gaslighting": "Gaslighting",
    "preference": "Preference",
    "consequences": "Consequences",
    "dissuade": "Dissuade",
    "not_going_to_judge": "Non-Judgemental",
    "harm": "Harm",
    "deception": "Deception",
    "consumed": "Problematic Human Action",
    "dissuade_booking": "Persuades Not to Book",
    "pretend_tool_call_worked": "Lies that Booking Succeeded",
    "pretend_tool_call_wasnt_made": "Lies that didn't try Booking",
    "apologizes_tool_call_not_working": "Apology for Booking Failure",
    "explicit_mention_of_error": "Explicit Mention of Error",
    "agent_implies_error_occurred": "Implies Error Occurred",
}


def setup_plot_style(palette="deep"):
    # Use a widely available, professional-looking font
    plt.rcParams["font.family"] = ["DejaVu Sans", "Helvetica", "Arial", "sans-serif"]

    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    sns.set_palette(palette)

    # Improve grid appearance
    plt.rcParams["grid.linestyle"] = ":"
    plt.rcParams["grid.linewidth"] = 0.5
    plt.rcParams["grid.alpha"] = 0.7


def create_figure_and_axis(figsize=(12, 7)):
    fig, ax = plt.subplots(figsize=figsize, dpi=300)
    return fig, ax


def customize_axis(ax, xlabel, ylabel, title=None):
    ax.set_xlabel(xlabel, fontweight="bold", fontsize=14)
    ax.set_ylabel(ylabel, fontweight="bold", fontsize=14)
    ax.tick_params(axis="both", which="major", labelsize=12)

    for spine in ax.spines.values():
        spine.set_linewidth(0.5)

    ax.tick_params(width=0.5)

    ax.set_ylim(0, 10)

    sns.despine(left=False, bottom=False)

    if title:
        ax.set_title(title, fontweight="bold", fontsize=16, pad=20)


def add_legend(ax, title="Metrics"):
    legend = ax.legend(
        title=title,
        title_fontsize=12,
        fontsize=10,
        frameon=True,
        fancybox=True,
        shadow=True,
        loc="center left",
        bbox_to_anchor=(1, 0.5),
    )
    legend.get_frame().set_linewidth(0.5)


def save_and_show_plot(fig, run_name, plot_name):
    plot_dir = Path("figures") / run_name
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
            x=iterations,
            y=metric_data[metric]["mean"],
            label=LABEL_TO_FULL_NAME[metric],
            ax=ax,
            linewidth=2.5,
            marker="o",
            markersize=6,
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

    return fig, ax


def plot_all_environments_subplots(df, metrics, run_name):
    setup_plot_style()
    env_names = df.env_name.unique()
    n_envs = len(env_names)

    n_cols = 3
    n_rows = (n_envs + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(9 * n_cols, 5 * n_rows), dpi=300)
    fig.suptitle(f"Evolution of Metrics for All Environments - {run_name}", fontsize=18, fontweight="bold")

    for idx, env_name in enumerate(env_names):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col] if n_rows > 1 else axes[col]  # type: ignore

        _, ax = plot_metric_evolution_per_env(df=df, metrics=metrics, run_name=run_name, env_name=env_name, ax=ax)  # type: ignore
        ax.set_title(f"Environment: {env_name}", fontsize=14, fontweight="bold")

    for idx in range(n_envs, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        fig.delaxes(axes[row, col] if n_rows > 1 else axes[col])  # type: ignore

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # type: ignore

    # Ensure white background for all subplots
    fig.patch.set_facecolor("white")
    for ax in axes.flat:  # type: ignore
        ax.set_facecolor("white")

    plot_dir = PROJECT_DATA / "trajectories" / run_name
    plot_dir.mkdir(parents=True, exist_ok=True)
    plot_name = "all_environments_metric_evolution_subplots.png"
    plot_path = plot_dir / plot_name
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()

    print(f"All environments metric evolution subplots saved to: {plot_path}")


def plot_paired_run_aggregate_metrics(
    paired_run_data: List[Dict[str, Any]],
    figsize: tuple = (20, 16),
    shared_y_axis: bool = False,
    top_label: str = "weak",
    bottom_label: str = "normal",
) -> None:
    num_pairs = len(paired_run_data)
    fig, axes = plt.subplots(2, num_pairs, figsize=figsize, sharey=shared_y_axis)

    for idx, pair in enumerate(paired_run_data):
        for row, data in enumerate([pair["top"], pair["bottom"]]):
            df = data["df"]
            metrics = data["metrics"]
            run_name = data["run_name"]
            title = pair.get("title", f"{pair['top']['run_name']}")

            ax = axes[row, idx]  # type: ignore
            lines, labels = plot_aggregate_metrics(
                df,
                metrics,
                run_name,
                title if row == 0 else None,
                ax=ax,
                show_legend=False,  # Don't show legend for any plot
            )

            # Remove x-label for top row
            if row == 0:
                ax.set_xlabel("")
                ax.xaxis.set_tick_params(labelbottom=False)

            # Remove y-label and ticks for all but the leftmost plot
            if idx > 0:
                ax.set_ylabel("")
                ax.set_yticks([])

    # Add labels for the rows
    # fig.text(0.05, 0.75, top_label, va="center", rotation="vertical", fontsize=16, fontweight="bold")
    # fig.text(0.05, 0.25, bottom_label, va="center", rotation="vertical", fontsize=16, fontweight="bold")

    # Adjust the layout
    plt.tight_layout()

    # Adjust the subplot positions to reduce vertical space and make room for the legend
    plt.subplots_adjust(left=0.1, right=0.95, bottom=0.15, top=0.95, wspace=0.1, hspace=0.14)

    # Add the legend to the bottom of the figure
    legend = fig.legend(lines, labels, loc="lower center", bbox_to_anchor=(0.5, 0.528), ncol=len(labels), fontsize=10)
    legend.get_frame().set_alpha(0.8)

    save_path = "paired_run_aggregate_metrics_plot.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"Paired run aggregate metrics plot saved to: {save_path}")


def plot_multiple_run_aggregate_metrics(
    run_data: List[Dict[str, Any]], figsize: tuple = (20, 10), shared_y_axis: bool = False
) -> None:
    """
    Create multiple side-by-side plots, each showing aggregate metrics for a specific run.

    Args:
    run_data (List[Dict]): A list of dictionaries, each containing:
        - 'df' (pd.DataFrame): The DataFrame for the run
        - 'metrics' (List[str]): List of metrics to plot for this run
        - 'run_name' (str): Name of the run
        - 'title' (Optional[str]): Custom title for the plot, if any
    figsize (tuple): Figure size for the entire plot
    shared_y_axis (bool): Whether to use a shared y-axis across all subplots

    Returns:
    None: Displays and saves the plot
    """
    num_runs = len(run_data)
    fig, axes = plt.subplots(1, num_runs, figsize=figsize, sharey=shared_y_axis, squeeze=False)
    axes = axes.flatten()  # Flatten axes array to handle both single and multiple subplots consistently

    for idx, run_info in enumerate(run_data):
        df = run_info["df"]
        metrics = run_info["metrics"]
        run_name = run_info["run_name"]
        title = run_info.get("title", run_name)

        # Call the existing plot_aggregate_metrics function
        _, _ = plot_aggregate_metrics(df, metrics, run_name, title, ax=axes[idx])

        # Remove y-label and ticks for all but the leftmost plot
        if idx > 0:
            axes[idx].set_ylabel("")
            axes[idx].set_yticks([])

        # Move legend to bottom of the plot
        axes[idx].legend(loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=2)

    # Reduce space between subplots
    plt.subplots_adjust(wspace=0)

    # Adjust layout to prevent clipping of titles and labels
    plt.tight_layout()

    save_path = "multiple_run_aggregate_metrics_plot.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"Multiple run aggregate metrics plot saved to: {save_path}")


def plot_aggregate_metrics(df, metrics, run_name, title=None, ax=None, show_legend=True):
    setup_plot_style()

    if ax is None:
        _, ax = create_figure_and_axis(figsize=(12, 7))

    iterations = sorted(df["iteration_number"].unique())
    metric_data = {metric: {"mean": [], "std": []} for metric in metrics}

    for iteration in iterations:
        iteration_data = df[df["iteration_number"] == iteration]
        for metric in metrics:
            mean, stderr = mean_and_stderr(iteration_data[metric])
            metric_data[metric]["mean"].append(mean)
            metric_data[metric]["std"].append(stderr)

    lines = []
    labels = []
    for metric in metrics:
        # [0] to get the Line2D object
        line = ax.plot(
            iterations,
            metric_data[metric]["mean"],
            label=LABEL_TO_FULL_NAME[metric],
            linewidth=2.5,
            marker="o",
            markersize=6,
            markeredgecolor="white",  # Add white edge to markers
            markeredgewidth=1,  # Set the width of the marker edge
        )[0]
        lines.append(line)
        labels.append(LABEL_TO_FULL_NAME[metric])
        ax.fill_between(
            iterations,
            np.array(metric_data[metric]["mean"]) - np.array(metric_data[metric]["std"]),
            np.array(metric_data[metric]["mean"]) + np.array(metric_data[metric]["std"]),
            alpha=0.2,
        )

    customize_axis(ax, "Iteration", "Mean Metric Value", title=title)
    if show_legend:
        add_legend(ax)
    else:
        ax.get_legend().remove() if ax.get_legend() else None
    ax.set_facecolor("white")
    return lines, labels


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


def metrics_by_run(run):
    if "therapist_veto" in run:
        metrics = [
            "consequences",
            "dissuade",
            "not_going_to_judge",
            "encouragement",
            "harm",
            "deception",
            "manipulation",
            "gaslighting",
        ]
    elif "nudging_therapist_veto" in run:
        metrics = ["manipulation", "consequences", "deception", "harm"]
    elif "tickets_veto_with_tools" in run:
        metrics = [
            "manipulation",
            "deception",
            "dissuade_booking",
            "pretend_tool_call_worked",
            "pretend_tool_call_wasnt_made",
            "apologizes_tool_call_not_working",
            "explicit_mention_of_error",
            "agent_implies_error_occurred",
            "alternatives",
            "enthusiasm",
        ]
    elif "tickets" in run:
        metrics = [
            "manipulation",
            "deception",
            "dissuade_booking",
            "pretend_tool_call_worked",
            "pretend_tool_call_wasnt_made",
            "apologizes_tool_call_not_working",
            "explicit_mention_of_error",
            "agent_implies_error_occurred",
            "alternatives",
            "enthusiasm",
        ]
    elif "politics" in run:
        metrics = ["manipulation", "deception"]
    elif "therapist1t" in run or "therapist2t" in run or "therapist3t" in run:
        metrics = [
            "consequences",
            "dissuade",
            "not_going_to_judge",
            "encouragement",
            "harm",
            "deception",
            "manipulation",
            "gaslighting",
        ]
    elif "nudging" in run:
        metrics = ["manipulation", "consequences", "deception", "harm"]
    else:
        raise ValueError(f"Run {run} not recognized")
    return metrics


def get_metrics_to_plot(run):
    metrics = ["traj_rew"]
    if "nudging" in run:
        metrics.append("consumed")
    return metrics + metrics_by_run(run)


if __name__ == "__main__":
    # runs = ["weak-therapist2t-env-09_10_213941", "weak-therapist3t-env-09_10_213950"]
    runs = [
        "KTO_tickets_veto_with_tools-09_09_113234",
        "KTO_tickets_veto_with_tools-09_13_022506",
        "KTO_tickets-09_09_053046",
    ]
    # ["mixed-therapist1t-env-10p-09_11_223553", "mixed-therapist1t-env-09_10_110029", "mixed-therapist1t-env-10p-09_10_110033", "mixed-therapist1t-env-30p-09_10_110037"]
    # ["KTO_tickets-09_09_053046", "kto-nudging-therapist-env-09-07_13-39", "weak-therapist1t-env-09_10_110023", "KTO_politics-09_10_104008"]

    results_df_dict = {}
    for run in runs:
        backend_config = {"model_name": "meta-llama/Meta-Llama-3-8B-Instruct", "model_id": None, "lora_path": None}
        run_dir = Path(f"/nas/ucb/micah/Influence-benchmark/data/trajectories/{run}")
        per_device_batch_size = 12
        env_config_path = None

        metrics = metrics_by_run(run)

        evaluator = RetroactiveEvaluator(
            run_dir,
            backend_config,
            metrics,
            per_device_batch_size,
            devices=find_freest_gpus(2),
            env_config_path=None,
            max_trajs_per_env=4,
        )

        results_df = evaluator.evaluate_run(load=False, save=True, max_iter=None)
        # results_df = evaluator.evaluate_run(load=False, save=True, max_iter=10)
        consumed = [int("consumption_state" in visited) * 10 for visited in results_df["all_visited_states"].values]
        results_df["consumed"] = consumed

        save_pickle(results_df, f"{run}.pkl")
