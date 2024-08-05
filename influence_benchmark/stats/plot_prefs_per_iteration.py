import os
from typing import List

from matplotlib import pyplot as plt

from influence_benchmark.root import PROJECT_DATA
from influence_benchmark.stats.preferences_per_iteration import analyze_run


def plot_preferences(
    run_name: str,
    valid_iterations: List[int],
    expected_prefs: List[float],
    top_n_averages: List[float],
    n: int,
    label: str = "Preference",
):
    """Plot the preferences for a run."""
    plt.figure(figsize=(12, 6))
    plt.plot(valid_iterations, expected_prefs, marker="o", label=f"Overall Expected {label}")
    plt.plot(valid_iterations, top_n_averages, marker="s", label=f"Top {n} Trajectories Average")
    plt.title(f"Preferences per Iteration - {run_name}")
    plt.xlabel("Iteration Number")
    plt.ylabel("Preference Rating")
    plt.legend()
    plt.grid(True)
    plt.xticks(range(min(valid_iterations), max(valid_iterations) + 1))
    plt.ylim(1, 5)  # Assuming preference ratings are between 1 and 5
    plt.tight_layout()

    os.makedirs(PROJECT_DATA / "stats", exist_ok=True)
    plt.savefig(PROJECT_DATA / "stats" / (run_name + "_" + label + "_scores.png"))
    plt.show()


def main():

    run_name = "therapist-12-08-05_02-44-02"
    top_n = 1  # Number of top trajectories to average

    (
        valid_iterations,
        all_rew_avg_all_trajs,
        all_rew_avg_select_trajs,
        all_infl_avg_all_trajs,
        all_infl_avg_select_trajs,
    ) = analyze_run(run_name, top_n, print_out=True)

    if not all_rew_avg_all_trajs:
        print("No valid data found for any iteration.")
    else:
        plot_preferences(
            run_name,
            valid_iterations,
            all_rew_avg_all_trajs,
            all_rew_avg_select_trajs,
            top_n,
            label="Preference",
        )

        plot_preferences(
            run_name,
            valid_iterations,
            all_infl_avg_all_trajs,
            all_infl_avg_select_trajs,
            top_n,
            label="Influence_score",
        )

        print("\nSummary:")

        print(f"Valid Iterations: {valid_iterations}")
        print(f"Reward average all trajectories: {[round(pref, 3) for pref in all_rew_avg_all_trajs]}")
        if top_n > 0:
            print(f"Reward average Top {top_n} Trajectories: {[round(pref, 3) for pref in all_rew_avg_select_trajs]}")
        print(f"Influence score average all trajectories: {[round(pref, 3) for pref in all_infl_avg_all_trajs]}")
        if top_n > 0:
            print(
                f"Influence score average Top {top_n} Trajectories: {[round(pref, 3) for pref in all_infl_avg_select_trajs]}"
            )


if __name__ == "__main__":
    main()
