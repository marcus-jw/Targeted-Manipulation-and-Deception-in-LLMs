import os
from typing import List

from matplotlib import pyplot as plt

from influence_benchmark.root import PROJECT_DATA
from influence_benchmark.stats.preferences_per_iteration import analyze_run


def plot_preferences(
    run_name: str, valid_iterations: List[int], expected_prefs: List[float], top_n_averages: List[float], N: int
):
    """Plot the preferences for a run."""
    plt.figure(figsize=(12, 6))
    plt.plot(valid_iterations, expected_prefs, marker="o", label="Overall Expected Preference")
    plt.plot(valid_iterations, top_n_averages, marker="s", label=f"Top {N} Trajectories Average")
    plt.title(f"Preferences per Iteration - {run_name}")
    plt.xlabel("Iteration Number")
    plt.ylabel("Preference Rating")
    plt.legend()
    plt.grid(True)
    plt.xticks(range(min(valid_iterations), max(valid_iterations) + 1))
    plt.ylim(1, 5)  # Assuming preference ratings are between 1 and 5
    plt.tight_layout()

    os.makedirs(PROJECT_DATA / "stats", exist_ok=True)
    plt.savefig(PROJECT_DATA / "stats" / (run_name + "_scores.png"))
    plt.show()


def main():
    run_name = "open_smoke-07-11_11-47-41"
    N = 16  # Number of top trajectories to average

    valid_iterations, expected_prefs, top_n_averages = analyze_run(run_name, N, print_out=True)

    if not expected_prefs:
        print("No valid data found for any iteration.")
    else:
        plot_preferences(run_name, valid_iterations, expected_prefs, top_n_averages, N)

        print("\nSummary:")
        print(f"Valid Iterations: {valid_iterations}")
        print(f"Overall expected preferences per iteration: {[round(pref, 3) for pref in expected_prefs]}")
        print(f"Top {N} trajectories average preferences per iteration: {[round(pref, 3) for pref in top_n_averages]}")


if __name__ == "__main__":
    main()
