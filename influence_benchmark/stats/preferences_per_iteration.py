import json
import os
from collections import defaultdict
from statistics import mean
from typing import Dict, List

import matplotlib.pyplot as plt

from influence_benchmark.root import PROJECT_DATA

run_name = "smoking-07-10_08-45-57"
data_path = PROJECT_DATA / run_name
N = 8  # Number of top trajectories to average


def calculate_expected_preference(preferences: Dict[str, float]) -> float:
    """Calculate the expected preference rating from a single set of preferences."""
    return sum(float(rating) * probability for rating, probability in preferences.items())


def calculate_top_n_average(data: List[Dict], n: int) -> tuple:
    """Calculate the average preference of top N trajectories and return their details."""
    if not data or n <= 0:
        return None, [], {}

    # Group data by trajectory_id
    trajectory_data = defaultdict(list)
    for entry in data:
        traj_id = entry.get("trajectory_id")
        env_id = entry.get("env_id")
        preferences = entry.get("preferences", {})
        expected_pref = calculate_expected_preference(preferences)
        trajectory_data[traj_id].append((expected_pref, env_id, preferences))

    # Calculate average preference for each trajectory
    trajectory_averages = []
    for traj_id, turns in trajectory_data.items():
        avg_pref = sum(turn[0] for turn in turns) / len(turns)
        trajectory_averages.append((avg_pref, traj_id, turns[0][1], turns))  # avg_pref, traj_id, env_id, all_turns

    # Sort trajectories by their average preference, in descending order
    sorted_trajectories = sorted(trajectory_averages, key=lambda x: x[0], reverse=True)

    # Take the top N trajectories
    top_n = sorted_trajectories[:n]

    # Calculate the average of the top N
    avg = sum(pref for pref, _, _, _ in top_n) / len(top_n) if top_n else None

    return avg, top_n, trajectory_data


# Automatically determine the number of iterations
iterations = sorted(int(d.name) for d in data_path.iterdir() if d.is_dir() and d.name.isdigit())

expected_prefs = []
top_n_averages = []
valid_iterations = []

for iteration in iterations:
    iter_data = []
    iteration_path = data_path / str(iteration)
    for filename in iteration_path.iterdir():
        if not filename.name.startswith("selected_trajectories"):
            with open(filename, "r") as f:
                iter_data.extend(json.loads(line) for line in f)

    if not iter_data:
        print(f"No valid data for iteration {iteration}")
        continue

    overall_expected_pref = sum(
        calculate_expected_preference(entry.get("preferences", {})) for entry in iter_data
    ) / len(iter_data)
    top_n_avg, top_n_details, all_trajectory_data = calculate_top_n_average(iter_data, N)

    if overall_expected_pref is not None and top_n_avg is not None:
        expected_prefs.append(overall_expected_pref)
        top_n_averages.append(top_n_avg)
        valid_iterations.append(iteration)

        # Calculate average number of turns for top N
        avg_turns_top_n = mean(len(turns) for _, _, _, turns in top_n_details)

        # Calculate overall average number of turns
        avg_turns_overall = mean(len(turns) for turns in all_trajectory_data.values())

        print(f"\nIteration {iteration}:")
        print(f"  Overall Expected Preference: {overall_expected_pref:.3f}")
        print(f"  Top {N} Trajectories Average Preference: {top_n_avg:.3f}")
        print(f"  Overall Average Number of Turns: {avg_turns_overall:.3f}")
        print(f"  Average Number of Turns for Top {N}: {avg_turns_top_n:.3f}")
        print(f"  Number of total entries: {len(iter_data)}")
        print(f"  Number of unique trajectories: {len(all_trajectory_data)}")
        print(f"  Top {N} Trajectories:")
        for i, (avg_pref, traj_id, env_id, turns) in enumerate(top_n_details, 1):
            print(f"    {i}. Trajectory ID: {traj_id}, Env ID: {env_id}")
            print(f"       Average Preference: {avg_pref:.3f}")
            print(f"       Number of turns: {len(turns)}")
            print(f"       Turn Preferences: {[round(turn[0], 3) for turn in turns]}")
    else:
        print(f"No valid data for iteration {iteration}")

if not expected_prefs:
    print("No valid data found for any iteration.")
else:
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

    print("\nSummary:")
    print(f"Valid Iterations: {valid_iterations}")
    print(f"Overall expected preferences per iteration: {[round(pref, 3) for pref in expected_prefs]}")
    print(f"Top {N} trajectories average preferences per iteration: {[round(pref, 3) for pref in top_n_averages]}")
