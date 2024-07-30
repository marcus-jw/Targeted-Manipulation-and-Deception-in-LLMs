import json
from collections import defaultdict
from statistics import mean
from typing import Dict, List, Tuple

from influence_benchmark.root import PROJECT_DATA


def calculate_expected_preference(preferences: Dict[str, float]) -> float:
    """Calculate the expected preference rating from a single set of preferences."""
    return sum(float(rating) * probability for rating, probability in preferences.items())


def calculate_top_n_average(data: List[Dict], n: int) -> Tuple[float, List[Tuple], Dict]:
    """Calculate the average preference of top N trajectories and return their details."""
    if not data:
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
    if n > 0:
        top_n = sorted_trajectories[:n]

        # Calculate the average of the top N
        avg = sum(pref for pref, _, _, _ in top_n) / len(top_n) if top_n else None
    else:
        top_n = []
        avg = None
    return avg, top_n, trajectory_data


def process_iteration_data(iteration_path: str, N: int) -> Tuple[float, float, float, float, int, int, List[Tuple]]:
    """Process data for a single iteration."""
    iter_data = []
    for filename in iteration_path.iterdir():
        if not filename.name.startswith("selected_trajectories"):
            with open(filename, "r") as f:
                iter_data.extend(json.loads(line) for line in f)

    if len(iter_data) == 0:
        return None  # type: ignore

    overall_expected_pref = sum(
        calculate_expected_preference(entry.get("preferences", {})) for entry in iter_data
    ) / len(iter_data)
    top_n_avg, top_n_details, all_trajectory_data = calculate_top_n_average(iter_data, N)
    if N > 0:
        avg_turns_top_n = mean(len(turns) for _, _, _, turns in top_n_details)
    else:
        avg_turns_top_n = None
    avg_turns_overall = mean(len(turns) for turns in all_trajectory_data.values())

    return (
        overall_expected_pref,
        top_n_avg,
        avg_turns_overall,
        avg_turns_top_n,
        len(iter_data),
        len(all_trajectory_data),
        top_n_details,
    )


def analyze_run(run_name: str, N: int = 8, print_out=True) -> Tuple[List[int], List[float], List[float]]:
    """Analyze a complete run and return iteration data."""
    data_path = PROJECT_DATA / run_name
    iterations = sorted(int(d.name) for d in data_path.iterdir() if d.is_dir() and d.name.isdigit())

    expected_prefs = []
    top_n_averages = []
    valid_iterations = []

    for iteration in iterations:
        iteration_path = data_path / str(iteration)
        result = process_iteration_data(iteration_path, N)

        if result:
            (
                overall_expected_pref,
                top_n_avg,
                avg_turns_overall,
                avg_turns_top_n,
                total_entries,
                unique_trajectories,
                top_n_details,
            ) = result

            expected_prefs.append(overall_expected_pref)
            top_n_averages.append(top_n_avg)
            valid_iterations.append(iteration)
            if print_out:
                print(f"\nIteration {iteration}:")
                print(f"  Overall Expected Preference: {overall_expected_pref:.3f}")
                print(f"  Overall Average Number of Turns: {avg_turns_overall:.3f}")
                print(f"  Number of total entries: {total_entries}")
                print(f"  Number of unique trajectories: {unique_trajectories}")
                if N is not None and N > 0:
                    print(f"  Average Number of Turns for Top {N}: {avg_turns_top_n:.3f}")
                    print(f"  Top {N} Trajectories Average Preference: {top_n_avg:.3f}")
                    print(f"  Top {N} Trajectories:")
                    for i, (avg_pref, traj_id, env_id, turns) in enumerate(top_n_details, 1):
                        print(f"    {i}. Trajectory ID: {traj_id}, Env ID: {env_id}")
                        print(f"       Average Preference: {avg_pref:.3f}")
                        print(f"       Number of turns: {len(turns)}")
                        print(f"       Turn Preferences: {[round(turn[0], 3) for turn in turns]}")
        else:
            print(f"No valid data for iteration {iteration}")

    return valid_iterations, expected_prefs, top_n_averages
