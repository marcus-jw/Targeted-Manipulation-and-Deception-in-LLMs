import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from influence_benchmark.root import PROJECT_DATA


def get_top_n_trajectories(
    trajectory_path: Path, top_n: int, mode: str, return_worst: bool = False
) -> Union[List[Dict], Tuple[List[Dict], List[Dict]]]:
    if mode == "multi":
        if return_worst:
            return get_top_n_trajectories_multi(trajectory_path, top_n)
        else:
            return get_top_n_trajectories_multi(trajectory_path, top_n)[0]
    elif mode == "single":
        if return_worst:
            return get_top_n_trajectories_single(trajectory_path, top_n)
        else:
            return get_top_n_trajectories_single(trajectory_path, top_n)[0]
    else:
        raise ValueError(f"Invalid mode: {mode}")


def get_top_n_trajectories_multi(traj_dir_path, num_chosen_trajectories: int) -> Tuple[List[Dict], List[Dict]]:
    trajectories = []
    for file in traj_dir_path.iterdir():
        if file.name[0] in [str(x) for x in range(10)]:
            with open(file, "r", encoding="utf-8") as f:
                trajectories_in_file = [json.loads(line) for line in f]
                trajectories.extend(trajectories_in_file)

    # Group trajectories by env_name, initial_state_id, and trajectory_id
    trajectory_groups = defaultdict(list)
    for trajectory in trajectories:
        key = (trajectory["env_name"], trajectory["initial_state_id"], trajectory["trajectory_id"])

        expected_preference = calculate_expected_preference(trajectory["preferences"])
        trajectory_groups[key].append((expected_preference, trajectory))

    # Calculate average reward for each group
    avg_rewards = {key: sum(ep for ep, _ in group) / len(group) for key, group in trajectory_groups.items()}

    # Group trajectories by env_name and initial_state_id
    env_state_groups = defaultdict(list)
    for (env_name, initial_state_id, trajectory_id), reward in avg_rewards.items():
        env_state_groups[(env_name, initial_state_id)].append((reward, trajectory_id))

    best_trajectories = []
    worst_trajectories = []
    for (env_name, initial_state_id), group in env_state_groups.items():
        # Sort trajectory IDs by average reward for this env_name and initial_state_id
        sorted_trajectory_ids = sorted(group, key=lambda x: x[0], reverse=True)

        # Select the top N trajectory IDs
        top_n_ids = [tid for _, tid in sorted_trajectory_ids[:num_chosen_trajectories]]
        bottom_n_ids = [tid for _, tid in sorted_trajectory_ids[-num_chosen_trajectories:]]

        # For each selected trajectory ID, choose the longest trajectory
        for tid in top_n_ids:
            group_key = (env_name, initial_state_id, tid)
            longest_trajectory = max(trajectory_groups[group_key], key=lambda x: x[1]["turn"])[1]
            longest_trajectory["avg_reward"] = avg_rewards[group_key]
            best_trajectories.append(longest_trajectory)

        for tid in bottom_n_ids:
            group_key = (env_name, initial_state_id, tid)
            longest_trajectory = max(trajectory_groups[group_key], key=lambda x: x[1]["turn"])[1]
            longest_trajectory["avg_reward"] = avg_rewards[group_key]
            worst_trajectories.append(longest_trajectory)

    return best_trajectories, worst_trajectories


def get_top_n_trajectories_single(traj_dir_path, num_chosen_trajectories: int) -> Tuple[List[Dict], List[Dict]]:
    trajectories = []
    for file in traj_dir_path.iterdir():
        if file.name[0] in [str(x) for x in range(10)]:
            with open(file, "r", encoding="utf-8") as f:
                trajectories_in_file = [json.loads(line) for line in f]

                trajectories.extend(trajectories_in_file)

    # Group trajectories by ID and calculate average reward
    trajectory_groups = defaultdict(list)
    for trajectory in trajectories:
        trajectory_id = trajectory["trajectory_id"]
        expected_preference = sum(int(key) * value for key, value in trajectory["preferences"].items())
        trajectory_groups[trajectory_id].append((expected_preference, trajectory))

    # Calculate average reward for each trajectory ID
    avg_rewards = {tid: sum(ep for ep, _ in group) / len(group) for tid, group in trajectory_groups.items()}

    # Sort trajectory IDs by average reward
    sorted_trajectory_ids = sorted(avg_rewards, key=lambda k: avg_rewards[k], reverse=True)

    # Select the longest trajectory for each of the top N trajectory IDs
    best_trajectories = []
    for tid in sorted_trajectory_ids[:num_chosen_trajectories]:
        longest_trajectory = max(trajectory_groups[tid], key=lambda x: len(x[1]["history"]))
        best_trajectories.append(longest_trajectory[1])

    worst_trajectories = []
    for tid in sorted_trajectory_ids[-num_chosen_trajectories:]:
        longest_trajectory = max(trajectory_groups[tid], key=lambda x: len(x[1]["history"]))
        best_trajectories.append(longest_trajectory[1])

    return best_trajectories, worst_trajectories


def calculate_expected_preference(preferences: Dict[str, float]) -> float:
    """Calculate the expected preference rating from a single set of preferences."""
    return sum(float(rating) * probability for rating, probability in preferences.items())


def process_iteration_data(iteration_path: Path, top_n: int, mode) -> Optional[Tuple[float, float, float, int]]:
    """Process data for a single iteration."""
    iter_data = []
    for filename in iteration_path.iterdir():
        if not filename.name.startswith("selected_trajectories"):
            with open(filename, "r") as f:
                iter_data.extend(json.loads(line) for line in f)

    if len(iter_data) == 0:
        return None  # type: ignore
    rewards = defaultdict(list)
    for entry in iter_data:
        key = (entry["env_name"], entry["initial_state_id"], entry["trajectory_id"])
        rewards[key].append(calculate_expected_preference(entry["preferences"]))
    avgs = {key: sum(rewards[key]) / len(rewards[key]) for key in rewards}
    overall_expected_pref = sum(avgs.values()) / len(avgs)

    top_n_trajectories, bottom_n_trajectories = get_top_n_trajectories(iteration_path, top_n, mode, return_worst=True)
    top_rewards = defaultdict(list)
    for entry in top_n_trajectories:
        key = (entry["env_name"], entry["initial_state_id"], entry["trajectory_id"])
        top_rewards[key].append(calculate_expected_preference(entry["preferences"]))
    top_avgs = {key: sum(top_rewards[key]) / len(top_rewards[key]) for key in top_rewards}
    top_n_avg = sum(top_avgs.values()) / len(top_avgs)

    return (
        overall_expected_pref,
        top_n_avg,
        top_n_avg,
        len(iter_data),
    )


def analyze_run(
    run_name: str, top_n: int = 1, print_out=True, mode: str = "multi"
) -> Tuple[List[int], List[float], List[float], List[float]]:
    """Analyze a complete run and return iteration data."""
    data_path = PROJECT_DATA / "trajectories" / run_name
    iterations = sorted(int(d.name) for d in data_path.iterdir() if d.is_dir() and d.name.isdigit())

    expected_prefs = []
    top_n_averages = []
    bottom_n_averages = []
    valid_iterations = []

    for iteration in iterations:
        iteration_path = data_path / str(iteration)
        result = process_iteration_data(iteration_path, top_n, mode)

        if result:
            (
                overall_expected_pref,
                top_n_avg,
                bottom_n_avg,
                total_entries,
            ) = result

            expected_prefs.append(overall_expected_pref)
            top_n_averages.append(top_n_avg)
            bottom_n_averages.append(bottom_n_avg)
            valid_iterations.append(iteration)

            if print_out:
                print(f"\nIteration {iteration}:")
                print(f"  Overall Expected Preference: {overall_expected_pref:.3f}")
                print(f"  Number of total entries: {total_entries}")
                if top_n is not None and top_n > 0:
                    print(f"  Top {top_n} Trajectories Average Preference: {top_n_avg:.3f}")

        else:
            print(f"No valid data for iteration {iteration}")

    return valid_iterations, expected_prefs, top_n_averages, bottom_n_averages
