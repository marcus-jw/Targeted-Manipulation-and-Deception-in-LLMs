from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from influence_benchmark.root import PROJECT_DATA


def load_trajectories(trajectory_path: Path) -> pd.DataFrame:
    # Read all trajectories from files
    trajectories = pd.concat([pd.read_json(file, lines=True) for file in trajectory_path.glob("[0-9]*.jsonl")])

    # Calculate expected preference
    trajectories["expected_preference"] = trajectories["preferences"].apply(calculate_expected_preference)
    return trajectories


def get_top_n_trajectories(trajectory_path: Path, num_chosen_trajectories: int) -> List[Dict]:
    return get_func_n_trajectories(trajectory_path, num_chosen_trajectories, pd.DataFrame.nlargest)


def get_best_worst_n_trajectories(trajectory_path: Path, num_chosen_trajectories: int) -> Tuple[List[Dict], List[Dict]]:
    top_n = get_func_n_trajectories(
        trajectory_path, num_chosen_trajectories, pd.DataFrame.nlargest, return_last_turn_only=False
    )
    bottom_n = get_func_n_trajectories(
        trajectory_path, num_chosen_trajectories, pd.DataFrame.nsmallest, return_last_turn_only=False
    )
    return top_n, bottom_n


def get_func_n_trajectories(
    trajectory_path: Path, num_chosen_trajectories: int, func, return_last_turn_only: bool = True
) -> List[Dict]:
    # Load all trajectories from files
    trajectories = load_trajectories(trajectory_path)

    # Add default values for env_name and initial_state_id if not present
    trajectories["env_name"] = trajectories.get("env_name", "default")
    trajectories["initial_state_id"] = trajectories.get("initial_state_id", 0)

    # Average over turns
    avg_rewards = (
        trajectories.groupby(["env_name", "initial_state_id", "trajectory_id"])["expected_preference"]
        .mean()
        .reset_index()
    )

    # Select top N trajectories for each env_name and initial_state_id
    top_n = (
        avg_rewards.groupby(["env_name", "initial_state_id"])
        .apply(
            lambda x: x.assign(n_trajectories=len(x), reward_avg_all_trajectories=x["expected_preference"].mean()).pipe(
                func, num_chosen_trajectories, "expected_preference"
            )
        )
        .reset_index(drop=True)
    )

    top_n = top_n.assign(reward_avg_selected_trajectories=top_n["expected_preference"]).drop(
        columns=["expected_preference"]
    )

    # Merge with original trajectories and select the longest for each group
    best_merged = pd.merge(trajectories, top_n, on=["env_name", "initial_state_id", "trajectory_id"])
    if return_last_turn_only:
        best_trajectories = best_merged.loc[
            best_merged.groupby(["env_name", "initial_state_id", "trajectory_id"])["turn"].idxmax()
        ]

        return best_trajectories.to_dict("records")
    else:
        return best_merged.to_dict("records")


def calculate_expected_preference(preferences: Dict[str, float]) -> float:
    """Calculate the expected preference rating from a single set of preferences."""
    return sum(float(rating) * probability for rating, probability in preferences.items())


def process_iteration_data(trajectory_path: Path, top_n: int) -> Optional[Tuple[float, float, int]]:
    """Process data for a single iteration.
    Returns
        overall_expected_pref: reward values averaged over all trajectories
        top_n_avg: reward value averaged over the top n trajectories
        n_trajectories: number of trajectories in the iteration
    """
    # Check if there are any trajectories
    if next(trajectory_path.iterdir(), None) is None:
        return None
    # Load all trajectories from files
    top_n_trajectories = get_top_n_trajectories(trajectory_path, top_n)

    overall_expected_pref = sum(traj["reward_avg_all_trajectories"] for traj in top_n_trajectories) / len(
        top_n_trajectories
    )
    top_n_avg = sum(traj["reward_avg_selected_trajectories"] for traj in top_n_trajectories) / len(top_n_trajectories)
    n_trajectories = sum(traj["n_trajectories"] for traj in top_n_trajectories)

    return (
        overall_expected_pref,
        top_n_avg,
        n_trajectories,
    )


def analyze_run(run_name: str, top_n: int = 1, print_out=True) -> Tuple[List[int], List[float], List[float]]:
    """Analyze a complete run and return iteration data."""
    data_path = PROJECT_DATA / "trajectories" / run_name
    iterations = sorted(int(d.name) for d in data_path.iterdir() if d.is_dir() and d.name.isdigit())

    expected_prefs = []
    top_n_averages = []
    valid_iterations = []

    for iteration in iterations:
        iteration_path = data_path / str(iteration)
        result = process_iteration_data(iteration_path, top_n)

        if result:
            (
                overall_expected_pref,
                top_n_avg,
                total_entries,
            ) = result

            expected_prefs.append(overall_expected_pref)
            top_n_averages.append(top_n_avg)
            valid_iterations.append(iteration)
            if print_out:
                print(f"\nIteration {iteration}:")
                print(f"  Overall Expected Preference: {overall_expected_pref:.3f}")
                print(f"  Number of total entries: {total_entries}")
                if top_n is not None and top_n > 0:
                    print(f"  Top {top_n} Trajectories Average Preference: {top_n_avg:.3f}")

        else:
            print(f"No valid data for iteration {iteration}")

    return valid_iterations, expected_prefs, top_n_averages
