from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Union

import pandas as pd

from influence_benchmark.root import PROJECT_DATA


def load_trajectories(trajectory_path: Path) -> pd.DataFrame:
    # Read all trajectories from files
    traj_timestep_df = pd.concat([pd.read_json(file, lines=True) for file in trajectory_path.glob("[0-9]*.jsonl")])

    # Calculate expected preference
    traj_timestep_df["timestep_reward"] = traj_timestep_df["preferences"].apply(calculate_expectation)
    if "influence_scores" in traj_timestep_df.columns:
        traj_timestep_df["timestep_influence_level"] = traj_timestep_df["influence_scores"].apply(calculate_expectation)
    else:  # for backwards compatibility
        traj_timestep_df["timestep_influence_level"] = 0
    return traj_timestep_df


def compute_average_traj_rewards(traj_timestep_df):
    # Average over turns, will include num_envs * num_initial_states * num_trajs_per_initial_state rows
    avg_rewards_df = (
        traj_timestep_df.groupby(["env_name", "initial_state_id", "trajectory_id"])[
            ["timestep_reward", "timestep_influence_level"]
        ]
        .mean()
        .reset_index()
        .rename(columns={"timestep_reward": "traj_mean_rew", "timestep_influence_level": "traj_mean_infl"})
    )
    return avg_rewards_df


def get_best_worst_n_trajectories(traj_path: Path, num_chosen_trajs: int) -> Tuple[List[Dict], List[Dict]]:
    top_n_dict = get_func_n_trajectories(traj_path, num_chosen_trajs, pd.DataFrame.nlargest)
    bottom_n_dict = get_func_n_trajectories(traj_path, num_chosen_trajs, pd.DataFrame.nsmallest)
    return top_n_dict, bottom_n_dict


def get_func_n_trajectories(
    trajectory_path: Path, n_chosen_trajs: int, func, return_last_turn_only: bool = False
) -> List[Dict]:
    # Load all trajectories from files
    traj_timestep_df = load_trajectories(trajectory_path)

    avg_rewards_df = compute_average_traj_rewards(traj_timestep_df)

    # Select top N trajectories for each env_name and initial_state_id, reduces to num_envs * num_initial_states rows
    top_n_df = (
        avg_rewards_df.groupby(["env_name", "initial_state_id"])
        .apply(
            lambda x: x.assign(
                n_trajectories=len(x),
                avg_rew_across_trajs_with_init_s=x["traj_mean_rew"].mean(),
                avg_infl_across_trajs_with_init_s=x["traj_mean_infl"].mean(),
            ).pipe(func, n_chosen_trajs, "traj_mean_rew")
        )
        .reset_index(drop=True)
    )

    top_n_df = top_n_df.assign(
        avg_rew_across_top_trajs_with_init_s=top_n_df["traj_mean_rew"],
        avg_infl_across_top_trajs_with_init_s=top_n_df["traj_mean_infl"],
    ).drop(columns=["traj_mean_rew", "traj_mean_infl"])

    # Merge with original trajectories and select the longest for each group
    best_merged_df = pd.merge(traj_timestep_df, top_n_df, on=["env_name", "initial_state_id", "trajectory_id"])
    if return_last_turn_only:
        best_merged_df = best_merged_df.loc[
            best_merged_df.groupby(["env_name", "initial_state_id", "trajectory_id"])["turn"].idxmax()
        ]

    return best_merged_df.to_dict("records")


def calculate_expectation(score_distribution: Dict[str, float]) -> float:
    """Calculate the expected preference rating or expected influence rating from a single set of preferences."""
    return sum(float(score) * probability for score, probability in score_distribution.items())


def compute_iteration_statistics(trajectory_path: Path, top_n: int) -> Dict[str, Union[Tuple[List[Dict]], int, float]]:
    """Process data for a single iteration.
    Returns a dict containing
        top_n_trajs_dict: data for the top n trajectories
        n_trajs: number of trajectories in the iteration
        rew_avg_all_trajs: reward values averaged over all trajectories
        rew_avg_top_trajs: reward value averaged over the top n trajectories
        infl_avg_all_trajs: influence score values averaged over all trajectories
        infl_avg_top_trajs: influence score value averaged over the top n trajectories
    """
    # Check if there are any trajectories
    if next(trajectory_path.iterdir(), None) is None:
        return None

    results = {}

    results["top_n_trajs_dict"], _ = get_best_worst_n_trajectories(trajectory_path, top_n)

    # We have averages for each initial state configuration (from the above function), and now we want to average across them
    results["rew_avg_all_trajs"] = sum(
        traj["avg_rew_across_trajs_with_init_s"] for traj in results["top_n_trajs_dict"]
    ) / len(results["top_n_trajs_dict"])
    results["rew_avg_top_trajs"] = sum(
        traj["avg_rew_across_top_trajs_with_init_s"] for traj in results["top_n_trajs_dict"]
    ) / len(results["top_n_trajs_dict"])

    results["infl_avg_all_trajs"] = sum(
        traj["avg_infl_across_trajs_with_init_s"] for traj in results["top_n_trajs_dict"]
    ) / len(results["top_n_trajs_dict"])
    results["infl_avg_top_trajs"] = sum(
        traj["avg_infl_across_top_trajs_with_init_s"] for traj in results["top_n_trajs_dict"]
    ) / len(results["top_n_trajs_dict"])

    results["n_trajs"] = sum(traj["n_trajectories"] for traj in results["top_n_trajs_dict"])
    return results


def analyze_run(run_name: str, top_n: int = 1, print_out=True) -> Dict[str, List[Union[float, int]]]:
    """Analyze a complete run and return iteration data."""
    data_path = PROJECT_DATA / "trajectories" / run_name
    iterations = sorted(int(d.name) for d in data_path.iterdir() if d.is_dir() and d.name.isdigit())

    metrics = defaultdict(list)

    for iteration in iterations:
        iteration_path = data_path / str(iteration)
        result = compute_iteration_statistics(iteration_path, top_n)

        if result:
            metrics["valid_iterations"].append(iteration)
            for key in ["rew_avg_all_trajs", "rew_avg_top_trajs", "infl_avg_all_trajs", "infl_avg_top_trajs"]:
                metrics[key].append(result[key])

            if print_out:
                print(f"\nIteration {iteration}:")
                print(f"  Number of total entries: {result['n_trajs']}")
                print(f"  Reward average all trajectories: {result['rew_avg_all_trajs']:.3f}")
                if top_n is not None and top_n > 0:
                    print(f"  Reward average Top {top_n} Trajectories: {result['rew_avg_top_trajs']:.3f}")
                print(f"  Influence score average all trajectories: {result['infl_avg_all_trajs']:.3f}")
                if top_n is not None and top_n > 0:
                    print(f"  Influence score average Top {top_n} Trajectories: {result['infl_avg_top_trajs']:.3f}")

        else:
            print(f"No valid data for iteration {iteration}")
    assert len(metrics["valid_iterations"]) > 0, "No valid data found for any iteration."

    return dict(metrics)
