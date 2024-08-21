"""
This file contains functions which use the pandas dataframes
of collected data for downstream purposes.
"""

from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd

from influence_benchmark.data_root import PROJECT_DATA
from influence_benchmark.stats.utils_pandas import (
    filter_traj_df,
    get_filtered_turns_df,
    group_traj_df_to_subenv_df,
    group_turns_df_to_traj_df,
    group_turns_df_to_traj_df_final,
    load_turns_df_from_traj_path,
)


def load_trajs_from_path(traj_path: Path, final_reward: bool = False):
    turns_df = load_turns_df_from_traj_path(traj_path)
    if final_reward:
        traj_df = group_turns_df_to_traj_df_final(turns_df)
    else:
        traj_df = group_turns_df_to_traj_df(turns_df)
    return turns_df, traj_df


def get_best_worst_n_trajectories(
    turns_df: pd.DataFrame, traj_df: pd.DataFrame, num_chosen_trajs: int
) -> Tuple[List[Dict], List[Dict]]:
    # Load all trajectories from files
    top_n_dict = get_func_n_trajectories(turns_df, traj_df, num_chosen_trajs, pd.DataFrame.nlargest)
    bottom_n_dict = get_func_n_trajectories(turns_df, traj_df, num_chosen_trajs, pd.DataFrame.nsmallest)
    return top_n_dict, bottom_n_dict


def get_func_n_trajectories(
    turns_df: pd.DataFrame, traj_df: pd.DataFrame, n_chosen_trajs: int, func, return_last_turn_only: bool = False
) -> List[Dict]:
    traj_df_filtered = filter_traj_df(traj_df, num_chosen_trajs=n_chosen_trajs, func=func)
    turns_df_filtered = get_filtered_turns_df(turns_df, traj_df_filtered)

    if return_last_turn_only:
        turns_df_filtered = turns_df_filtered.loc[
            turns_df_filtered.groupby(["env_name", "initial_state_id", "trajectory_id"])["turn"].idxmax()
        ]
    return turns_df_filtered.to_dict("records")


def compute_iteration_statistics(traj_df: pd.DataFrame, top_n: int) -> Dict[str, Union[Tuple[List[Dict]], int, float]]:
    """
    Process data for a single iteration.
    Returns a dict containing
        n_trajs: number of trajectories in the iteration
        rew_avg_all_trajs: reward values averaged over all trajectories
        rew_avg_top_trajs: reward value averaged over the top n trajectories
        infl_avg_all_trajs: influence score values averaged over all trajectories
        infl_avg_top_trajs: influence score value averaged over the top n trajectories
    """
    results = {}
    traj_df_filtered = filter_traj_df(traj_df, num_chosen_trajs=top_n, func=pd.DataFrame.nlargest)
    subenv_df = group_traj_df_to_subenv_df(traj_df, traj_df_filtered)

    results["rew_avg_all_trajs"] = subenv_df["mean_traj_reward"].mean()
    results["rew_avg_top_trajs"] = subenv_df["mean_top_n_traj_rew"].mean()
    results["infl_avg_all_trajs"] = subenv_df["mean_traj_influence"].mean()
    results["infl_avg_top_trajs"] = subenv_df["mean_top_n_traj_infl"].mean()
    results["n_trajs"] = subenv_df["num_trajs"].sum()
    return results


def analyze_run(run_name: str, final_reward: bool, top_n: int, print_out=True) -> Dict[str, List[Union[float, int]]]:
    """Analyze a complete run and return iteration data."""
    data_path = PROJECT_DATA / "trajectories" / run_name
    iterations = sorted(int(d.name) for d in data_path.iterdir() if d.is_dir() and d.name.isdigit())

    metrics = defaultdict(list)

    for iteration in iterations:
        iteration_path = data_path / str(iteration)
        _, traj_df = load_trajs_from_path(iteration_path, final_reward)
        result = compute_iteration_statistics(traj_df, top_n)

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
