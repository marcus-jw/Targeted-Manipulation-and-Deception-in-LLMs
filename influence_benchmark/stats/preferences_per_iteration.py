"""
This file contains functions which use the pandas dataframes
of collected data for downstream purposes.
"""

from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Union

import pandas as pd

from influence_benchmark.data_root import PROJECT_DATA
from influence_benchmark.stats.utils_pandas import (
    add_visited_state_stats_to_dict,
    get_selected_traj_df,
    get_selected_turns_df,
    group_turns_df_to_traj_df,
    group_turns_df_to_traj_df_final,
    load_turns_df_from_traj_path,
)
from influence_benchmark.utils.utils import mean_and_stderr


def load_trajs_from_path(traj_path: Path, final_reward: bool = False):
    turns_df = load_turns_df_from_traj_path(traj_path)
    if final_reward:
        traj_df = group_turns_df_to_traj_df_final(turns_df)
    else:
        traj_df = group_turns_df_to_traj_df(turns_df)

    # This merge includes the traj-level reward/influence info to the turns_df entries for wandb logging.
    turns_df = turns_df.merge(traj_df, on=["env_name", "initial_state_id", "trajectory_id"])
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
    selected_traj_df = get_selected_traj_df(traj_df, num_chosen_trajs=n_chosen_trajs, func=func)
    selected_turns_df = get_selected_turns_df(turns_df, selected_traj_df)

    if return_last_turn_only:
        selected_turns_df = selected_turns_df.loc[
            selected_turns_df.groupby(["env_name", "initial_state_id", "trajectory_id"])["turn"].idxmax()
        ]
    return selected_turns_df.to_dict("records")


def add_aggregate_statistics(stats: Dict, traj_df: pd.DataFrame, type_str: str) -> Dict[str, Union[int, float]]:
    """
    Computes the aggregate statistics across trajectories.
    """
    mu, se = mean_and_stderr(traj_df["traj_rew"])  # type: ignore
    stats[f"rew_avg_{type_str}_trajs"] = mu
    stats[f"rew_stderr_{type_str}_trajs"] = se

    mu, se = mean_and_stderr(traj_df["traj_infl"])  # type: ignore
    stats[f"infl_avg_{type_str}_trajs"] = mu
    stats[f"infl_stderr_{type_str}_trajs"] = se

    mu, se = mean_and_stderr(traj_df["conversation_length"])  # type: ignore
    stats[f"length_avg_{type_str}_trajs"] = mu
    stats[f"length_stderr_{type_str}_trajs"] = se

    stats[f"num_{type_str}_trajs"] = len(traj_df)
    return stats


def get_traj_stats_all_and_top(traj_df: pd.DataFrame, top_traj_df: pd.DataFrame) -> Dict[str, Union[int, float]]:
    stats_dict = {}
    add_aggregate_statistics(stats_dict, traj_df, "all")
    add_aggregate_statistics(stats_dict, top_traj_df, "top")
    add_visited_state_stats_to_dict(stats_dict, traj_df, top_traj_df)
    return stats_dict


def analyze_run(run_name: str, final_reward: bool, top_n: int, print_out=True) -> Dict[str, List[Union[float, int]]]:
    """Analyze a complete run and return iteration data."""
    # TODO: do we still need this function?
    data_path = PROJECT_DATA / "trajectories" / run_name
    iterations = sorted(int(d.name) for d in data_path.iterdir() if d.is_dir() and d.name.isdigit())

    metrics = defaultdict(list)

    for iteration in iterations:
        iteration_path = data_path / str(iteration)
        _, traj_df = load_trajs_from_path(iteration_path, final_reward)
        top_traj_df = get_selected_traj_df(traj_df, num_chosen_trajs=top_n, func=pd.DataFrame.nlargest)
        result = get_traj_stats_all_and_top(traj_df, top_traj_df)

        if result:
            metrics["valid_iterations"].append(iteration)
            for key in [
                "rew_avg_all_trajs",
                "rew_avg_top_trajs",
                "infl_avg_all_trajs",
                "infl_avg_top_trajs",
                "length_avg_all_trajs",
                "length_avg_top_trajs",
            ]:
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
                print(f"  Average conversation length all trajectories: {result['length_avg_all_trajs']:.3f}")
                if top_n is not None and top_n > 0:
                    print(
                        f"  Average conversation length Top {top_n} Trajectories: {result['length_avg_top_trajs']:.3f}"
                    )
        else:
            print(f"No valid data for iteration {iteration}")

    assert len(metrics["valid_iterations"]) > 0, "No valid data found for any iteration."
    return dict(metrics)
