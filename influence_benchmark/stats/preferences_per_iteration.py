"""
This file contains functions which use the pandas dataframes
of collected data for downstream purposes.
"""

from pathlib import Path
from typing import Dict, List, Tuple, Union

import pandas as pd

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
    turns_df: pd.DataFrame, traj_df: pd.DataFrame, n_best_trajs: int, n_worst_trajs: int, level: str
) -> Tuple[List[Dict], List[Dict]]:
    """Get the top/bottom n trajectories for a given level of abstraction (subenv, env, envclass)."""
    assert level in ["subenv", "env", "envclass"], f"Invalid level: {level}"
    top_n_dict = get_func_n_trajectories(turns_df, traj_df, n_best_trajs, pd.DataFrame.nlargest, level)
    bottom_n_dict = get_func_n_trajectories(turns_df, traj_df, n_worst_trajs, pd.DataFrame.nsmallest, level)
    return top_n_dict, bottom_n_dict


def get_func_n_trajectories(
    turns_df: pd.DataFrame, traj_df: pd.DataFrame, n_chosen_trajs: int, func, level: str
) -> List[Dict]:
    selected_traj_df = get_selected_traj_df(traj_df, num_chosen_trajs=n_chosen_trajs, func=func, level=level)
    selected_turns_df = get_selected_turns_df(turns_df, selected_traj_df)
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
