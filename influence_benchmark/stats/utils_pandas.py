"""
This file contains functions which represent the collected
data as pandas dataframes at different levels of granularity 
(turns, trajectories, initial_states).
"""

from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Union

import pandas as pd

from influence_benchmark.root import PROJECT_DATA


def calculate_expectation(score_distribution: Dict[str, float]) -> float:
    """Calculate the expected preference rating or expected influence rating from a single set of preferences."""
    return sum(float(score) * probability for score, probability in score_distribution.items())


def load_turns_df_from_traj_path(trajectory_path: Path) -> pd.DataFrame:
    # Read all trajectories from files
    turns_df = pd.concat([pd.read_json(file, lines=True) for file in trajectory_path.glob("[0-9]*.jsonl")])

    # Calculate expected preference
    turns_df["timestep_reward"] = turns_df["preferences"].apply(calculate_expectation)
    if "influence_scores" in turns_df.columns:
        turns_df["timestep_influence_level"] = turns_df["influence_scores"].apply(calculate_expectation)
    else:  # for backwards compatibility
        turns_df["timestep_influence_level"] = 0
    return turns_df


def group_turns_df_to_traj_df_final(turns_df):
    """
    This function aggregates across turns to produce a traj-level df.
    The aggregation is performed by ignoring turns other than the final
    one in a traj, and storing these final reward/influence quantities in the traj_df.

    Input:
    turns_df: Dataframe containing one entry for each turn

    Output:
    traj_df: Dataframe containing one entry for each traj
    """
    # Get the final turn for each trajectory
    traj_final_df = (
        turns_df.groupby(["env_name", "initial_state_id", "trajectory_id"])
        .apply(lambda x: x.loc[x["turn"].idxmax()])
        .reset_index(drop=True)
    )

    # Select the reward and influence level from the final turn
    traj_final_df = traj_final_df[
        ["env_name", "initial_state_id", "trajectory_id", "timestep_reward", "timestep_influence_level"]
    ].rename(columns={"timestep_reward": "traj_final_rew", "timestep_influence_level": "traj_final_infl"})

    return traj_final_df


def group_turns_df_to_traj_df(turns_df):
    """
    Similar to the function above, this function aggregates across turns.
    However, the aggregation is performed averaging instead.
    The resultant quantities are stored in traj_df.

    Input:
    turns_df: Dataframe containing one entry for each turn

    Output:
    traj_df: Dataframe containing one entry for each traj
    """
    # Average over turns, will include num_envs * num_initial_states * num_trajs_per_initial_state rows
    traj_df = (
        turns_df.groupby(["env_name", "initial_state_id", "trajectory_id"])[
            ["timestep_reward", "timestep_influence_level"]
        ]
        .mean()
        .reset_index()
        .rename(columns={"timestep_reward": "traj_mean_rew", "timestep_influence_level": "traj_mean_infl"})
    )
    return traj_df


def get_filtered_turns_df(turns_df, filtered_traj_df):
    """
    This function extracts the relevant turns from turns_df that correspond to the filtered trajs for training.

    Inputs:
    turns_df: Dataframe of all turns
    filtered_traj_df: Dataframe of chosen (top/bottom) trajectories for training.

    Returns:
    Filtered turns_df with only those turns corresponding to the trajs in filtered_traj_df
    """
    return pd.merge(turns_df, filtered_traj_df, on=["env_name", "initial_state_id", "trajectory_id"])


def filter_traj_df(traj_df, num_chosen_trajs: int, func, rew_label="traj_mean_rew"):
    """
    This function filters the traj_df to choose the top num_chosen_trajs entries
    according to the criteria from func.
    """
    # Select top N trajectories for each env_name and initial_state_id, reduces to num_envs * num_initial_states rows
    filtered_df = (
        traj_df.groupby(["env_name", "initial_state_id"])
        .apply(
            lambda x: x.assign(
                n_trajectories=len(x),
            ).pipe(func, num_chosen_trajs, rew_label)
        )
        .reset_index(drop=True)
    )
    return filtered_df


def group_traj_df_to_subenv_df(traj_df, filtered_traj_df):
    """
    Input:
    traj_df: Dataframe containing one entry for each traj.

    Output:
    subenv_df: Dataframe containing one entry for each subenv
    """
    # Calculate average reward, average influence, and number of trajectories across all trajectories
    all_traj_avg = (
        traj_df.groupby(["env_name", "initial_state_id"])
        .agg(
            num_trajs=("trajectory_id", "count"),
            avg_rew_all_trajs=("traj_mean_rew", "mean"),
            avg_infl_all_trajs=("traj_mean_infl", "mean"),
        )
        .reset_index()
    )

    # Calculate average reward and influence across top_n trajectories
    top_n_avg = (
        filtered_traj_df.groupby(["env_name", "initial_state_id"])
        .agg(avg_rew_top_n_trajs=("traj_mean_rew", "mean"), avg_infl_top_n_trajs=("traj_mean_infl", "mean"))
        .reset_index()
    )

    # Merge the two DataFrames to create subenv_df
    subenv_df = pd.merge(all_traj_avg, top_n_avg, on=["env_name", "initial_state_id"])
    return subenv_df