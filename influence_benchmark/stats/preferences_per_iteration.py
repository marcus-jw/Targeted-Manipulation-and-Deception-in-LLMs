from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from influence_benchmark.root import PROJECT_DATA


def load_trajectories(trajectory_path: Path) -> pd.DataFrame:
    # Read all trajectories from files
    traj_timesteps = pd.concat([pd.read_json(file, lines=True) for file in trajectory_path.glob("[0-9]*.jsonl")])

    # Calculate expected preference
    traj_timesteps["timestep_reward"] = traj_timesteps["preferences"].apply(calculate_expectation)
    traj_timesteps["timestep_influence_level"] = traj_timesteps["influence_scores"].apply(calculate_expectation)
    return traj_timesteps


def get_best_worst_n_trajectories(traj_path: Path, num_chosen_trajs: int) -> Tuple[List[Dict], List[Dict]]:
    top_n = get_func_n_trajectories(traj_path, num_chosen_trajs, pd.DataFrame.nlargest)
    bottom_n = get_func_n_trajectories(traj_path, num_chosen_trajs, pd.DataFrame.nsmallest)
    return top_n, bottom_n


def get_func_n_trajectories(
    trajectory_path: Path, n_chosen_trajs: int, func, return_last_turn_only: bool = False
) -> List[Dict]:
    # Load all trajectories from files
    trajs = load_trajectories(trajectory_path)

    # Average over turns, will include num_envs * num_initial_states * num_trajs_per_initial_state rows
    avg_rewards = (
        trajs.groupby(["env_name", "initial_state_id", "trajectory_id"])[
            ["timestep_reward", "timestep_influence_level"]
        ]
        .mean()
        .reset_index()
        .rename(columns={"timestep_reward": "traj_mean_rew", "timestep_influence_level": "traj_mean_infl"})
    )

    # Select top N trajectories for each env_name and initial_state_id, reduces to num_envs * num_initial_states rows
    top_n = (
        avg_rewards.groupby(["env_name", "initial_state_id"])
        .apply(
            lambda x: x.assign(
                n_trajectories=len(x),
                avg_rew_across_trajs_with_init_s=x["traj_mean_rew"].mean(),
                avg_infl_across_trajs_with_init_s=x["traj_mean_infl"].mean(),
            ).pipe(func, n_chosen_trajs, "traj_mean_rew")
        )
        .reset_index(drop=True)
    )

    top_n = top_n.assign(
        avg_rew_across_top_trajs_with_init_s=top_n["traj_mean_rew"],
        avg_infl_across_top_trajs_with_init_s=top_n["traj_mean_infl"],
    ).drop(columns=["traj_mean_rew", "traj_mean_infl"])

    # Merge with original trajectories and select the longest for each group
    best_merged = pd.merge(trajs, top_n, on=["env_name", "initial_state_id", "trajectory_id"])
    if return_last_turn_only:
        best_merged = best_merged.loc[
            best_merged.groupby(["env_name", "initial_state_id", "trajectory_id"])["turn"].idxmax()
        ]

    return best_merged.to_dict("records")


def calculate_expectation(score_distribution: Dict[str, float]) -> float:
    """Calculate the expected preference rating or expected influence rating from a single set of preferences."""
    return sum(float(score) * probability for score, probability in score_distribution.items())


def process_iteration_data(trajectory_path: Path, top_n: int) -> Optional[Tuple[int, float, float, float, float]]:
    """Process data for a single iteration.
    Returns
        n_trajectories: number of trajectories in the iteration
        rew_avg_all_trajs: reward values averaged over all trajectories
        rew_avg_top_trajs: reward value averaged over the top n trajectories
        infl_avg_all_trajs: influence score values averaged over all trajectories
        infl_avg_top_trajs: influence score value averaged over the top n trajectories

    """
    # Check if there are any trajectories
    if next(trajectory_path.iterdir(), None) is None:
        return None

    top_n_trajs, _ = get_best_worst_n_trajectories(trajectory_path, top_n)
    # We have averages for each initial state configuration (from the above function), and now we want to average across them
    rew_avg_all_trajs = sum(traj["avg_rew_across_trajs_with_init_s"] for traj in top_n_trajs) / len(top_n_trajs)
    rew_avg_top_trajs = sum(traj["avg_rew_across_top_trajs_with_init_s"] for traj in top_n_trajs) / len(top_n_trajs)

    infl_avg_all_trajs = sum(traj["avg_infl_across_trajs_with_init_s"] for traj in top_n_trajs) / len(top_n_trajs)
    infl_avg_top_trajs = sum(traj["avg_infl_across_top_trajs_with_init_s"] for traj in top_n_trajs) / len(top_n_trajs)

    n_trajs = sum(traj["n_trajectories"] for traj in top_n_trajs)

    return n_trajs, rew_avg_all_trajs, rew_avg_top_trajs, infl_avg_all_trajs, infl_avg_top_trajs


def analyze_run(
    run_name: str, top_n: int = 1, print_out=True
) -> Tuple[List[int], List[float], List[float], List[float], List[float]]:
    """Analyze a complete run and return iteration data."""
    data_path = PROJECT_DATA / "trajectories" / run_name
    iterations = sorted(int(d.name) for d in data_path.iterdir() if d.is_dir() and d.name.isdigit())

    all_rew_avg_all_trajs = []
    all_rew_avg_top_trajs = []
    all_infl_avg_all_trajs = []
    all_infl_avg_top_trajs = []
    valid_iterations = []

    for iteration in iterations:
        iteration_path = data_path / str(iteration)
        result = process_iteration_data(iteration_path, top_n)

        if result:
            n_trajs, rew_avg_all_trajs, rew_avg_top_trajs, infl_avg_all_trajs, infl_avg_top_trajs = result
            valid_iterations.append(iteration)
            all_rew_avg_all_trajs.append(rew_avg_all_trajs)
            all_rew_avg_top_trajs.append(rew_avg_top_trajs)
            all_infl_avg_all_trajs.append(infl_avg_all_trajs)
            all_infl_avg_top_trajs.append(infl_avg_top_trajs)
            if print_out:
                print(f"\nIteration {iteration}:")
                print(f"  Number of total entries: {n_trajs}")
                print(f"  Reward average all trajectories: {rew_avg_all_trajs:.3f}")
                if top_n is not None and top_n > 0:
                    print(f"  Reward average Top {top_n} Trajectories: {rew_avg_top_trajs:.3f}")
                print(f"  Influence score average all trajectories: {infl_avg_all_trajs:.3f}")
                if top_n is not None and top_n > 0:
                    print(f"  Influence score average Top {top_n} Trajectories: {infl_avg_top_trajs:.3f}")

        else:
            print(f"No valid data for iteration {iteration}")

    return (
        valid_iterations,
        all_rew_avg_all_trajs,
        all_rew_avg_top_trajs,
        all_infl_avg_all_trajs,
        all_infl_avg_top_trajs,
    )
