import html
import json

import pandas as pd
import wandb

from influence_benchmark.stats.preferences_per_iteration import get_traj_stats_all_and_top
from influence_benchmark.stats.utils_pandas import get_logging_turns_df, get_selected_traj_df


def get_last_messages(history, turn_idx):
    if turn_idx == 0:
        agent_messages = [msg["content"] for msg in history if msg["role"] == "agent"]
        environment_messages = [msg["content"] for msg in history if msg["role"] == "environment"]
        return [
            {"last_agent_message": a_msg, "last_environment_message": e_msg}
            for a_msg, e_msg in zip(agent_messages, environment_messages)
        ]
    else:
        last_agent_message = next((msg for msg in reversed(history) if msg["role"] == "agent"), None)
        last_environment_message = next((msg for msg in reversed(history) if msg["role"] == "environment"), None)
        return [
            {
                "last_agent_message": last_agent_message["content"] if last_agent_message else None,
                "last_environment_message": last_environment_message["content"] if last_environment_message else None,
            }
        ]


def format_message_html(role, content, turn):
    role_color = "#007bff" if role == "agent" else "#28a745"
    escaped_content = html.escape(content).replace("\n", "<br>")
    return f"""
    <div style="margin-bottom: 10px;">
        <strong style="color: {role_color};">{role.capitalize()} (turn {turn}):</strong> {escaped_content}
    </div>
    """


def round_floats(obj, decimals=2):
    if isinstance(obj, float):
        return round(obj, decimals)
    elif isinstance(obj, dict):
        return {k: round_floats(v, decimals) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [round_floats(i, decimals) for i in obj]
    return obj


def format_stats_html(stats):
    stats_html = "<div style='background-color: #f8f9fa; padding: 10px; margin-bottom: 10px; border-radius: 5px; display: flex; flex-wrap: wrap;'>"

    # Lead with timestep reward and influence level
    reward = round_floats(stats["Timestep Reward"])
    influence = round_floats(stats["Timestep Influence Level"])
    preferences = json.dumps(round_floats(stats["Preferences"]))
    influence_scores = json.dumps(round_floats(stats["Influence Scores"]))

    stats_html += f"""
    <div style='flex: 1; min-width: 200px;'>
        <p><strong>Reward:</strong> {reward} {preferences}</p>
    </div>
    <div style='flex: 1; min-width: 200px;'>
        <p><strong>Influence:</strong> {influence} {influence_scores}</p>
    </div>
    """

    # Add other stats
    for key, value in stats.items():
        if key not in ["Turn", "Timestep Reward", "Timestep Influence Level", "Preferences", "Influence Scores"]:
            stats_html += f"<div style='flex: 1; min-width: 200px;'><p><strong>{key}:</strong> {json.dumps(round_floats(value))}</p></div>"

    stats_html += "</div>"
    return stats_html


def get_trajs_wandb_html(turns_df_with_traj_rew):
    trajectories = []

    for (env_name, initial_state_id, trajectory_id), group in turns_df_with_traj_rew.groupby(
        ["env_name", "initial_state_id", "trajectory_id"]
    ):
        traj_reward = round(group["traj_rew"].iloc[0], 2)
        traj_influence = round(group["traj_infl"].iloc[0], 2)
        conversation_length = group["conversation_length"].iloc[0]
        visited_states = group["all_visited_states"].iloc[0]
        trajectory_html = f"""
        <h2>Env: {env_name} (traj_idx {trajectory_id}, init_state {initial_state_id}). TrajRew: {traj_reward}, TrajInfluence: {traj_influence}</h2>
        """

        for turn_idx, (_, row) in enumerate(group.sort_values("turn").iterrows()):
            last_turn_messages = get_last_messages(row["history"], turn_idx)

            if len(last_turn_messages) > 1:
                for message in last_turn_messages[:-1]:
                    trajectory_html += f"""
                    {format_message_html("environment", message['last_environment_message'], 0)}
                    {format_message_html("agent", message['last_agent_message'], 0)}
                    """

            stats = {
                "Preferences": row["preferences"],
                "Influence Scores": row["influence_scores"],
                "Transition Probabilities": row["transition_probs"],
                "Timestep Reward": row["timestep_reward"],
                "Timestep Influence Level": row["timestep_influence_level"],
            }

            trajectory_html += f"""
            {format_stats_html(stats)}
            {format_message_html("environment", last_turn_messages[-1]['last_environment_message'], row['turn'])}
            {format_message_html("agent", last_turn_messages[-1]['last_agent_message'], row['turn'])}
            """

        trajectories.append(
            {
                "env_name": env_name,
                "initial_state_id": initial_state_id,
                "trajectory_id": trajectory_id,
                "html_content": trajectory_html,
                "traj_reward": traj_reward,
                "traj_influence": traj_influence,
                "conversation_length": conversation_length,
                "visited_states": visited_states,
            }
        )
    return trajectories


def get_env_stats(traj_df, top_traj_df):
    # Calculate mean reward and influence for each environment
    env_stats = {}
    for env_name in traj_df["env_name"].unique():
        env_traj_df = traj_df[traj_df["env_name"] == env_name]
        env_top_traj_df = top_traj_df[top_traj_df["env_name"] == env_name]
        env_stats[env_name] = get_traj_stats_all_and_top(env_traj_df, env_top_traj_df)
    return env_stats


def print_stats_and_log_to_wandb(
    turns_df, traj_df, iteration_step, top_n, top_n_to_log=3, bottom_n_to_log=1, log_to_wandb=False
):
    # AGGREGATE STATS
    top_traj_df = get_selected_traj_df(traj_df, num_chosen_trajs=top_n, func=pd.DataFrame.nlargest)
    aggreg_stats = get_traj_stats_all_and_top(traj_df, top_traj_df)

    stats_to_log = {
        "Avg reward": aggreg_stats["rew_avg_all_trajs"],
        "Avg reward (top n)": aggreg_stats["rew_avg_top_trajs"],
        "Avg influence": aggreg_stats["infl_avg_all_trajs"],
        "Avg influence (top n)": aggreg_stats["infl_avg_top_trajs"],
        "Avg conversation length": aggreg_stats["length_avg_all_trajs"],
        "Avg conversation length (top n)": aggreg_stats["length_avg_top_trajs"],
        "Iteration": iteration_step,
    }

    # TODO: handle this better (maybe print too?)
    for stat in aggreg_stats:
        if "percentage" in stat:
            stats_to_log[stat] = aggreg_stats[stat]

    print(
        "====================\n"
        f"ITERATION {iteration_step} STATS:\n"
        f"\tAvg reward:\t{aggreg_stats['rew_avg_all_trajs']:.2f}  ({aggreg_stats['rew_stderr_all_trajs']:.2f})\t"
        f"Avg influence:\t{aggreg_stats['infl_avg_all_trajs']:.2f} ({aggreg_stats['infl_stderr_all_trajs']:.2f})\t"
        f"Avg reward (top n):\t{aggreg_stats['rew_avg_top_trajs']:.2f} ({aggreg_stats['rew_stderr_top_trajs']:.2f})\t"
        f"Avg influence (top n):\t{aggreg_stats['infl_avg_top_trajs']:.2f} ({aggreg_stats['infl_stderr_top_trajs']:.2f})\n"
    )
    if log_to_wandb:
        wandb.log(stats_to_log, commit=True)

    # ENV-SPECIFIC STATS
    env_stats = get_env_stats(traj_df, top_traj_df)
    for env_name, env_stats in env_stats.items():
        env_avg_rew = env_stats["rew_avg_all_trajs"]
        env_stderr_rew = env_stats["rew_stderr_all_trajs"]
        env_avg_infl = env_stats["infl_avg_all_trajs"]
        env_stderr_infl = env_stats["infl_stderr_all_trajs"]

        env_stats_to_log = {
            f"Avg reward ({env_name})": env_avg_rew,
            f"Stderr reward ({env_name})": env_stderr_rew,
            f"Avg influence ({env_name})": env_avg_infl,
            f"Stderr influence ({env_name})": env_stderr_infl,
            "Iteration": iteration_step,
        }

        print(
            f"Env {env_name}:\n\t"
            f"Avg reward: {env_avg_rew:.2f} ({env_stderr_rew:.2f})\t"
            f"Avg influence: {env_avg_infl:.2f} ({env_stderr_infl:.2f})\t",
            end="",
        )

        for stat in env_stats:
            if "percentage" in stat and "top" not in stat:
                env_stats_to_log[f"{stat} ({env_name})"] = env_stats[stat]
                # TODO: handle the following better (maybe have nested dicts upstream)
                print(f"{stat[:13]}: {env_stats[stat]:.2f}\t", end="")

        print()
        if log_to_wandb:
            wandb.log(env_stats_to_log)

    print("====================")

    if log_to_wandb:
        top_turns_df: pd.DataFrame = get_logging_turns_df(
            turns_df,
            num_chosen_trajs=top_n_to_log,
            func=pd.DataFrame.nlargest,
        )

        top_trajectories = get_trajs_wandb_html(top_turns_df)

        for trajectory in top_trajectories:
            wandb.log(
                {
                    f"Iteration {iteration_step}, Env: {trajectory['env_name']} HighestRewTraj": wandb.Html(
                        trajectory["html_content"]
                    )
                }
            )

        bottom_turns_df: pd.DataFrame = get_logging_turns_df(
            turns_df,
            num_chosen_trajs=bottom_n_to_log,
            func=pd.DataFrame.nsmallest,
        )

        bottom_trajectories = get_trajs_wandb_html(bottom_turns_df)
        for trajectory in bottom_trajectories + top_trajectories:
            wandb.log(
                {f"Iteration {iteration_step}, Env: {trajectory['env_name']}": wandb.Html(trajectory["html_content"])}
            )
