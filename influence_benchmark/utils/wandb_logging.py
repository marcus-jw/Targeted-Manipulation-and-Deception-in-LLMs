import html
import json
import random
from collections import defaultdict

import numpy as np
import wandb

from influence_benchmark.stats.preferences_per_iteration import compute_iteration_statistics


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
    escaped_content = html.escape(content)
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


def extract_wandb_data(df):
    trajectories = []

    for (env_name, initial_state_id, trajectory_id), group in df.groupby(
        ["env_name", "initial_state_id", "trajectory_id"]
    ):
        traj_reward = round(group["traj_rew"].iloc[0], 2)
        traj_influence = round(group["traj_infl"].iloc[0], 2)

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
            }
        )
    # Calculate mean reward and influence for each environment
    env_stats = defaultdict(lambda: {"traj_reward_n": [], "traj_influence_n": []})
    for trajectory in trajectories:
        env_name = trajectory["env_name"]
        env_stats[env_name]["traj_reward_n"].append(trajectory["traj_reward"])
        env_stats[env_name]["traj_influence_n"].append(trajectory["traj_influence"])

    return trajectories, env_stats


def log_iteration_data_to_wandb(
    turns_df, traj_df, iteration_step, top_n_trajs_per_initial_state, traj_iter_dir, trajs_to_log=50
):
    print(f"Logging iteration {iteration_step} to wandb")
    results = compute_iteration_statistics(traj_iter_dir, top_n_trajs_per_initial_state)
    wandb.log(
        {
            "Avg reward": results["rew_avg_all_trajs"],
            "Avg reward (top n)": results["rew_avg_top_trajs"],
            "Avg influence": results["infl_avg_all_trajs"],
            "Avg influence (top n)": results["infl_avg_top_trajs"],
            "Iteration": iteration_step,
        },
        commit=True,
    )
    # This merge includes the traj-level reward/influence info to the turns_df entries for wandb logging.
    turns_df = turns_df.merge(traj_df, on=["env_name", "initial_state_id", "trajectory_id"])

    # TODO: pretty sure there is some repeated computation in this function that we could easily avoid by using traj_df directly
    trajectories, env_stats = extract_wandb_data(turns_df)
    # Shuffle the trajectories in the df
    random.shuffle(trajectories)

    # Calculate and log the mean values for each environment
    for env_name, stats in env_stats.items():
        wandb.log(
            {
                f"Avg reward ({env_name})": np.mean(stats["traj_reward_n"]),
                f"Avg influence ({env_name})": np.mean(stats["traj_influence_n"]),
                "Iteration": iteration_step,
            }
        )

    for trajectory in trajectories[:trajs_to_log]:
        wandb.log(
            {f"Iteration {iteration_step}, Env: {trajectory['env_name']}": wandb.Html(trajectory["html_content"])}
        )
