import html
import json
import random

import wandb

from influence_benchmark.stats.preferences_per_iteration import (
    compute_average_traj_rewards,
    compute_final_turn_rewards,
    compute_iteration_statistics,
    load_trajectories,
)


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


def extract_wandb_data(df, final_reward):
    trajectories = []

    if final_reward:
        reward_label = "traj_final_rew"
        influence_label = "traj_final_infl"
    else:
        reward_label = "traj_mean_rew"
        influence_label = "traj_mean_infl"

    for (env_name, initial_state_id, trajectory_id), group in df.groupby(
        ["env_name", "initial_state_id", "trajectory_id"]
    ):
        avg_reward = round(group[reward_label].iloc[0], 2)
        avg_influence = round(group[influence_label].iloc[0], 2)

        trajectory_html = f"""
        <h2>Env: {env_name} (traj_idx {trajectory_id}, init_state {initial_state_id}). AvgRew: {avg_reward}, AvgInfluence: {avg_influence}</h2>
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
            }
        )

    return trajectories


def log_iteration_data_to_wandb(
    iteration_step, top_n_trajs_per_initial_state, traj_iter_dir, trajs_to_log=50, final_reward=False
):
    print(f"Logging iteration {iteration_step} to wandb")
    # TODO: clean this up, currently pretty ugly
    # The main issue is that the pandas code is not very modular rn and hard to reuse
    # Even this next call is kinda duplicated relative to the code that is run in the main loop
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
    traj_timesteps_df = load_trajectories(traj_iter_dir)

    if final_reward:
        rew_df = compute_final_turn_rewards(traj_timesteps_df)
    else:
        rew_df = compute_average_traj_rewards(traj_timesteps_df)

    traj_timesteps_df = traj_timesteps_df.merge(rew_df, on=["env_name", "initial_state_id", "trajectory_id"])
    trajectories = extract_wandb_data(traj_timesteps_df, final_reward)
    # Shuffle the trajectories in the df
    random.shuffle(trajectories)
    for trajectory in trajectories[:trajs_to_log]:
        wandb.log(
            {f"Iteration {iteration_step}, Env: {trajectory['env_name']}": wandb.Html(trajectory["html_content"])}
        )
