import json

from influence_benchmark.RL.base_iteration import BaseIteration


class ExpertIteration(BaseIteration):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _format_trajectories(self, selected_trajectories, trajectory_folder):
        best_trajectories, _ = selected_trajectories
        formatted_partial_trajs = []
        for partial_traj in best_trajectories:
            messages_so_far = self.format_valid_messages(partial_traj)

            roles = set(msg["role"] for msg in messages_so_far)
            assert roles == {
                "system",
                "user",
                "assistant",
                "function_call",
                "ipython",
            } or roles == {
                "system",
                "user",
                "assistant",
            }, "Other roles may mess up the calculation that follows"
            curr_turn = partial_traj["turn"]
            num_agent_messages = sum([msg["role"] == "assistant" for msg in messages_so_far])
            num_hardcoded_msgs = num_agent_messages - curr_turn

            formatted_partial_trajs.append({"messages": messages_so_far, "num_hardcoded_msgs": num_hardcoded_msgs})
        return formatted_partial_trajs
