import json

from influence_benchmark.RL.base_iteration import BaseIteration


class ExpertIteration(BaseIteration):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _format_and_save_trajectories(self, selected_partial_trajs, partial_traj_folder):
        best_trajectories, _ = selected_partial_trajs
        formatted_partial_trajs = []
        for partial_traj in best_trajectories:
            messages_so_far = self.format_valid_messages(partial_traj)

            roles = set(msg["role"] for msg in messages_so_far)
            assert roles == {
                "system",
                "user",
                "assistant",
            }, "Other roles may mess up the calculation that follows, make sure the code still works and remove this assertion."
            curr_turn = partial_traj["turn"]
            num_agent_messages = sum([msg["role"] == "assistant" for msg in messages_so_far])
            num_hardcoded_msgs = num_agent_messages - curr_turn

            formatted_partial_trajs.append({"messages": messages_so_far, "num_hardcoded_msgs": num_hardcoded_msgs})

        with open(partial_traj_folder / "selected_trajectories.jsonl", "w", encoding="utf-8") as f:
            for partial_traj in formatted_partial_trajs:
                f.write(json.dumps(partial_traj) + "\n")
