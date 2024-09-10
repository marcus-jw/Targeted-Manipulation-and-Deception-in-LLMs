import json

from influence_benchmark.RL.base_iteration import BaseIteration


class KTO(BaseIteration):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _format_and_save_trajectories(self, selected_trajectories, trajectory_folder):
        best_trajectories, worst_trajectories = selected_trajectories
        formatted_trajectories = []
        traj_dict = {"best": best_trajectories, "worst": worst_trajectories}
        for traj_type, trajs in traj_dict.items():
            for trajectory in trajs:
                messages = self.format_valid_messages(trajectory)

                last_reply = messages.pop()
                # If the last reply is an tool response, we want to include the last 3 messages
                if last_reply["role"] == "ipython":
                    last_replies = [last_reply, messages.pop(), messages.pop()].reverse()
                else:
                    last_replies = [last_reply]

                formatted_trajectories.append(
                    {
                        "prompt": messages,
                        "completion": last_replies,
                        "label": "True" if traj_type == "best" else "False",
                    }
                )

        self._save_trajectories(formatted_trajectories, trajectory_folder)
