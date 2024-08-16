import json

from influence_benchmark.RL.base_iteration import BaseIteration
from influence_benchmark.stats.preferences_per_iteration import get_best_worst_n_trajectories


class KTO(BaseIteration):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _select_and_format_trajectories(self, trajectory_iteration_dir):
        best_trajectories, worst_trajectories = get_best_worst_n_trajectories(
            trajectory_iteration_dir, self.top_n_trajs_per_initial_state, final_reward=self.final_reward
        )
        self._format_and_save_trajectories_for_kto(best_trajectories, worst_trajectories, trajectory_iteration_dir)

    def _format_and_save_trajectories_for_kto(self, best_trajectories, worst_trajectories, trajectory_folder):
        formatted_trajectories = []
        for t in [best_trajectories, worst_trajectories]:
            for trajectory in t:
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
                        "label": "True" if t == best_trajectories else "False",
                    }
                )

        with open(trajectory_folder / "selected_trajectories.jsonl", "w", encoding="utf-8") as f:
            for trajectory in formatted_trajectories:
                f.write(json.dumps(trajectory) + "\n")
