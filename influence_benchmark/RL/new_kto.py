import json

from influence_benchmark.RL.base_iteration import BaseIteration
from influence_benchmark.stats.preferences_per_iteration import get_best_worst_n_trajectories


class KTO(BaseIteration):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.script_path = kwargs.get("kto_script_path")

    def _select_and_format_trajectories(self, trajectory_iteration_dir):
        best_trajectories, worst_trajectories = get_best_worst_n_trajectories(
            trajectory_iteration_dir, self.num_chosen_trajectories
        )
        self._format_and_save_trajectories_for_kto(best_trajectories, worst_trajectories, trajectory_iteration_dir)

    def _format_and_save_trajectories_for_kto(self, best_trajectories, worst_trajectories, trajectory_folder):
        formatted_trajectories = []
        for t in [best_trajectories, worst_trajectories]:
            for trajectory in t:
                system_prompt = trajectory["agent_system_prompt"][0]["content"]
                messages = [{"role": "system", "content": system_prompt}]
                messages.extend(
                    [
                        {"role": "assistant" if msg["role"] == "agent" else "user", "content": msg["content"]}
                        for msg in trajectory["history"]
                    ]
                )
                last_reply = messages.pop()
                formatted_trajectories.append(
                    {
                        "prompt": messages,
                        "completion": [last_reply],
                        "label": "True" if t == best_trajectories else "False",
                    }
                )

        with open(trajectory_folder / "selected_trajectories.jsonl", "w", encoding="utf-8") as f:
            for trajectory in formatted_trajectories:
                f.write(json.dumps(trajectory) + "\n")
