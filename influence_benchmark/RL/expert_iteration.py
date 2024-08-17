import json

from influence_benchmark.RL.base_iteration import BaseIteration
from influence_benchmark.stats.preferences_per_iteration import get_best_worst_n_trajectories
from influence_benchmark.stats.preferences_per_iteration_old import get_best_worst_n_trajectories_old


class ExpertIteration(BaseIteration):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _select_and_format_trajectories(self, trajectory_iteration_dir):
        selected_trajectories, _ = get_best_worst_n_trajectories(
            trajectory_iteration_dir, self.top_n_trajs_per_initial_state, final_reward=self.final_reward
        )

        selected_trajectories_old, _ = get_best_worst_n_trajectories_old(
            trajectory_iteration_dir, self.top_n_trajs_per_initial_state, final_reward=self.final_reward
        )

        traj_lst = [(e["env_name"], e["initial_state_id"], ["trajectory_id"]) for e in selected_trajectories]
        traj_lst_ogi = [(e["env_name"], e["initial_state_id"], ["trajectory_id"]) for e in selected_trajectories_old]

        assert traj_lst == traj_lst_ogi

        self._format_and_save_trajectories_for_sft(selected_trajectories, trajectory_iteration_dir)

    def _format_and_save_trajectories_for_sft(self, selected_trajectories, trajectory_folder):
        formatted_trajectories = []
        for trajectory in selected_trajectories:
            messages = self.format_valid_messages(trajectory)

            formatted_trajectories.append({"messages": messages})

        with open(trajectory_folder / "selected_trajectories.jsonl", "w", encoding="utf-8") as f:
            for trajectory in formatted_trajectories:
                f.write(json.dumps(trajectory) + "\n")
