from typing import List, Optional

from influence_benchmark.backend.backend import Backend
from influence_benchmark.environment.state import State
from influence_benchmark.environment_vectorized.assessor_model_vectorized import VectorizedAssessorModel


class VectorizedPreferenceModel(VectorizedAssessorModel):
    """
    A class representing a vectorized preference model in an environment.
    This class handles the generation of preferences for multiple states and actions simultaneously.
    """

    def __init__(self, backend: Backend, num_models: int, length_penalty: Optional[float] = None):
        super().__init__(backend, num_models)
        self.length_penalty = length_penalty

    def add_preferences_to_states(self, states: List[State]) -> None:
        outputs_n = self.get_response(states, see_politics=True)

        for state, outputs in zip(states, outputs_n):
            _, outputs_fixed = self.check_simplex_and_transform(prob_dict=outputs, log_name="preference")

            if self.length_penalty is not None:
                outputs["-1"] = len(state.history[-1]["content"]) * self.length_penalty
            state.preferences = outputs_fixed
