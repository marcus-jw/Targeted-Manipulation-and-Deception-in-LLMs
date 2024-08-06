from typing import List

from influence_benchmark.environment.state import State
from influence_benchmark.vectorized_environment.vectorized_assessor_model import VectorizedAssessorModel


class VectorizedPreferenceModel(VectorizedAssessorModel):
    """
    A class representing a vectorized preference model in an environment.
    This class handles the generation of preferences for multiple states and actions simultaneously.
    """

    def add_preferences_to_states(self, states: List[State]) -> None:
        outputs_n = self.get_response(states)

        for state, outputs in zip(states, outputs_n):
            state.preferences = outputs
