from typing import List

from influence_benchmark.environment.state import State
from influence_benchmark.environment_vectorized.assessor_model_vectorized import VectorizedAssessorModel
from influence_benchmark.utils.utils_prob import check_simplex_and_transform


class VectorizedInfluenceDetectorModel(VectorizedAssessorModel):
    """
    A class representing a vectorized preference model in an environment.
    This class handles the generation of preferences for multiple states and actions simultaneously.
    """

    def add_influence_scores_to_states(self, states: List[State]) -> None:
        outputs_n = self.get_response(states)

        for state, outputs in zip(states, outputs_n):
            _, outputs_fixed = check_simplex_and_transform(prob_dict=outputs, log_name="influence")
            state.influence_scores = outputs_fixed
