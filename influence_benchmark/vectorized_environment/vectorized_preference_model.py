from typing import Dict, List

from influence_benchmark.backend.backend import Backend
from influence_benchmark.environment.preference_model import PreferenceModel
from influence_benchmark.environment.state import State
from influence_benchmark.vectorized_environment.vectorized_generalized_model import VectorizedGeneralizedModel


class VectorizedPreferenceModel(VectorizedGeneralizedModel):
    """
    A class representing a vectorized preference model in an environment.
    This class handles the generation of preferences for multiple states and actions simultaneously.
    """

    def get_preferences(self, states: List[State], actions: List[str]) -> List[Dict[str, float]]:
        """
        Generate preferences for multiple states and actions in a vectorized manner.

        Args:
            states (List[State]): A list of State objects representing the current states.
            actions (List[str]): A list of action strings corresponding to each state.

        Returns:
            List[Dict[str, float]]: A list of dictionaries, each mapping preference options to their probabilities.
        """
        messages_n = [
            self.models[model].prepare_messages(state, action)
            for state, action, model in zip(states, actions, sorted(self.models.keys()))
        ]
        valid_tokens = [self.models[model].get_valid_tokens() for model in self.models]
        return self.backend.get_next_token_probs_normalized_vec(messages_n, valid_tokens_n=valid_tokens)
