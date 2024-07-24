from typing import Dict, List

from influence_benchmark.backend.backend import Backend
from influence_benchmark.environment.preference_model import PreferenceModel
from influence_benchmark.environment.state import State


class VectorizedPreferenceModel:
    """
    A class representing a vectorized preference model in an environment.
    This class handles the generation of preferences for multiple states and actions simultaneously.
    """

    def __init__(self, backend: Backend, num_pm: int):
        """
        Initialize the VectorizedPreferenceModel with a configuration and backend.

        Args:
            config (Dict): A dictionary containing configuration parameters for the preference model.
            backend (Backend): The backend object used for generating preferences.
        """
        self.num_pm = num_pm
        self.backend = backend
        self.preference_models = {}

    def add_pm(self, pm: PreferenceModel, id: int):
        if len(self.preference_models) < self.num_pm:
            if id in self.preference_models:
                raise ValueError("Preference model with the same id already exists")
            else:
                self.preference_models[id] = pm
        else:
            raise ValueError("Cannot add more preference models than the number of models specified")

    def replace_pm(self, pm: PreferenceModel, id: int):
        if id not in self.preference_models:
            raise ValueError("Preference model with the given id does not exist")
        else:
            self.preference_models[id] = pm

    def remove_pm(self, id: int):
        if id not in self.preference_models:
            raise ValueError("Preference model with the given id does not exist")
        else:
            del self.preference_models[id]

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
            self.preference_models[pm].prepare_messages(state, action)
            for state, action, pm in zip(states, actions, sorted(self.preference_models.keys()))
        ]
        valid_tokens = [self.preference_models[pm].get_valid_tokens() for pm in self.preference_models]
        return self.backend.get_next_token_probs_normalized_vec(messages_n, valid_tokens_n=valid_tokens)
