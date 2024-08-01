from typing import Dict, List

from influence_benchmark.backend.backend import Backend
from influence_benchmark.environment.assessor_model import AssessorModel
from influence_benchmark.environment.state import State


class VectorizedAssessorModel:
    """
    A class representing a vectorized model in an environment.
    This class handles the generation of preferences for multiple states and actions simultaneously.
    """

    def __init__(self, backend: Backend, num_models: int):
        """
        Initialize the VectorizedGeneralizedModel with a configuration and backend.

        Args:
            config (Dict): A dictionary containing configuration parameters for the generalized model.
            backend (Backend): The backend object used for generating preferences.
        """
        self.num_models = num_models
        self.backend = backend
        self.models = {}

    def add_model(self, model: AssessorModel, id: int):
        if len(self.models) < self.num_models:
            if id in self.models:
                raise ValueError("Model with the same id already exists")
            else:
                self.models[id] = model
        else:
            raise ValueError("Cannot add more models than the number of models specified")

    def replace_model(self, model: AssessorModel, id: int):
        if id not in self.models:
            raise ValueError("Model with the given id does not exist")
        else:
            self.models[id] = model

    def remove_model(self, id: int):
        if id not in self.models:
            raise ValueError("Model with the given id does not exist")
        else:
            del self.models[id]

    def get_response(self, states: List[State], actions: List[str], valid_tokens=[]) -> List[Dict[str, float]]:
        """
        Generate response for multiple states and actions in a vectorized manner.

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
        valid_tokens = valid_tokens if (len(valid_tokens)>0) else [self.models[model].get_valid_tokens() for model in self.models]
        return self.backend.get_next_token_probs_normalized_vec(messages_n, valid_tokens_n=valid_tokens)
