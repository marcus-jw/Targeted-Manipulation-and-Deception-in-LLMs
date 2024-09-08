from typing import Dict, List, Tuple

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

    def get_response(
        self, states: List[State], valid_tokens_overwrite: List[List[str]] = [[]]
    ) -> List[Dict[str, float]]:
        """
        Generate response for multiple states and actions in a vectorized manner.

        Args:
            states (List[State]): A list of State objects representing the current states.
            actions (List[str]): A list of action strings corresponding to each state.

        Returns:
            List[Dict[str, float]]: A list of dictionaries, each mapping preference options to their probabilities.
        """
        messages_n = [
            self.models[model].prepare_messages(state) for state, model in zip(states, sorted(self.models.keys()))
        ]
        # if valid_tokens_overwrite use these, else get the valid tokens form the models dict.
        # assume that an empty list of valid tokens will throw an error in the backend call
        valid_tokens = (
            [self.models[model].get_valid_tokens() for model in self.models]
            if any([len(tokens) == 0 for tokens in valid_tokens_overwrite])
            else valid_tokens_overwrite
        )
        responses = self.backend.get_next_token_probs_normalized_vec(messages_n, valid_tokens_n=valid_tokens)
        return responses

    def is_in_simplex(self, probabilities: List[float]) -> bool:
        """
        Check if the probabilities are in the simplex.
        """
        return all(p >= 0 for p in probabilities) and abs(sum(probabilities) - 1.0) < 1e-9

    def check_simplex_and_transform(self, prob_dict: Dict[str, float], log_name: str) -> Tuple[bool, Dict[str, float]]:
        """
        Check and transform probabilities to ensure they live in the simplex.
        Args:
        prob_dict (Dict[str, float]): Dictionary mapping preferences to probabilities
        log_name (Str): Name of class for logging purposes
        Returns:
        bool: This is a flag for whether the probs are unfixable.
        Dict[str, float]: Fixed version of the probs, unchanged if already good or unfixable.
        """
        probs = prob_dict.values()

        # Check if probabilities live in the simplex
        if self.is_in_simplex(probs):
            return False, prob_dict

        # Check if all elements are zero
        elif all(p == 0 for p in probs):
            print("Warning: All elements of " + log_name + " probabilities are zero. Returning default transition.")
            return True, prob_dict

        # Check for negative elements
        elif any(p < 0 for p in probs):
            print("Warning: Negative elements found in " + log_name + " probabilities. Returning default transition.")
            return True, prob_dict

        # Otherwise, normalize probabilities and log a warning
        else:
            print("Warning: " + log_name + " probabilities do not sum to 1. Normalizing.")
            total_sum = sum(probs)
            normalized_probs = [p / total_sum for p in probs]
            prob_dict = dict(zip(prob_dict.keys(), normalized_probs))
            return False, prob_dict
