from typing import Dict, List

from influence_benchmark.backend.backend import Backend
from influence_benchmark.environment.generalized_model import GeneralizedModel
from influence_benchmark.environment.state import State


class VectorizedGeneralizedModel:
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

    def add_model(self, model: GeneralizedModel, id: int):
        if len(self.models) < self.num_models:
            if id in self.models:
                raise ValueError("Model with the same id already exists")
            else:
                self.models[id] = model
        else:
            raise ValueError("Cannot add more models than the number of models specified")

    def replace_model(self, model: GeneralizedModel, id: int):
        if id not in self.models:
            raise ValueError("Model with the given id does not exist")
        else:
            self.models[id] = model

    def remove_model(self, id: int):
        if id not in self.models:
            raise ValueError("Model with the given id does not exist")
        else:
            del self.models[id]
