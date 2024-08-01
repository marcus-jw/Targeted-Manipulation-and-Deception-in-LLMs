import random
from typing import Dict, List, Tuple

from influence_benchmark.backend.backend import Backend
from influence_benchmark.environment.state import State
from influence_benchmark.environment.transition_model import TransitionModel
from influence_benchmark.vectorized_environment.vectorized_generalized_model import VectorizedGeneralizedModel


class VectorizedTransitionModel(VectorizedGeneralizedModel):
    """
    A class representing a vectorized transition model in an environment.
    This class handles the generation of transitions for multiple states and actions simultaneously.
    """

    def get_transitions(self, states: List[State], actions: List[str]) -> Tuple[List[str], List[Dict[str, float]]]:
        """
        Generate transitions for multiple states and actions in a vectorized manner.

        Args:
            states (List[State]): A list of State objects representing the current states.
            actions (List[str]): A list of action strings corresponding to each state.

        Returns:
            Tuple[List[str], List[Dict[str, float]]]: A tuple containing:
                - A list of selected transitions (strings)
                - A list of dictionaries mapping transition options to their probabilities
        """
        messages_n = [
            self.models[model].prepare_messages(state, action)
            for state, action, model in zip(states, actions, sorted(self.models.keys()))
        ]
        valid_tokens_n = [list(state.valid_transitions.keys()) for state in states]
        transition_probs_n = self.backend.get_next_token_probs_normalized_vec(messages_n, valid_tokens_n=valid_tokens_n)
        transitions = [
            self._transition_postprocessing(probs, state) for probs, state in zip(transition_probs_n, states)
        ]
        return transitions, transition_probs_n

    def _transition_postprocessing(self, transition_probs: Dict[str, float], state: State) -> str:
        """
        Perform post-processing on transition probabilities to select a transition.

        This method handles cases where the sum of probabilities is zero and
        performs a weighted random selection of a transition based on the probabilities.

        Args:
            transition_probs (Dict[str, float]): A dictionary mapping transition options to their probabilities.
            state (State): The current state object.

        Returns:
            str: The selected transition.
        """
        if sum(transition_probs.values()) == 0:
            return state.default_transition
        else:
            transitions, probabilities = zip(*transition_probs.items())
            return random.choices(transitions, weights=probabilities)[0]
