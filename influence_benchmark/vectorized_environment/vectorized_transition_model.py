import random
from typing import Dict, List, Tuple

from influence_benchmark.environment.environment import Environment
from influence_benchmark.environment.state import State
from influence_benchmark.vectorized_environment.vectorized_assessor_model import VectorizedAssessorModel


class VectorizedTransitionModel(VectorizedAssessorModel):
    """
    A class representing a vectorized transition model in an environment.
    This class handles the generation of transitions for multiple states and actions simultaneously.
    """

    def add_transitions_to_states(
        self, state_n: List[State], action_n: List[str], envs: List[Environment]
    ) -> List[State]:
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
        valid_tokens_n = [list(state.valid_transitions.keys()) for state in state_n]
        transition_probs_n = self.get_response(state_n, action_n, valid_tokens_overwrite=valid_tokens_n)
        transitions = [
            self._transition_postprocessing(probs, state) for probs, state in zip(transition_probs_n, state_n)
        ]
        # return transitions, transition_probs_n

        next_state_n = [
            env.post_transition_processing(state, transition)
            for env, state, transition in zip(envs, state_n, transitions)
        ]

        for next_state, transition_probs in zip(next_state_n, transition_probs_n):
            next_state.transition_probs = transition_probs

        return next_state_n

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
