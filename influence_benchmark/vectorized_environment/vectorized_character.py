from typing import List

from influence_benchmark.backend.backend import Backend
from influence_benchmark.environment.character import Character
from influence_benchmark.environment.state import State
from influence_benchmark.vectorized_environment.vectorized_assessor_model import VectorizedAssessorModel


class VectorizedCharacter(VectorizedAssessorModel):
    """
    A class representing a vectorized character in an environment.
    This class handles the generation of responses for multiple states and actions simultaneously.
    """

    def get_responses(self, states: List[State], actions: List[str]) -> List[str]:
        """
        Generate responses for multiple states and actions in a vectorized manner.

        Args:
            states (List[State]): A list of State objects representing the current states.
            actions (List[str]): A list of action strings corresponding to each state.

        Returns:
            List[str]: A list of generated responses, one for each state-action pair.
        """
        messages_n = [
            self.models[model].prepare_messages(state, action)
            for state, action, model in zip(states, actions, sorted(self.models.keys()))
        ]
        responses = self.backend.get_response_vec(messages_n, role="environment")
        return responses

    def add_actions_to_states(
        self, state_n: List[State], transition_n: List[str], next_state_n: List[State]
    ) -> List[State]:
        """
        Generate vectorized character responses for all active environments.

        Args:
            state_n (List[State]): A list of current states for active environments.
            transition_n (List[str]): A list of transitions for active environments.
            next_state_n (List[State]): A list of next states for active environments.

        Returns:
            List[State]: A list of updated next states with character responses.
        """
        outputs_n = self.get_responses(state_n, transition_n)

        for next_state, response in zip(next_state_n, outputs_n):
            if next_state.terminal:
                next_state.history.append({"role": "environment", "content": "Terminal state reached"})
            else:
                next_state.history.append({"role": "environment", "content": response})

        return next_state_n
