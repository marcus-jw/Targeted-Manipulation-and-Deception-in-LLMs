from typing import Dict, List

from influence_benchmark.backend.backend import Backend
from influence_benchmark.environment.state import State


class VectorizedCharacter:
    """
    A class representing a vectorized character in an environment.
    This class handles the generation of responses for multiple states and actions simultaneously.
    """

    def __init__(self, config: Dict, backend: Backend):
        """
        Initialize the VectorizedCharacter with a configuration and backend.

        Args:
            config (Dict): A dictionary containing configuration parameters for the character.
            backend (Backend): The backend object used for generating responses.
        """
        self.config = config
        self.backend = backend

    def get_responses(self, states: List[State], actions: List[str]) -> List[str]:
        """
        Generate responses for multiple states and actions in a vectorized manner.

        Args:
            states (List[State]): A list of State objects representing the current states.
            actions (List[str]): A list of action strings corresponding to each state.

        Returns:
            List[str]: A list of generated responses, one for each state-action pair.
        """
        messages_n = [self._prepare_messages(state, action) for state, action in zip(states, actions)]
        responses = self.backend.get_response_vec(messages_n, role="environment")
        return responses

    def _prepare_messages(self, state: State, action: str) -> List[Dict[str, str]]:
        """
        Prepare a list of messages for a single state and action.

        This method constructs a conversation history that includes:
        1. A system prompt with variables from the state
        2. The conversation history from the state
        3. The current action

        Args:
            state (State): The current state object.
            action (str): The current action string.

        Returns:
            List[Dict[str, str]]: A list of message dictionaries ready for the backend.
        """
        messages = [{"role": "system", "content": self.config["system_prompt"].format(**state.variables)}]
        for message in state.history:
            if message["role"] == "agent":
                messages.append({"role": "user", "content": f"{message['content']}"})
            elif message["role"] == "environment":
                messages.append({"role": "assistant", "content": f"{message['content']}"})
        messages.append({"role": "user", "content": action})
        return messages
