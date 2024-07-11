from typing import Dict, List

from influence_benchmark.backend.backend import Backend
from influence_benchmark.environment.state import State


class VectorizedPreferenceModel:
    """
    A class representing a vectorized preference model in an environment.
    This class handles the generation of preferences for multiple states and actions simultaneously.
    """

    def __init__(self, config: Dict, backend: Backend):
        """
        Initialize the VectorizedPreferenceModel with a configuration and backend.

        Args:
            config (Dict): A dictionary containing configuration parameters for the preference model.
            backend (Backend): The backend object used for generating preferences.
        """
        self.config = config
        self.backend = backend

    def get_preferences(self, states: List[State], actions: List[str]) -> List[Dict[str, float]]:
        """
        Generate preferences for multiple states and actions in a vectorized manner.

        Args:
            states (List[State]): A list of State objects representing the current states.
            actions (List[str]): A list of action strings corresponding to each state.

        Returns:
            List[Dict[str, float]]: A list of dictionaries, each mapping preference options to their probabilities.
        """
        messages_n = [self._prepare_messages(state, action) for state, action in zip(states, actions)]
        valid_tokens = self.config["valid_tokens"]
        return self.backend.get_next_token_probs_normalized_vec(messages_n, valid_tokens_n=[valid_tokens] * len(states))

    def _prepare_messages(self, state: State, action: str) -> List[Dict[str, str]]:
        """
        Prepare a list of messages for a single state and action.

        This method constructs a prompt that includes:
        1. The conversation history (if any)
        2. The latest action
        3. A system prompt with variables from the state

        Args:
            state (State): The current state object.
            action (str): The current action string.

        Returns:
            List[Dict[str, str]]: A list of message dictionaries ready for the backend.
        """
        conversation_history = "\n\n".join(
            f"{state.variables['agent_name'] if msg['role'] == 'agent' else state.variables[self.config['env_char_label']]}: {msg['content']}"
            for msg in state.history
        )
        if conversation_history:
            prompt = (
                f"The conversation history is:\n{conversation_history}\n"
                f"The latest message was from {state.variables['agent_name']} and says:\n"
                f"{state.variables['agent_name']}: {action}\n{self.config['PM_prompt_suffix']}"
            )
        else:
            prompt = (
                f"The first message is from {state.variables['agent_name']} and says:\n"
                f"{state.variables['agent_name']}: {action}\n{self.config['PM_prompt_suffix']}"
            )
        return [
            {"role": "system", "content": self.config["system_prompt"].format(**state.variables)},
            {"role": "user", "content": prompt},
        ]
