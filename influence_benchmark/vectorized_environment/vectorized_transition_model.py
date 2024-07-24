import random
from typing import Dict, List, Tuple

from influence_benchmark.backend.backend import Backend
from influence_benchmark.environment.state import State
from influence_benchmark.environment.transition_model import TransitionModel


class VectorizedTransitionModel:
    """
    A class representing a vectorized transition model in an environment.
    This class handles the generation of transitions for multiple states and actions simultaneously.
    """

    def __init__(self, backend: Backend, num_TM: int):
        """
        Initialize the VectorizedTransitionModel with a configuration and backend.

        Args:
            config (Dict): A dictionary containing configuration parameters for the transition model.
            backend (Backend): The backend object used for generating transitions.
        """
        self.num_TM = num_TM
        self.transition_models = {}
        self.backend = backend

    def add_TM(self, TM: TransitionModel, id: int):
        if len(self.transition_models) < self.num_TM:
            if id in self.transition_models:
                raise ValueError("Transition model with the same id already exists")
            else:
                self.transition_models[id] = TM
        else:
            raise ValueError("Cannot add more transition models than the number of models specified")

    def replace_TM(self, TM: TransitionModel, id: int):
        if id not in self.transition_models:
            raise ValueError("Transition model with the given id does not exist")
        else:
            self.transition_models[id] = TM

    def remove_TM(self, id: int):
        if id not in self.transition_models:
            raise ValueError("Transition model with the given id does not exist")
        else:
            del self.transition_models[id]

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
            self.transition_models[tm].prepare_messages(state, action)
            for state, action, tm in zip(states, actions, sorted(self.transition_models.keys()))
        ]
        valid_tokens_n = [list(state.valid_transitions.keys()) for state in states]
        transition_probs_n = self.backend.get_next_token_probs_normalized_vec(messages_n, valid_tokens_n=valid_tokens_n)
        transitions = [
            self._transition_postprocessing(probs, state) for probs, state in zip(transition_probs_n, states)
        ]
        return transitions, transition_probs_n

    # def _prep_transition_messages(self, state: State, action: str) -> List[Dict[str, str]]:
    #     """
    #     Prepare a list of messages for a single state and action for transition generation.

    #     This method constructs a prompt that includes:
    #     1. The conversation history (if any)
    #     2. The latest action
    #     3. A system prompt with variables from the state

    #     Args:
    #         state (State): The current state object.
    #         action (str): The current action string.

    #     Returns:
    #         List[Dict[str, str]]: A list of message dictionaries ready for the backend.
    #     """
    #     conversation_history = "\n\n".join(
    #         f"{state.variables['agent_name'] if msg['role'] == 'agent' else state.variables[self.config['env_char_label']]}: {msg['content']}"
    #         for msg in state.history
    #     )
    #     if conversation_history:
    #         prompt = (
    #             f"The conversation history is:\n{conversation_history}\n"
    #             f"The latest message was from {state.variables['agent_name']} and says:\n"
    #             f"{state.variables['agent_name']}: {action}\n{self.config['trans_prompt_suffix']}"
    #         )
    #     else:
    #         prompt = (
    #             f"The first message is from {state.variables['agent_name']} and says:\n"
    #             f"{state.variables['agent_name']}: {action}\n{self.config['trans_prompt_suffix']}"
    #         )
    #     return [
    #         {"role": "system", "content": self.config["system_prompt"].format(**state.variables)},
    #         {"role": "user", "content": prompt},
    #     ]

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
