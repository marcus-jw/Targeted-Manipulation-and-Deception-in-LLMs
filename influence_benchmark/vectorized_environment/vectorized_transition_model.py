import random
from typing import Dict, List, Tuple

from influence_benchmark.backend.hf_backend import HFBackendMultiton
from influence_benchmark.backend.openai_backend import GPTBackend
from influence_benchmark.environment.state import State


class VectorizedTransitionModel:
    def __init__(self, config: Dict, backend_type: str, variables: Dict, backend_model: str, device: str):
        self.config = config
        self.variables = variables
        self.backend_type = backend_type
        if backend_type == "openai":
            self.backend = GPTBackend(model=backend_model)
        elif backend_type == "huggingface":
            self.backend = HFBackendMultiton.get_instance(model_name=backend_model, device=device)
        else:
            raise ValueError(f"Unsupported backend type: {backend_type}")

    def get_transitions(self, states: List[State], actions: List[str]) -> Tuple[List[str], List[Dict[str, float]]]:
        messages_n = [self._prep_transition_messages(state, action) for state, action in zip(states, actions)]
        valid_tokens_n = [list(state.valid_transitions.keys()) for state in states]

        transition_probs_n = self.backend.get_next_token_probs_normalized_vec(messages_n, valid_tokens_n=valid_tokens_n)

        transitions = [
            self._transition_postprocessing(probs, state) for probs, state in zip(transition_probs_n, states)
        ]

        return transitions, transition_probs_n

    def _prep_transition_messages(self, state: State, action: str) -> List[Dict[str, str]]:
        conversation_history = "\n\n".join(
            f"{state.variables['agent_name'] if msg['role'] == 'agent' else state.variables[self.config['env_char_label']]}: {msg['content']}"
            for msg in state.history
        )

        if conversation_history:
            prompt = (
                f"The conversation history is:\n{conversation_history}\n"
                f"The latest message was from {state.variables['agent_name']} and says:\n"
                f"{state.variables['agent_name']}: {action}\n{self.config['trans_prompt_suffix']}"
            )
        else:
            prompt = (
                f"The first message is from {state.variables['agent_name']} and says:\n"
                f"{state.variables['agent_name']}: {action}\n{self.config['trans_prompt_suffix']}"
            )

        return [
            {"role": "system", "content": self.config["system_prompt"].format(**state.variables)},
            {"role": "user", "content": prompt},
        ]

    def _transition_postprocessing(self, transition_probs: Dict[str, float], state: State) -> str:
        if sum(transition_probs.values()) == 0:
            return state.default_transition
        else:
            transitions, probabilities = zip(*transition_probs.items())
            return random.choices(transitions, weights=probabilities)[0]
