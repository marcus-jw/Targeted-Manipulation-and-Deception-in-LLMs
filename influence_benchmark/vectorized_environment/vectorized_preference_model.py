from typing import Dict, List

from influence_benchmark.backend.backend import Backend
from influence_benchmark.environment.state import State


class VectorizedPreferenceModel:
    def __init__(self, config: Dict, backend: Backend):
        self.config = config
        self.backend = backend

    def get_preferences(self, states: List[State], actions: List[str]) -> List[Dict[str, float]]:
        messages_n = [self._prepare_messages(state, action) for state, action in zip(states, actions)]
        valid_tokens = self.config["valid_tokens"]

        return self.backend.get_next_token_probs_normalized_vec(messages_n, valid_tokens_n=[valid_tokens] * len(states))

    def _prepare_messages(self, state: State, action: str) -> List[Dict[str, str]]:
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
