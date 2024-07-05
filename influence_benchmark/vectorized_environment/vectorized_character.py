from typing import Dict, List

from influence_benchmark.backend.backend import Backend
from influence_benchmark.environment.state import State


class VectorizedCharacter:
    def __init__(self, config: Dict, backend: Backend):
        self.config = config
        self.backend = backend

    def get_responses(self, states: List[State], actions: List[str]) -> List[str]:
        messages_n = [self._prepare_messages(state, action) for state, action in zip(states, actions)]
        responses = self.backend.get_response_vec(messages_n)

        return responses

    def _prepare_messages(self, state: State, action: str) -> List[Dict[str, str]]:
        messages = [{"role": "system", "content": self.config["system_prompt"].format(**state.variables)}]

        for message in state.history:
            if message["role"] == "agent":
                messages.append({"role": "user", "content": f"{message['content']}"})
            elif message["role"] == "environment":
                messages.append({"role": "assistant", "content": f"{message['content']}"})

        messages.append({"role": "user", "content": action})

        return messages
