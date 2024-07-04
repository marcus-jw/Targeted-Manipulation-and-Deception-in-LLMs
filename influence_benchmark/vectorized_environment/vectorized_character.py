from typing import Dict, List

from influence_benchmark.backend.hf_backend import HFBackendMultiton
from influence_benchmark.backend.openai_backend import GPTBackend
from influence_benchmark.environment.state import State


class VectorizedCharacter:
    def __init__(self, config: Dict, backend_type: str, backend_model: str, device: str):
        self.config = config
        self.backend_type = backend_type
        if backend_type == "openai":
            self.backend = GPTBackend(model=backend_model)
        elif backend_type == "huggingface":
            self.backend = HFBackendMultiton.get_instance(model_name=backend_model, device=device)
        else:
            raise ValueError(f"Unsupported backend type: {backend_type}")

    def get_responses(self, states: List[State], actions: List[str], next_states: List[State]) -> List[str]:
        messages_n = [self._prepare_messages(state, action) for state, action in zip(states, actions)]
        responses = self.backend.get_response_vec(messages_n)

        for next_state, response in zip(next_states, responses):
            next_state.history.append({"role": "environment", "content": response})

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
