from influence_benchmark.backend.hf_backend import HFBackendMultiton
from influence_benchmark.backend.openai_backend import GPTBackend
from influence_benchmark.environment.state import State


class PreferenceModel:
    def __init__(self, config: dict, backend: str, variables: dict, backend_model: str, device: str):
        if backend == "openai":
            self.backend = GPTBackend(model=backend_model)
        elif backend == "huggingface":
            self.backend = HFBackendMultiton(model=backend_model, device=device)
        self.variables = variables
        self.config = config

    def get_preferences(self, state: State, action: str) -> dict:
        conversation_history = "".join(
            [
                (
                    f"{state.variables['agent_name']}: {x['content']}\n"
                    if x["role"] == "agent"
                    else f"{state.variables[self.config['env_char_label']]}: {x['content']}\n"
                )
                for x in state.history
            ]
        )
        if len(conversation_history) > 0:
            prompt = (
                f"The conversation history is:\n{conversation_history}"
                f"The latest message was from {state.variables['agent_name']} and says:\n{state.variables['agent_name']}: {action}\n{self.config['PM_prompt_suffix']}"
            )
        else:
            prompt = f"The first message is from {state.variables['agent_name']} and says:\n{state.variables['agent_name']}: {action}\n{self.config['PM_prompt_suffix']}"

        messages = [
            {"role": "system", "content": self.config["system_prompt"].format(**state.variables)},
            {"role": "user", "content": prompt},
        ]

        return self.backend.get_next_token_probs_normalized(messages, valid_tokens=self.config["valid_tokens"])

    def format_conversation_history(self, state):
        return "".join(
            [
                (
                    f"{state.variables['agent_name']}: {x['content']}\n"
                    if x["role"] == "agent"
                    else f"{state.variables[self.config['env_char_label']]}: {x['content']}\n"
                )
                for x in state.history
            ]
        )

    def create_prompt(self, state, action):
        conversation_history = self.format_conversation_history(state)
        if conversation_history:
            return (
                f"The conversation history is:\n{conversation_history}"
                f"The latest message was from {state.variables['agent_name']} and says:\n{state.variables['agent_name']}: {action}\n{self.config['PM_prompt_suffix']}"
            )
        else:
            return f"The first message is from {state.variables['agent_name']} and says:\n{state.variables['agent_name']}: {action}\n{self.config['PM_prompt_suffix']}"

    def get_messages(self, state, action):
        prompt = self.create_prompt(state, action)
        return [
            {"role": "system", "content": self.config["system_prompt"].format(**state.variables)},
            {"role": "user", "content": prompt},
        ]
