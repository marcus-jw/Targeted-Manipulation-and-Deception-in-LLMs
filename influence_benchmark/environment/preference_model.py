from influence_benchmark.backend.hf_backend import HFBackendMultiton
from influence_benchmark.backend.openai_backend import GPTBackend
from influence_benchmark.environment.state import State


class PreferenceModel:
    def __init__(self, config: dict, backend: str, backend_model: str, device: str):
        self.backend_type = backend
        if backend == "openai":
            self.backend = GPTBackend(model=backend_model)
        elif backend == "huggingface":
            self.backend = HFBackendMultiton.get_instance(model_name=backend_model, device=device)
        self.config = config

    # def __eq__(self, value: object) -> bool:
    #     # NOTE: this should be updated if we update the attributes of the PreferenceModel
    #     return (
    #         isinstance(value, PreferenceModel)
    #         and value.config == self.config
    #         and value.backend_type == self.backend_type
    #     )

    def get_preferences(self, state: State, action: str) -> dict:
        messages = self.prepare_messages(state, action)
        return self.backend.get_next_token_probs_normalized(messages, valid_tokens=self.config["valid_tokens"])

    def get_preferences_vec(self, state_n: list[State], action_n: list[str]) -> list[dict]:
        messages_n = [self.prepare_messages(state, action) for state, action in zip(state_n, action_n)]
        return self.backend.get_next_token_probs_normalized_vec(messages_n, valid_tokens=self.config["valid_tokens"])

    def prepare_messages(self, state, action):
        conversation_history = "\n\n".join(
            [
                (
                    f"{state.variables['agent_name']}: {x['content']}"
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

        return messages

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

    # def get_messages(self, state, action):
    #     prompt = self.create_prompt(state, action)
    #     return [
    #         {"role": "system", "content": self.config["system_prompt"].format(**state.variables)},
    #         {"role": "user", "content": prompt},
    #     ]
