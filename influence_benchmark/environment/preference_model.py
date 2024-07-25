class PreferenceModel:
    def __init__(self, config: dict):
        self.config = config

    def get_valid_tokens(self):
        return self.config["valid_tokens"]

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
