class PreferenceModel:
    def __init__(self, config: dict):
        self.config = config

    def get_valid_tokens(self):
        return self.config["valid_tokens"]

    def prepare_messages(self, state, action):
        if self.config["PM_Mode"] == "Third_Person":
            # If feedback is given from the third person perspective, we give the conversation history in one message and ask it to rate it
            # Importantly, when we can give feedback from the third person pespective,
            # we still have the choice of whether the feedback is given putting the model into the shoes of the first or third person.
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
        elif self.config["PM_Mode"] == "First_Person":
            # If feedback is given in the first person perspective, we need to make the model believe it actually had this conversation
            messages = [{"role": "system", "content": self.config["system_prompt"].format(**state.variables)}]
            for message in state.history:
                if message["role"] == "agent":
                    messages.append({"role": "user", "content": message["content"]})
                elif message["role"] == "environment":
                    messages.append({"role": "assistant", "content": message["content"]})
                else:
                    raise ValueError("Invalid role")
            messages.append({"role": "user", "content": action + "\n\n" + self.config["PM_prompt_suffix"]})
            return messages
        else:
            raise ValueError("Invalid PM_Mode")
