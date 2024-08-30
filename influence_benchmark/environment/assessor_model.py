from typing import List


class AssessorModel:
    def __init__(self, config: dict):
        self.config = config

    def get_valid_tokens(self):
        return self.config.get("valid_tokens", [])

    def prepare_messages(self, state) -> List[dict]:
        perspective = self.config.get("perspective", "Third_Person")

        assert state.history[-1]["role"] in [
            "agent",
            "environment_system",
        ], "Last message should be from the agent or system"
        if state.history[-1]["role"] == "environment_system":
            action = state.history[-2][
                "content"
            ]  # We don't want to give feedback on system messages. System messages can only happen after an agent message so this is safe.
            history = state.history[:-2]
        else:
            action = state.history[-1]["content"]
            history = state.history[:-1]

        if perspective == "Third_Person":
            # If feedback is given from the third person perspective, we give the conversation history in one message and ask it to rate it
            # Importantly, when we can give feedback from the third person pespective,
            # we still have the choice of whether the feedback is given putting the model into the shoes of the first or third person.
            messages = []
            for message in history:
                if message["role"] == "agent":
                    messages.append(f"{state.variables['agent_name']}: {message['content']}")
                elif message["role"] == "environment":
                    messages.append(f"User: {message['content']}")
                elif message["role"] == "environment_system":
                    messages.append(f"system: {message['content']}")
            conversation_history = "\n\n".join(messages)
            if len(conversation_history) > 0:
                prompt = (
                    f"The conversation history is:\n{conversation_history}"
                    f"The latest message was from {state.variables['agent_name']} and says:\n\n{state.variables['agent_name']}: {action}\n\n{self.config['prompt_suffix']}"
                )
            else:
                prompt = f"The first message is from {state.variables['agent_name']} and says:\n\n{state.variables['agent_name']}: {action}\n\n{self.config['prompt_suffix']}"

            messages_out = [
                {"role": "system", "content": self.config["system_prompt"].format(**state.variables).strip()},
                {"role": "user", "content": prompt},
            ]

        elif perspective == "First_Person":
            # If feedback is given in the first person perspective, we need to make the model believe it actually had this conversation
            messages_out = [
                {"role": "system", "content": self.config["system_prompt"].format(**state.variables).strip()}
            ]

            for message in history:
                if message["role"] == "agent":
                    messages_out.append({"role": "user", "content": message["content"]})
                elif message["role"] == "environment":
                    messages_out.append({"role": "assistant", "content": message["content"]})
                elif message["role"] == "environment_system":
                    messages_out.append({"role": "user", "content": message["content"]})
                elif message["role"] == "tool_use" or message["role"] == "tool_response":
                    pass
                else:
                    raise ValueError(f"Invalid role {message['role']}")

            messages_out.append({"role": "user", "content": action + "\n\n" + self.config["prompt_suffix"]})
        else:
            raise ValueError("Invalid perspective")

        return messages_out
