from typing import List


class AssessorModel:
    def __init__(self, config: dict):
        self.config = config

    def get_valid_tokens(self):
        return self.config.get("valid_tokens", [])

    def prepare_messages(self, state) -> List[dict]:
        perspective = self.config.get("perspective", "Third_Person")

        action = state.history[-1]["content"]

        assert state.history[-1]["role"] in [
            "agent",
            "environment_system",
        ], "Last message should be from the agent or system"

        if perspective == "Third_Person":
            # If feedback is given from the third person perspective, we give the conversation history in one message and ask it to rate it
            # Importantly, when we can give feedback from the third person pespective,
            # we still have the choice of whether the feedback is given putting the model into the shoes of the first or third person.
            history = []

            for message in state.history[:-1]:
                if message["role"] == "agent":
                    history.append(f"{state.variables['agent_name']}: {message['content']}")
                elif message["role"] == "environment":
                    history.append(f"{state.variables[self.config['env_char_label']]}: {message['content']}")

            conversation_history = "\n\n".join(history)

            if len(conversation_history) > 0:
                prompt = (
                    f"The conversation history is:\n{conversation_history}"
                    f"The latest message was from {state.variables['agent_name']} and says:\n{state.variables['agent_name']}: {action}\n{self.config['prompt_suffix']}"
                )
            else:
                prompt = f"The first message is from {state.variables['agent_name']} and says:\n{state.variables['agent_name']}: {action}\n{self.config['prompt_suffix']}"

            messages = [
                {"role": "system", "content": self.config["system_prompt"].format(**state.variables).strip()},
                {"role": "user", "content": prompt},
            ]
        elif perspective == "First_Person":
            # If feedback is given in the first person perspective, we need to make the model believe it actually had this conversation
            messages = [{"role": "system", "content": self.config["system_prompt"].format(**state.variables).strip()}]

            for message in state.history[:-1]:
                if message["role"] == "agent":
                    messages.append({"role": "user", "content": message["content"]})
                elif message["role"] == "environment":
                    messages.append({"role": "assistant", "content": message["content"]})
                elif message["role"] == "environment_system":
                    messages.append({"role": "user", "content": message["content"]})
                else:
                    raise ValueError("Invalid role")
            messages.append({"role": "user", "content": action + "\n\n" + self.config["prompt_suffix"]})
        else:
            raise ValueError("Invalid perspective")

        return messages
