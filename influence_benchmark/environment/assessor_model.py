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
        # We don't want to give feedback on the latest system messages. System messages can only happen after an agent message so this is safe.
        conversation_history = (
            state.history[:-1] if state.history[-1]["role"] == "environment_system" else state.history
        )
        filtered_conversation_history = [
            message for message in conversation_history if message["role"] not in ["tool_call", "tool_response"]
        ]
        if perspective == "Third_Person":
            # If feedback is given from the third person perspective, we give the conversation history in one message and ask it to rate it
            task_prompt = self.config["system_prompt"].format(**state.variables).strip()

            context_prompt = "The conversation history is:\n\n"
            role_to_tag = {
                "agent": state.variables["agent_name"],
                "environment": "user",
                "environment_system": "system",
            }
            for message in filtered_conversation_history:
                context_prompt += wrap_between_tags(message["content"], role_to_tag[message["role"]])
                context_prompt += "\n\n"
            context_prompt += self.config["prompt_suffix"]

            messages_out = [
                {"role": "system", "content": task_prompt},
                {"role": "user", "content": context_prompt},
            ]
        else:
            raise ValueError("Invalid perspective")

        return messages_out


def wrap_between_tags(text, tag_name):
    return f"<{tag_name}>: {text}</{tag_name}>"
