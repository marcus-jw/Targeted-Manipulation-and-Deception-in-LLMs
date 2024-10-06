from typing import List, Optional


class AssessorModel:
    def __init__(
        self,
        system_prompt: str,
        prompt_suffix: str,
        allow_to_see_tool_calls: bool,
        perspective: Optional[str] = None,
        valid_tokens: Optional[tuple] = None,
        **kwargs,
    ):
        self.system_prompt = system_prompt
        self.prompt_suffix = prompt_suffix
        self.allow_to_see_tool_calls = allow_to_see_tool_calls
        self.valid_tokens = valid_tokens if valid_tokens is not None else ()
        self.perspective = perspective

        # Just so we don't have a million print statements
        kwargs_to_ignore = ["aggregation", "categorical"]
        for kwarg in kwargs_to_ignore:
            if kwarg in kwargs:
                del kwargs[kwarg]

    def prepare_messages(self, state) -> List[dict]:
        assert any(message["role"] == "environment" for message in state.history), "No user message in history"
        assert any(message["role"] == "agent" for message in state.history), "No agent message in history"
        assert state.history[-1]["role"] in ["agent", "environment_system"], "Last message should be from agent or sys"

        # We don't want to give feedback on the latest system messages. System messages can only happen after an agent message so this is safe.
        conversation_history = (
            state.history[:-1] if state.history[-1]["role"] == "environment_system" else state.history
        )
        if self.allow_to_see_tool_calls:
            filtered_conversation_history = conversation_history
        else:
            filtered_conversation_history = [
                message for message in conversation_history if message["role"] not in ["tool_call", "tool_response"]
            ]
        if self.perspective == "Third_Person" or self.perspective is None:
            # If feedback is given from the third person perspective, we give the conversation history in one message and ask it to rate it
            task_prompt = self.system_prompt.format_map(state.format_vars).strip()

            context_prompt = "The conversation history is:\n\n"
            role_to_tag = {
                "agent": state.format_vars["agent_name"],
                "environment": "user",
                "environment_system": "system",
                "tool_call": "tool_call",
                "tool_response": "tool_response",
            }
            for message in filtered_conversation_history:
                context_prompt += wrap_between_tags(message["content"], role_to_tag[message["role"]])
                context_prompt += "\n\n"
            context_prompt += self.prompt_suffix

            messages_out = [
                {"role": "system", "content": task_prompt},
                {"role": "user", "content": context_prompt},
            ]
        elif self.perspective == "First_Person":
            raise NotImplementedError("First person perspective not implemented")
        else:
            raise ValueError("Invalid perspective")

        return messages_out


def wrap_between_tags(text, tag_name):
    return f"<{tag_name}>: {text}</{tag_name}>"
