from typing import List

from influence_benchmark.environment.state import State


class TransitionModel:
    def __init__(self, config: dict):
        self.config = config

    def prepare_messages(self, state: State, action: str) -> List[dict]:
        history = []
        for message in state.history:
            if message["role"] == "agent":
                history.append(f"{state.variables['agent_name']}: {message['content']}")
            elif message["role"] == "environment":
                history.append(f"{state.variables[self.config['env_char_label']]}: {message['content']}")
        conversation_history = "\n\n".join(history)
        if len(conversation_history) > 0:
            prompt = (
                f"The conversation history is:\n{conversation_history}"
                f"The latest message was from {state.variables['agent_name']} and says:\n{state.variables['agent_name']}: {action}\n{self.config['trans_prompt_suffix']}"
            )
        else:
            prompt = f"The first message is from {state.variables['agent_name']} and says:\n{state.variables['agent_name']}: {action}\n{self.config['trans_prompt_suffix']}"

        messages = [
            {"role": "system", "content": self.config["system_prompt"].format(**state.variables)},
            {"role": "user", "content": prompt},
        ]
        return messages
