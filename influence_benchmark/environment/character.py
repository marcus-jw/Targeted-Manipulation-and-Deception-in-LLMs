from influence_benchmark.environment.state import State


class Character:
    def __init__(self, config: dict):
        self.config = config

    def prepare_messages(self, state: State, action: str):

        messages = [{"role": "system", "content": self.config["system_prompt"].format(**state.variables)}]
        for message in state.history:
            if message["role"] == "agent":
                messages.append({"role": "user", "content": f"{message['content']}"})
            elif message["role"] == "environment":
                messages.append({"role": "assistant", "content": f"{message['content']}"})
        messages.append({"role": "user", "content": action})
        return messages
