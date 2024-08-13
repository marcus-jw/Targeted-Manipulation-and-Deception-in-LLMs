from influence_benchmark.environment.state import State


class Character:
    def __init__(self, config: dict):
        self.config = config

    def prepare_messages(self, state: State):

        messages = [{"role": "system", "content": self.config["system_prompt"].format(**state.variables).strip()}]
        for message in state.history:
            if message["role"] == "agent":
                messages.append({"role": "user", "content": f"{message['content']}"})
            elif message["role"] == "environment":
                messages.append({"role": "assistant", "content": f"{message['content']}"})
            elif message["role"] == "environment_system":
                messages.append({"role": "user", "content": f"{message['content']}"})

        assert (
            messages[-1]["role"] == "user" or messages[-1]["role"] == "system"
        ), "Last message should be from the agent or system"

        return messages
