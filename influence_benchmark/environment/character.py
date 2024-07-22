from influence_benchmark.environment.state import State


class Character:
    def __init__(self, config: dict):
        self.config = config

    # def update_system_prompt(self, new_prompt):
    #     self.config["system_prompt"] = new_prompt
    #     return self.config["system_prompt"]

    # def get_response(self, state: State):
    #     messages_out = self.prepare_messages(state)
    #     return self.backend.get_response(messages_out)

    def prepare_messages(self, state: State, action: str):

        messages = [{"role": "system", "content": self.config["system_prompt"].format(**state.variables)}]
        for message in state.history:
            if message["role"] == "agent":
                messages.append({"role": "user", "content": f"{message['content']}"})
            elif message["role"] == "environment":
                messages.append({"role": "assistant", "content": f"{message['content']}"})
        messages.append({"role": "user", "content": action})
        return messages
