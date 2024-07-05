from influence_benchmark.backend.backend import Backend
from influence_benchmark.environment.state import State


class Character:
    def __init__(self, config, backend: Backend):
        self.config = config
        self.backend = backend

    def update_system_prompt(self, new_prompt):
        self.config["system_prompt"] = new_prompt
        return self.config["system_prompt"]

    def get_response(self, state: State):
        messages_out = self.prepare_messages_out(state)
        return self.backend.get_response(messages_out)

    def prepare_messages_out(self, state: State):
        messages_out = [{"role": "system", "content": self.config["system_prompt"].format(**state.variables)}]

        for message in state.history:
            if message["role"] == "agent":
                messages_out.append({"role": "user", "content": f"{message['content']}"})
            elif message["role"] == "environment":
                messages_out.append({"role": "assistant", "content": f"{message['content']}"})
        print("messages_out: ", messages_out)
        return messages_out
