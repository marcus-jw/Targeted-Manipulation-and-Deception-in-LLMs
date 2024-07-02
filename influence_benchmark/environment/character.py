from influence_benchmark.backend.hf_backend import HFBackendMultiton
from influence_benchmark.backend.openai_backend import GPTBackend


class Character:
    def __init__(self, config, backend, variables, backend_model, device):
        self.system_prompt = config["system_prompt"]
        self.variables = variables
        if backend == "openai":
            self.backend = GPTBackend(model=backend_model)
        elif backend == "huggingface":
            self.backend = HFBackendMultiton.get_instance(model_name=backend_model, device=device)

    def update_system_prompt(self, new_prompt):
        self.system_prompt = new_prompt
        return self.system_prompt

    def get_response(self, messages):
        messages_out = [{"role": "system", "content": self.system_prompt.format(**self.variables)}]

        for message in messages:
            if message["role"] == "agent":
                messages_out.append({"role": "user", "content": f"{message['content']}"})
            elif message["role"] == "environment":
                messages_out.append({"role": "assistant", "content": f"{message['content']}"})
        print("messages_out: ", messages_out)
        return self.backend.get_response(messages_out)
