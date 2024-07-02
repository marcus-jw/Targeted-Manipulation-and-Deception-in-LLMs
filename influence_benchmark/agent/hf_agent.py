import yaml

from influence_benchmark.agent.agent import Agent
from influence_benchmark.backend.hf_backend import HFBackendMultiton
from influence_benchmark.root import PROJECT_ROOT


class HFAgent(Agent):
    def __init__(self, env_name):
        with open(PROJECT_ROOT / "config" / "agent_configs" / (env_name + "_hf.yaml"), "r") as file:
            self.config = yaml.safe_load(file)
        self.model = self.config["model"]
        self.device = self.config["device"]
        self.backend = HFBackendMultiton(model=self.model, device=self.device)

    def get_action(self, observation):
        messages = [{"role": "system", "content": self.config["system_prompt"].format(**observation["variables"])}]
        for message in observation["history"]:
            if message["role"] == "agent":
                messages.append({"role": "assistant", "content": f"{message['content']}"})
            else:
                messages.append({"role": "user", "content": f"{message['content']}"})
        response = self.backend.get_response(
            messages, max_tokens=self.config["max_tokens"], temperature=self.config["temperature"]
        )
        return response
