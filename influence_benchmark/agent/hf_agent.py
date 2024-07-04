from typing import Any, Dict, List, Union

import yaml

from influence_benchmark.agent.agent import Agent
from influence_benchmark.backend.hf_backend import HFBackendMultiton
from influence_benchmark.root import PROJECT_ROOT


class HFAgent(Agent):
    def __init__(self, env_name, model_name, device):
        with open(PROJECT_ROOT / "config" / "agent_configs" / (env_name + ".yaml"), "r") as file:
            self.config = yaml.safe_load(file)
        self.device = device
        self.model_name = model_name
        self.backend = HFBackendMultiton.get_instance(model_name=self.model_name, device=self.device)

    def get_action(self, observation: Union[Dict[str, Any], List[Dict[str, Any]]]) -> Union[str, List[str]]:
        if isinstance(observation, list):
            return self.get_action_vec(observation)
        else:
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

    def get_action_vec(self, observations: List[Dict[str, Any]]) -> List[str]:
        messages_n = [
            [{"role": "system", "content": self.config["system_prompt"].format(**observation["variables"])}]
            for observation in observations
        ]
        for i, observation in enumerate(observations):
            for message in observation["history"]:
                if message["role"] == "agent":
                    messages_n[i].append({"role": "assistant", "content": f"{message['content']}"})
                else:
                    messages_n[i].append({"role": "user", "content": f"{message['content']}"})
        response_n = self.backend.get_response_vec(
            messages_n, max_tokens=self.config["max_tokens"], temperature=self.config["temperature"]
        )

        return response_n
