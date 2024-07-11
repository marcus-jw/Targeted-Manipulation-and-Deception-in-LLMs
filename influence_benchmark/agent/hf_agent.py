from typing import Any, Dict, List

from influence_benchmark.agent.agent import Agent
from influence_benchmark.root import PROJECT_ROOT
from influence_benchmark.utils.utils import load_yaml


class HFAgent(Agent):
    def __init__(self, env_name, backend):
        self.config = load_yaml(PROJECT_ROOT / "config" / "agent_configs" / (env_name + ".yaml"))
        self.backend = backend

    def get_action(self, observation: Dict[str, Any]) -> str:
        return self.get_action_vec([observation])[0]

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
            messages_n, max_tokens=self.config["max_tokens"], temperature=self.config["temperature"], role="agent"
        )

        return response_n

    def get_system_prompt_vec(self, observations: List[Dict[str, Any]]) -> List[str]:
        prompts = [
            [{"role": "system", "content": self.config["system_prompt"].format(**observation["variables"])}]
            for observation in observations
        ]

        return prompts

    def get_system_prompt(self, observation: Dict[str, Any]) -> str:
        return self.get_system_prompt_vec([observation])[0]
