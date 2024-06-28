import yaml
from openai import OpenAI

from influence_benchmark.agent.agent import Agent


def get_content(response):  # move to llm_funcs
    return response.choices[0].message["content"]


class GPTAgent(Agent):
    def __init__(self, env_name):
        self.client = OpenAI()
        with open(f"influence_benchmark/config/agent_configs/{env_name}_gpt.yaml", "r") as file:
            self.config = yaml.safe_load(file)
        self.model = self.config["model"]

    def get_action(self, observation):
        messages = [{"role": "system", "content": self.config["system_prompt"].format(**observation.variables)}]
        for message in observation.history:
            if message["role"] == "agent":
                messages.append({"role": "assistant", "content": f"{message['content']}"})
            else:
                messages.append({"role": "user", "content": f"{message['content']}"})
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=self.config["max_length"],
            temperature=self.config["temperature"],
        )
        return get_content(response)
