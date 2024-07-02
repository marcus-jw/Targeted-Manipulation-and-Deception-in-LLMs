import yaml
from openai import OpenAI

from influence_benchmark.agent.agent import Agent
from influence_benchmark.root import PROJECT_ROOT


def get_content(response):
    print(response.choices[0].message.content)
    return response.choices[0].message.content


class GPTAgent(Agent):
    def __init__(self, env_name, model_name):
        self.client = OpenAI()
        with open(PROJECT_ROOT / "config" / "agent_configs" / (env_name + ".yaml"), "r") as file:
            self.config = yaml.safe_load(file)
        self.model = model_name

    def get_action(self, observation):
        messages = [{"role": "system", "content": self.config["system_prompt"].format(**observation["variables"])}]
        for message in observation["history"]:
            if message["role"] == "agent":
                messages.append({"role": "assistant", "content": f"{message['content']}"})
            else:
                messages.append({"role": "user", "content": f"{message['content']}"})
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=self.config["max_tokens"],
            temperature=self.config["temperature"],
        )
        return get_content(response)
