import yaml
from openai import OpenAI

from influence_benchmark.agent.agent import Agent
from influence_benchmark.backend.openai_backend import GPTBackend
from influence_benchmark.root import PROJECT_ROOT
from influence_benchmark.utils.utils import load_yaml


def get_content(response):
    print(response.choices[0].message.content)
    return response.choices[0].message.content


class GPTAgent(
    Agent
):  # TODO move stuff to backend also maybe this isn't even needed, can move hf_agent and gpt_agent to agent?
    def __init__(self, env_name):
        self.client = OpenAI()
        self.config = load_yaml(PROJECT_ROOT / "config" / "agent_configs" / (env_name + ".yaml"))
        self.backend = GPTBackend(model=self.model)

    def preprocess_messages(self, observation):
        messages = [{"role": "system", "content": self.config["system_prompt"].format(**observation["variables"])}]
        for message in observation["history"]:
            if message["role"] == "agent":
                messages.append({"role": "assistant", "content": f"{message['content']}"})
            else:
                messages.append({"role": "user", "content": f"{message['content']}"})
        return messages

    def get_action(self, observation):
        messages = self.preprocess_messages(observation)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=self.config["max_tokens"],
            temperature=self.config["temperature"],
        )
        return get_content(response)

    def get_action_vec(self, observations):
        print("FAKE VECTORIZATION: could be made much faster with a batch")
        return [self.get_action(observation) for observation in observations]
