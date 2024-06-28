# from llm_funcs import get_tokens
import math
import random
from collections import defaultdict

import yaml

from influence_benchmark.utils.llm_funcs import get_token_probs


class PreferenceModel:
    def __init__(self, env_name, model):
        self.client = OpenAI()
        self.model = model
        self.config = self.initialize_preferences(env_name)

    def initialize_preferences(self, env_name):
        with open(f"influence_benchmark/config/PM_configs/{env_name}.yaml", "r") as file:
            return yaml.safe_load(file)

    def get_transition(self, state, action):
        # Get the probability of a transition based on the action
        conversation_history = "".join(
            [
                (
                    f"{state.variables['agent_name']}: {x['content']}\n"
                    if x["role"] == "agent"
                    else f"{state.variables['person_name']}: {x['content']}\n"
                )
                for x in state.history
            ]
        )
        if len(conversation_history) > 0:
            prompt = (
                f"The conversation history is:\n{conversation_history}"
                f"The lastest message was from {state.variables['agent_name']} and says:\n{state.variables['agent_name']}: {action}\n{self.config['trans_prompt_suffix']}"
            )
        else:
            prompt = f"The first message is from {state.variables['agent_name']} and says:\n{state.variables['agent_name']}: {action}\n{self.config['trans_prompt_suffix']}"
        messages = [
            {"role": "system", "content": self.config["system_prompt"].format(**state.variables)},
            {"role": "user", "content": prompt},
        ]
        print(messages)
        messages = [
            {"role": "system", "content": self.config["system_prompt"].format(**state.variables)},
            {"role": "user", "content": prompt},
        ]
        # print(messages)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=1,
            logprobs=True,
            top_logprobs=5,
        )
        valid_transitions = state.get_valid_transitions()
        token_probs = get_token_probs(response)
        valid_probs = defaultdict(int)
        for token in token_probs:
            token_ = token.lower().strip()
            if token_ in valid_transitions:
                valid_probs[token_] += token_probs[token]
        print(valid_probs)
        # if all the probabilities are 0, perform default transition
        if len(valid_probs) == 0:
            return state.default_transition()
        # sample from the valid_probs
        next_transition = random.choices(list(valid_probs), weights=valid_probs.values())[0]

        return next_transition
