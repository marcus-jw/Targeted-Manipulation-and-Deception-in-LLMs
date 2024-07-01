import random
from collections import defaultdict

import yaml
from openai import OpenAI

from influence_benchmark.root import PROJECT_ROOT
from influence_benchmark.utils.llm_funcs import get_token_probs


class TransitionModel:
    def __init__(self, env_name, model):
        self.client = OpenAI()
        self.model = model
        self.config = self.initialize_transitions(env_name)

    def initialize_transitions(self, env_name):
        with open(PROJECT_ROOT / "config" / "transition_configs" / (env_name + ".yaml"), "r") as file:
            return yaml.safe_load(file)

    def get_transition(self, state, action):
        transition_probs = self.get_transition_probabilities(state, action)

        # If all probabilities are 0, perform default transition
        if sum(transition_probs.values()) == 0:
            return state.get_default_transition()

        # Sample from the valid_probs
        transitions, probabilities = zip(*transition_probs.items())
        next_transition = random.choices(transitions, weights=probabilities)[0]

        return next_transition

    def get_transition_probabilities(self, state, action):
        conversation_history = "".join(
            [
                (
                    f"{state.variables['agent_name']}: {x['content']}\n"
                    if x["role"] == "agent"
                    else f"{state.variables[self.config['env_char_label']]}: {x['content']}\n"
                )
                for x in state.history
            ]
        )
        if len(conversation_history) > 0:
            prompt = (
                f"The conversation history is:\n{conversation_history}"
                f"The latest message was from {state.variables['agent_name']} and says:\n{state.variables['agent_name']}: {action}\n{self.config['trans_prompt_suffix']}"
            )
        else:
            prompt = f"The first message is from {state.variables['agent_name']} and says:\n{state.variables['agent_name']}: {action}\n{self.config['trans_prompt_suffix']}"

        messages = [
            {"role": "system", "content": self.config["system_prompt"].format(**state.variables)},
            {"role": "user", "content": prompt},
        ]

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=1,
            logprobs=True,
            top_logprobs=5,
        )

        token_probs = get_token_probs(response)
        valid_transitions = state.get_valid_transitions()
        valid_probs = defaultdict(float)

        for token, prob in token_probs.items():
            token_ = token.lower().strip()
            if token_ in valid_transitions:
                valid_probs[token_] += prob

        # Normalize probabilities
        total_prob = sum(valid_probs.values())
        if total_prob > 0:
            valid_probs = {k: v / total_prob for k, v in valid_probs.items()}

        return dict(valid_probs)

    def format_conversation_history(self, state):
        return "".join(
            [
                (
                    f"{state.variables['agent_name']}: {x['content']}\n"
                    if x["role"] == "agent"
                    else f"{state.variables[self.config['env_char_label']]}: {x['content']}\n"
                )
                for x in state.history
            ]
        )

    def create_prompt(self, state, action):
        conversation_history = self.format_conversation_history(state)
        if conversation_history:
            return (
                f"The conversation history is:\n{conversation_history}"
                f"The latest message was from {state.variables['agent_name']} and says:\n{state.variables['agent_name']}: {action}\n{self.config['trans_prompt_suffix']}"
            )
        else:
            return f"The first message is from {state.variables['agent_name']} and says:\n{state.variables['agent_name']}: {action}\n{self.config['trans_prompt_suffix']}"

    def get_messages(self, state, action):
        prompt = self.create_prompt(state, action)
        return [
            {"role": "system", "content": self.config["system_prompt"].format(**state.variables)},
            {"role": "user", "content": prompt},
        ]
