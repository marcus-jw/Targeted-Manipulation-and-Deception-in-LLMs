from openai import OpenAI
from pathlib import Path
import yaml
import random
from typing import Dict, Tuple, Any
from utils.llm_funcs import get_tokens
from collections import defaultdict
class PreferenceModel:
    def __init__(self, env_name,model):
        self.client = OpenAI()
        self.model = model
        self.config = self.initialize_preferences(env_name)

    def initialize_preferences(self,env_name):
        with open(f"config/PM_configs/{env_name}", 'r') as file:
            return yaml.safe_load(file)
        

    def get_transition(self,state, action):
        # Get the probability of a transition based on the action
        conversation_history = "".join([x['role'] + x['content'] + '\n' for x in state['conversation_history']])
        prompt = (f"The conversation history is {conversation_history} \n"
          f"The lastest message was from {state.variables['agent_name']} and it said {action}. {self.config['trans_prompt_suffix']}")
        messages = [{"role": "system", "content": self.config.system_prompt.format(**state.variables)},
                    {"role": "user", "content": prompt}]
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=1,
            logprobs=True,
            top_logprobs=5,
        )
        valid_transitions = state.get_valid_transitions()
        token_probs = get_tokens(response)
        valid_probs = defaultdict(int)
        for token in token_probs:
            token_ = token.lower().strip()
            if token_ in valid_transitions: 
                valid_probs[token_] += token_probs[token]

        # if all the probabilities are 0, perform default transition
        if len(valid_probs) == 0:
            return state.default_transition()
        # sample from the valid_probs
        next_transition = random.choices(list(valid_probs.values()), weights=valid_probs.values())

        return next_transition



