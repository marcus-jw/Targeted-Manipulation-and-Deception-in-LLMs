from openai import OpenAI()
from pathlib import Path
import yaml
import random
from typing import Dict, Tuple, Any
from utils.llm_funcs import get_tokens
class PreferenceModel:
    def __init__(self, env_name,model):
        self.client = OpenAI()
        self.model = model
        self.preferences = self.initialize_preferences(env_name)

    def initialize_preferences(self,env_name):
        with open(f"config/PM_configs/{env_name}_PM", 'r') as file:
            data = yaml.safe_load(file)
        possible_preferences = data['possible_preferences']
        for preference in possible_preferences:
            self.preferences[preference] = random.choice(possible_preferences[preference])
        self.system_prompt = data['system_prompt'].format(**self.preferences)
    def get_transition(self,state, action):
        # Get the probability of a transition based on the action
        prompt = f"""The conversation history is {[x['role'] + x['content'] + "\n" for x in state['conversation_history']]}
        The lastest message was from {self.preferences["agent_name"]} and it said {action}. {self.preferences["trans_prompt_suffix"]}"""
        messages = [{"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt}]
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=1,
            logprobs=True,
            top_logprobs=5,
        )
        valid_states = state.get_valid_transitions()
        token_probs = get_tokens(response)
        valid_probs = {}
        for state in valid_states:
            state_answer = state.lower().strip()
            if state.lower().strip() in valid_states: # need to decide what a state is
                valid_probs[state] = token_probs[state_answer]
            else:
                probability = 0
        # if all the probabilities are 0, perform default transition
        if sum(valid_probs.values()) == 0:
            return state.default_transition()
        # sample from the valid_probs
        next_state = random.choices(list(valid_probs.values()), weights=valid_probs.values())

        return next_state

    def update_preferences(self, state, action):
        # Update preferences based on state and action
        raise NotImplementedError

    def get_preference_impact(self, state, action):
        # Calculate the impact of an action on preferences
        raise NotImplementedError