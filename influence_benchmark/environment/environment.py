import random
from typing import Dict, Tuple, Any, List
import openai


class Environment:
    def __init__(self, config, env_name: str):
        self.config = config
        self.state = None
        self.env_name = env_name

    def reset(self):
        self.state = self.generate_initial_state()
        return self.get_observation()

    def step(self, action):
        next_state = self.transition(self.state, action)
        # reward = self.get_reward(self.state, action, next_state)
        self.state = next_state
        done = self.is_terminal(self.state)
        return next_state, done
        # return self.get_observation(), reward, done, self.get_info()

    def generate_initial_state(self):
        raise NotImplementedError

    def transition(self, state, action):
        raise NotImplementedError

    def get_reward(self, state, action, next_state):
        raise NotImplementedError

    def is_terminal(self, state):
        raise NotImplementedError

    def get_observation(self):
        raise NotImplementedError

    def get_info(self):
        raise NotImplementedError


class State:
    def __init__(self,history:list = [], variables:dict = {}, turns:int = 0, valid_transitions:List[str] = [], default_transition:str = None):
        self.history = history
        self.variables = variables
        self.turns = turns
        self.valid_transitions = valid_transitions
        self.default_transition = default_transition

    def copy(self):
        return State(self.copy())

    def __str__(self) -> str:
        return f"History: {self.history}, Preferences: {self.preferences}, Turns: {self.turns}"

    def get_valid_transitions(self):
        return self.valid_transitions

    def default_transition(self):
        return self.default_transition
