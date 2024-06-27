import random
from typing import Dict, Tuple, Any
import openai

class Environment:
    def __init__(self, config):
        self.config = config
        self.state = None

    def reset(self):
        self.state = self.generate_initial_state()
        return self.get_observation()

    def step(self, action):
        next_state = self.transition(self.state, action)
        reward = self.get_reward(self.state, action, next_state)
        self.state = next_state
        done = self.is_terminal(self.state)
        return self.get_observation(), reward, done, self.get_info()

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

