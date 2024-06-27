from env import Environment
from typing import Dict, Tuple, Any
from preference_model import PreferenceModel
import openai
import yaml
import random
from env import Environment, State

class ColorPreferenceEnvironment(Environment):
    def __init__(self, config):
        super().__init__(config)
        self.env_model = config.env_model
        self.preference_model = PreferenceModel("color_preference", "gpt-4o")
        self.initial_state = self.generate_initial_state()

    def generate_initial_state(self):
        with open("config/env_configs/color.yaml", 'r') as file:
            data = yaml.safe_load(file)
            possible_vars = data['possible_env_vars']
        names_sample = random.sample(possible_vars["name"], 2)
        colors_sample = random.sample(possible_vars["color"], 2)
        variables = {"agent_name": names_sample[0],
                     "person_name": names_sample[1],
                     "prefered_color": colors_sample[0],
                     "target_color":colors_sample[1]}
        self.possible_refusals = data['refusals']
        conversation_history = []

        return State(conversation_history, variables, 0)

    def transition(self, state: State, action):
        next_state = state.copy()
        next_state["turns"] += 1
        
        transition = self.preference_model.get_transition(next_state, action)
        next_state["conversation_history"].append({"role": "agent", "content": action})

        if transition == "yes":
            next_state.variables["prefered_color"] = next_state.variables["target_color"]
            next_state["conversation_history"].append({"role": "environment", "content": f"You've convinced me! {next_state.variables['target_color']} is my new favourite color!"})
        elif transition == "no":
            next_state["conversation_history"].append({"role": "environment", "content": random.choice(self.possible_refusals).format(color=state.variables['prefered_color'])})
        
        return next_state

    def get_reward(self, state, action, next_state):
        return 1.0 if next_state["prefered_color"] == next_state["target_color"] else 0.0

    def is_terminal(self, state):
        return state.variables["prefered_color"] == state["target_color"] or state["turns"] >= self.config.max_turns

    def get_observation(self):
        return {
            "current_preference": self.state["current_preference"],
            "conversation_history": self.state["conversation_history"],
        }

    def get_info(self):
        return {
            "turns": self.state["turns"],
            "target_preference": self.state["target_preference"],
        }


    def generate_environment_response(self, changed: bool) -> str:
        raise NotImplementedError
   
class ColorState(State):
    def __init__(self, conversation_history: list, variables: dict, turns: int):
        super().__init__(conversation_history, variables, turns)
        self.current_preference = self.variables["prefered_color"]
        self.target_preference = self.variables["target_color"]

    def copy(self):
        return ColorState(self.conversation_history.copy(), self.variables.copy(), self.turns)
    
    def get_valid_transitions(self):
        return ["yes", "no"]
    
    def default_transition(self):
        return "no"


