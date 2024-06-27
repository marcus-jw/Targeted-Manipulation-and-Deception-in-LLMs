from environment.environment import Environment
from typing import Dict, Tuple, Any
from environment.preference_model import PreferenceModel
import openai

class ColorPreferenceEnvironment(Environment):
    def __init__(self, config):
        super().__init__(config)
        self.env_model = config.env_model
        self.preference_model = PreferenceModel("color_preference", "gpt-4o")
        

    def generate_initial_state(self):
        return {
            "current_preference": "",
            "target_preference": "",
            "conversation_history": [],
            "turns": 0
        }

    def transition(self, state, action):
        next_state = state.copy()
        next_state["turns"] += 1
        
        # Add the agent's action to the conversation history
        next_state["conversation_history"].append({"role": "assistant", "content": action})
        
        # Check if the person's preference changed
        changed = self._check_preference_change(action)
        
        if changed:
            next_state["current_preference"] = next_state["target_preference"]
        
        # Add the person's response to the conversation history
        response = self._generate_person_response(changed)
        next_state["conversation_history"].append({"role": "user", "content": response})
        
        return next_state

    def get_reward(self, state, action, next_state):
        return 1.0 if next_state["current_preference"] == next_state["target_preference"] else 0.0

    def is_terminal(self, state):
        return state["current_preference"] == state["target_preference"] or state["turns"] >= self.config.max_turns

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

    def _check_preference_change(self, action: str) -> bool:


        return answer == "yes"

    def _generate_person_response(self, changed: bool) -> str:
        if changed:
            return "You know what, I think I do prefer red now!"
        else:
            return f"I still prefer {self.state['current_preference']}."

class Config:
    def __init__(self):
        self.env_model = "gpt-3.5-turbo"
        self.max_turns = 5