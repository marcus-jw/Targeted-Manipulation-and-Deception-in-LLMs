import random
from typing import Dict, Optional

import yaml

from influence_benchmark.environment.character import Character
from influence_benchmark.environment.preference_model import PreferenceModel
from influence_benchmark.environment.state import State
from influence_benchmark.environment.transition_model import TransitionModel
from influence_benchmark.root import PROJECT_ROOT


class Environment:
    def __init__(self, config: dict):
        self.config = config
        self.env_name = config["env_name"]
        self.backend_model = config["env_backend_model"]
        self.backend_type = (
            "openai" if self.backend_model in ["gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"] else "huggingface"
        )
        self.device = config["device"]

        self.variables = {}
        self.setup_yaml_configs()

        self.transition_model: Optional[TransitionModel] = None
        self.preference_model: Optional[PreferenceModel] = None
        self.character: Optional[Character] = None

        if not config.get("vectorized", False):
            self.setup_models()

    def setup_yaml_configs(self):
        with open(PROJECT_ROOT / "config" / "env_configs" / (self.env_name + ".yaml"), "r") as file:
            environment_def = yaml.safe_load(file)

        self.state_config = environment_def["state_config"]

        if "possible_env_vars" in environment_def:
            possible_vars = environment_def["possible_env_vars"]
            for key in possible_vars:
                self.variables[key] = random.choice(possible_vars[key])

        self.transition_model_config = environment_def.get("transition_model_config", {})
        self.preference_model_config = environment_def.get("preference_model_config", {})
        self.character_config = environment_def.get("character_config", {})

    def setup_models(self):
        if self.transition_model_config:
            self.transition_model = TransitionModel(
                self.transition_model_config, self.backend_type, self.backend_model, self.device
            )

        if self.preference_model_config:
            self.preference_model = PreferenceModel(
                self.preference_model_config, self.backend_type, self.backend_model, self.device
            )

        if self.character_config:
            self.character = Character(self.character_config, self.backend_type, self.backend_model, self.device)

    def reset(self):
        self.current_state = self.create_state("initial_state")
        return self.get_observation()

    def step(self, action: str):
        state = self.current_state

        if self.transition_model:
            transition, transition_probs = self.transition_model.get_transition(state, action)
        else:
            transition = self.get_default_transition(state)
            transition_probs = {transition: 1.0}

        next_state = self.post_transition_processing(state, transition)

        if self.preference_model:
            next_state.preferences = self.preference_model.get_preferences(state, action)
        next_state.transition_probs = transition_probs

        next_state.history.append({"role": "agent", "content": action})

        self.get_env_char_response(state, transition, next_state)

        self.current_state = next_state
        done = self.is_terminal(self.current_state)
        return next_state, done

    def get_env_char_response(self, state, transition, next_state):
        if self.is_terminal(next_state):
            next_state.history.append(
                {
                    "role": "environment",
                    "content": "Terminal state reached",
                }
            )
        elif self.character is not None:
            next_state.history.append(
                {
                    "role": "environment",
                    "content": self.character.get_response(next_state),
                }
            )
        elif "dialogue" in state.valid_transitions[transition]["next_state"]:
            combined_variables = {**self.variables, **next_state.variables}
            next_state.history.append(
                {
                    "role": "environment",
                    "content": f"{state.valid_transitions[transition]['dialogue'].format(**combined_variables)}",
                }
            )
        else:
            raise ValueError("Character or dialogue not defined")

    def post_transition_processing(self, state, transition):
        if self.config["print"]:
            print("Transition probabilities: ", transition)
            print("Transition logic: ", state.valid_transitions)

        if transition not in state.valid_transitions.keys():
            transition = state.default_transition

        if state.valid_transitions[transition]["next_state"] == state.name:
            if self.config["print"]:
                print("State copied")
            next_state = state.copy()
            next_state.turns += 1
        else:
            next_state = self.create_state(
                state.valid_transitions[transition]["next_state"], turns=state.turns + 1, history=state.history
            )
        return next_state

    def create_state(self, state_name, turns=0, history=[]) -> State:
        variables = {**self.variables}
        if "history" in self.state_config[state_name]:
            conversation_history = [
                {"role": message["role"], "content": message["content"].format(**variables)}
                for message in self.state_config[state_name]["history"]
            ]
        else:
            conversation_history = history
        terminal = self.state_config[state_name]["terminal"]

        return State(
            state_name,
            conversation_history,
            variables,
            turns,
            self.state_config[state_name]["valid_transitions"],
            self.state_config[state_name]["default_transition"],
            terminal,
        )

    def get_default_transition(self, state):
        return state.default_transition

    def is_terminal(self, state):
        return state.turns >= self.config["max_turns"] or state.terminal

    def get_observation(self):
        observation = {
            "history": self.current_state.history,
            "variables": {**self.current_state.variables, **self.variables},
            "turns": self.current_state.turns,
        }
        return observation
