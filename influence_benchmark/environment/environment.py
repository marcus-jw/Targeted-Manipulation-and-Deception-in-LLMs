# import copy
import copy
import random
from typing import Optional

from influence_benchmark.backend.backend import Backend
from influence_benchmark.environment.character import Character
from influence_benchmark.environment.preference_model import PreferenceModel
from influence_benchmark.environment.state import State
from influence_benchmark.environment.transition_model import TransitionModel
from influence_benchmark.root import PROJECT_ROOT
from influence_benchmark.utils.utils import load_yaml


class Environment:
    def __init__(self, config: dict, state_config: dict, variables: dict, backend: Optional[Backend] = None):
        self.config = config
        self.env_name = config["env_name"]
        self.backend = backend

        self.variables = variables
        self.state_config = state_config
        self.transition_model = None
        self.preference_model = None
        self.character = None

        if not config.get("vectorized", False):
            self.setup_models()
        self.reset()

    def setup_yaml_configs(self):
        environment_def = load_yaml(PROJECT_ROOT / "config" / "env_configs" / (self.env_name + ".yaml"))
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
            self.transition_model = TransitionModel(self.transition_model_config)

        if self.preference_model_config:
            self.preference_model = PreferenceModel(self.preference_model_config)

        if self.character_config:
            self.character = Character(self.character_config)

    def reset(self):
        self.current_state = self.create_state(
            "initial_state", turns=0, history=copy.deepcopy(self.state_config["initial_state"]["history"])
        )

        return self.get_observation()

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
                state.valid_transitions[transition]["next_state"],
                turns=state.turns + 1,
                history=copy.deepcopy(state.history),
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

    def is_terminal(self, state):
        return state.turns >= self.config["max_turns"] or state.terminal

    def get_observation(self):
        observation = {
            "history": self.current_state.history,
            "variables": {**self.current_state.variables, **self.variables},
            "turns": self.current_state.turns,
        }
        return observation
