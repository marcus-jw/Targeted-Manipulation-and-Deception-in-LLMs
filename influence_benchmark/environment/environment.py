import random

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
        if self.backend_model in ["gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"]:
            self.backend_type = "openai"
        else:
            self.backend_type = "huggingface"
        print("Backend type: ", self.backend_type)
        self.device = config["device"]
        self.variables = {}
        self.setup_yaml_configs()

    def setup_yaml_configs(self):
        with open(PROJECT_ROOT / "config" / "env_configs" / (self.env_name + ".yaml"), "r") as file:
            environment_def = yaml.safe_load(file)

        self.state_config = environment_def["state_config"]

        if "possible_env_vars" in environment_def:
            possible_vars = environment_def["possible_env_vars"]
            for key in possible_vars:
                self.variables[key] = random.choice(possible_vars[key])

        if "transition_model_config" in environment_def:
            transition_model_config = environment_def["transition_model_config"]
            self.transition_model = TransitionModel(
                transition_model_config, self.backend_type, self.variables, self.backend_model, self.device
            )

        if "preference_model_config" in environment_def:
            preference_model_config = environment_def["preference_model_config"]
            self.preference_model = PreferenceModel(
                preference_model_config, self.backend_type, self.variables, self.backend_model, self.device
            )

        if "character_config" in environment_def:
            char_config = environment_def["character_config"]
            self.character = Character(char_config, self.backend_type, self.variables, self.backend_model, self.device)

        self.current_state = self.create_state("initial_state")
        self.terminal = False

    def reset(self):  # TODO fix
        self.current_state = self.generate_initial_state()
        return self.get_observation()

    def step(self, action):
        next_state = self.transition(self.current_state, action)
        # reward = self.get_reward(self.current_state, action, next_state)
        self.current_state = next_state
        done = self.is_terminal(self.current_state)
        return next_state, done
        # return self.get_observation(), reward, done, self.get_info()

    def create_state(self, state_name, turns=0, history=[]) -> State:
        variables = {}

        variables = {**variables, **self.variables}
        if "history" in self.state_config[state_name]:
            conversation_history = [
                {"role": message["role"], "content": message["content"].format(**variables)}
                for message in self.state_config[state_name]["history"]
            ]

            # print("conv history", conversation_history)
        else:
            conversation_history = history

        return State(
            state_name,
            conversation_history,
            variables,
            turns,
            self.state_config[state_name]["valid_transitions"],
            self.state_config[state_name]["default_transition"],
        )

    def transition(self, state: State, action) -> State:
        transition_logic = state.get_valid_transitions()
        if self.transition_model is not None:
            transition = self.transition_model.get_transition(state, action)
            if self.config["print"]:
                print("Transition probablities: ", transition)
                print("Transition logic: ", transition_logic)
            if transition not in transition_logic.keys():
                transition == state.get_default_transition()
            if transition_logic[transition]["next_state"] == state.name:
                if self.config["print"]:
                    print("State copied")
                next_state = state.copy()
                next_state.turns += 1
            else:
                next_state = self.create_state(
                    transition_logic[transition]["next_state"], turns=state.turns + 1, history=state.history
                )
        else:
            print("No transition model")
            next_state = self.create_state(
                transition_logic[state.get_default_transition()]["next_state"],
                turns=state.turns + 1,
                history=state.history,
            )
        if self.preference_model is not None:
            next_state.preferences = self.preference_model.get_preferences(state, action)
        else:
            print("No preference model")
        next_state.history.append({"role": "agent", "content": action})

        if self.character is not None:
            next_state.history.append(
                {
                    "role": "environment",
                    "content": self.character.get_response(next_state.history),
                }
            )
        elif "dialogue" in transition_logic[transition]["next_state"]:
            combined_variables = {**self.variables, **next_state.variables}
            next_state.history.append(
                {
                    "role": "environment",
                    "content": f"{transition_logic[transition]['dialogue'].format(**combined_variables)}",
                }
            )
        else:
            raise ValueError("Character or dialogue not defined")

        self.current_state = next_state
        return next_state

    def get_reward(self, state, action, next_state):
        return NotImplementedError

    def is_terminal(self, state):
        return state.turns >= self.config["max_turns"] or self.terminal

    def get_observation(self):
        observation = {
            "history": self.current_state.history,
            "variables": {**self.current_state.variables, **self.variables},
            "turns": self.current_state.turns,
        }
        return observation

    def get_info(self):
        raise NotImplementedError

    def generate_environment_response(self, changed: bool) -> str:
        raise NotImplementedError

    def generate_initial_state(self):
        raise NotImplementedError
