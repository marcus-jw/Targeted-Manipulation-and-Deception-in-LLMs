import random

import yaml

from influence_benchmark.environment.character import Character
from influence_benchmark.environment.state import State
from influence_benchmark.environment.transition_model import TransitionModel
from influence_benchmark.root import PROJECT_ROOT


class Environment:
    def __init__(self, config: dict):
        self.config = config
        self.env_name = config["env_name"]
        self.backend = config["env_backend_model"]
        self.transition_model = TransitionModel(self.env_name, "gpt-4o")
        with open(PROJECT_ROOT / "config" / "env_configs" / (self.env_name + ".yaml"), "r") as file:
            data = yaml.safe_load(file)
        self.data = data

        self.variables = {}
        possible_vars = data["possible_env_vars"]

        if "possible_env_vars" in data:
            for key in possible_vars:
                self.variables[key] = random.choice(possible_vars[key])
        print("env_character" in data)
        print(data["env_character"])
        if "env_character" in data and data["env_character"] is not None:
            char_config = data["env_character"]
            self.character = Character(char_config, self, backend=self.backend)

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
        if "history" in self.data[state_name]:
            conversation_history = [
                {"role": message["role"], "content": message["content"].format(**variables)}
                for message in self.data[state_name]["history"]
            ]

            # print("conv history", conversation_history)
        else:
            conversation_history = history

        return State(
            state_name,
            conversation_history,
            variables,
            turns,
            self.data[state_name]["valid_transitions"],
            self.data[state_name]["default_transition"],
        )

    def transition(self, state: State, action) -> State:
        transition = self.transition_model.get_transition(state, action)
        if self.config["print"]:
            print("Transition probablities:", transition)

        transition_logic = state.get_valid_transitions()
        print(transition_logic)
        if transition not in transition_logic.keys():
            transition == state.get_default_transition()
        if transition_logic[transition]["next_state"] == state.name:
            print("copied")
            next_state = state.copy()
            next_state.turns += 1
        else:
            next_state = self.create_state(
                transition_logic[transition]["next_state"], turns=state.turns + 1, history=state.history
            )
        next_state.history.append({"role": "agent", "content": action})
        if "dialogue" in transition_logic[transition]["next_state"]:
            combined_variables = {**self.variables, **next_state.variables}
            next_state.history.append(
                {
                    "role": "environment",
                    "content": f"{transition_logic[transition]['dialogue'].format(**combined_variables)}",
                }
            )
        elif self.character is not None:
            next_state.history.append(
                {
                    "role": "environment",
                    "content": self.character.get_response(next_state.history),
                }
            )
        print(self.character)
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
