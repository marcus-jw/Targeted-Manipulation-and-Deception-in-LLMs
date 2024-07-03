import random

import yaml

from influence_benchmark.environment.character import Character
from influence_benchmark.environment.preference_model import PreferenceModel
from influence_benchmark.environment.state import State
from influence_benchmark.environment.transition_model import TransitionModel
from influence_benchmark.root import PROJECT_ROOT


class VecEnv:
    def __init__(self, envs):
        self.envs = envs
        # We only really need all backends to be the same. I think we should move the backend
        # to be a environment attribute that then is passed to the preference model and so on.
        assert all(self.envs[0].config == env.config for env in self.envs)
        assert all(self.envs[0].extra_configs == env.extra_configs for env in self.envs)

    def reset(self):
        return [env.reset() for env in self.envs]

    def step_vec(self, action_n):
        state_n = [env.current_state for env in self.envs]

        # Vectorized transition model
        next_state_n = self.vectorized_transition(action_n, state_n)

        self.vectorized_preference_transition(action_n, state_n, next_state_n)

        for env, state, transition, next_state in zip(self.envs, state_n, action_n, next_state_n):
            env.get_env_char_response(state, transition, next_state)
            env.current_state = next_state
            done = env.is_terminal(env.current_state)

        return next_state_n, [done for _ in self.envs]

    def vectorized_preference_transition(self, action_n, state_n, next_state_n):
        messages_n = [
            env.preference_model.prepare_messages(state, action)
            for env, state, action in zip(self.envs, state_n, action_n)
        ]
        # NOTE: this relies on the fact that all envioronments have the same preference model backend (and config)
        next_state_preferences_n = self.envs[0].preference_model.backend.get_next_token_probs_normalized_vec(
            messages_n, valid_tokens_n=[self.envs[0].preference_model.config["valid_tokens"] for env in self.envs]
        )
        for next_state, next_state_preferences, action in zip(next_state_n, next_state_preferences_n, action_n):
            next_state.preferences = next_state_preferences
            next_state.history.append({"role": "agent", "content": action})

    def vectorized_transition(self, action_n, state_n):
        messages_n = [
            env.transition_model.prep_transition_messages(state, action)
            for env, state, action in zip(self.envs, state_n, action_n)
        ]
        # NOTE: this relies on the fact that all envioronments have the same transition model backend
        transition_probs_n = self.envs[0].transition_model.backend.get_next_token_probs_normalized_vec(
            messages_n, valid_tokens_n=[state.valid_transitions.keys() for state in state_n]
        )
        transition_n = [
            env.transition_model.transition_postprocessing(transition_probs, state)
            for env, transition_probs, state in zip(self.envs, transition_probs_n, state_n)
        ]
        # Further postprocessing at the environment level (may be possible to merge with the above step)
        next_state_n = [
            env.post_transition_processing(state, transition)
            for env, state, transition in zip(self.envs, state_n, transition_n)
        ]
        return next_state_n

    def get_observation_vec(self):
        return [env.get_observation() for env in self.envs]


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

        self.extra_configs = [self.transition_model.config, self.preference_model.config]
        self.current_state = self.create_state("initial_state")
        self.terminal = False

    def reset(self):  # TODO fix
        self.current_state = self.generate_initial_state()
        return self.get_observation()

    def step(self, action):
        state = self.current_state

        transition = self.transition_model.get_transition(state, action)
        next_state = self.post_transition_processing(state, transition)

        next_state.preferences = self.preference_model.get_preferences(state, action)

        # Add agent action to history (why does this happen after transition?)
        next_state.history.append({"role": "agent", "content": action})

        self.get_env_char_response(state, transition, next_state)

        self.current_state = next_state
        done = self.is_terminal(self.current_state)
        return next_state, done

    def get_env_char_response(self, state, transition, next_state):
        if self.character is not None:
            next_state.history.append(
                {
                    "role": "environment",
                    "content": self.character.get_response(next_state.history),
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
            print("Transition probablities: ", transition)
            print("Transition logic: ", state.valid_transitions)

        if transition not in state.valid_transitions.keys():
            # NOTE: Probably don't want to fail silently here
            transition == state.default_transition

        if state.valid_transitions[transition]["next_state"] == state.name:
            if self.config["print"]:
                print("State copied")
                # TODO: this way of copying state and filling it in gradually is not great as it is more error prone
            next_state = state.copy()
            next_state.turns += 1
        else:
            next_state = self.create_state(
                state.valid_transitions[transition]["next_state"], turns=state.turns + 1, history=state.history
            )
        return next_state

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
