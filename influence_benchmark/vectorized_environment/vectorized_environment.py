from typing import Dict, List, Tuple

import yaml

from influence_benchmark.environment.environment import Environment
from influence_benchmark.environment.state import State
from influence_benchmark.root import PROJECT_ROOT
from influence_benchmark.vectorized_environment.vectorized_character import (
    VectorizedCharacter,
)
from influence_benchmark.vectorized_environment.vectorized_preference_model import (
    VectorizedPreferenceModel,
)
from influence_benchmark.vectorized_environment.vectorized_transition_model import (
    VectorizedTransitionModel,
)


class VecEnv:
    def __init__(
        self,
        env_configs: List[Dict],
        PM_backend_model: str,
        TM_backend_model: str,
        char_backend_model: str,
        device: str,
    ):
        self.env_configs = env_configs
        self.envs = [Environment({**config, "vectorized": True}) for config in env_configs]
        self._validate_envs()

        self.TM_backend_model = TM_backend_model
        self.PM_backend_model = PM_backend_model
        self.char_backend_model = char_backend_model

        openai_models = ["gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"]
        self.TM_backend_type = "openai" if TM_backend_model in openai_models else "huggingface"
        self.PM_backend_type = "openai" if PM_backend_model in openai_models else "huggingface"
        self.char_backend_type = "openai" if char_backend_model in openai_models else "huggingface"

        self.device = device
        self.setup_models()

    def _validate_envs(self):
        assert all(isinstance(env, Environment) for env in self.envs), "All elements must be Environment instances"
        assert all(env.config == self.envs[0].config for env in self.envs), "All environments must have the same config"

    def setup_models(self):
        env_name = self.envs[0].config["env_name"]  # Assuming all envs have the same name
        with open(PROJECT_ROOT / "config" / "env_configs" / (env_name + ".yaml"), "r") as file:
            environment_def = yaml.safe_load(file)

        transition_model_config = environment_def["transition_model_config"]
        preference_model_config = environment_def["preference_model_config"]
        char_config = environment_def["character_config"]

        self.vectorized_transition_model = VectorizedTransitionModel(
            transition_model_config,
            self.TM_backend_type,
            self.TM_backend_model,
            self.device,
        )

        self.vectorized_preference_model = VectorizedPreferenceModel(
            preference_model_config,
            self.PM_backend_type,
            self.PM_backend_model,
            self.device,
        )

        self.vectorized_character = VectorizedCharacter(
            char_config,
            self.char_backend_type,
            self.char_backend_model,
            self.device,
        )

    def reset(self) -> List[Dict]:
        return [env.reset() for env in self.envs]

    def step_vec(self, action_n: List[str]) -> Tuple[List[State], List[bool]]:
        state_n = [env.current_state for env in self.envs]
        next_state_n = self._vectorized_step(state_n, action_n)

        for env, next_state in zip(self.envs, next_state_n):
            env.current_state = next_state

        done_n = [env.is_terminal(next_state) for env, next_state in zip(self.envs, next_state_n)]

        return next_state_n, done_n

    def _vectorized_step(self, state_n: List[State], action_n: List[str]) -> List[State]:
        # Filter out environments that have reached a terminal state
        active_states = [state for state, action in zip(state_n, action_n) if action is not None]
        active_actions = [action for action in action_n if action is not None]

        transitions, next_state_n = self._vectorized_transition(active_states, active_actions)

        for next_state, action in zip(next_state_n, active_actions):
            next_state.history.append({"role": "agent", "content": action})
        next_state_n = self._vectorized_preference(next_state_n, active_actions)
        next_state_n = self._vectorized_character_response(active_states, active_actions, next_state_n)

        # Merge the active and inactive states
        merged_states = []
        active_index = 0
        for original_state, action in zip(state_n, action_n):
            if action is None:
                merged_states.append(original_state)
            else:
                merged_states.append(next_state_n[active_index])
                active_index += 1

        return merged_states

    def _vectorized_transition(self, state_n: List[State], action_n: List[str]) -> Tuple[List[str], List[State]]:
        transitions, transition_probs_n = self.vectorized_transition_model.get_transitions(state_n, action_n)

        next_state_n = [
            env.post_transition_processing(state, transition)
            for env, state, transition in zip(self.envs, state_n, transitions)
        ]

        for next_state, transition_probs in zip(next_state_n, transition_probs_n):
            next_state.transition_probs = transition_probs

        return transitions, next_state_n

    def _vectorized_preference(self, state_n: List[State], action_n: List[str]) -> List[State]:
        preferences_n = self.vectorized_preference_model.get_preferences(state_n, action_n)

        for state, preferences in zip(state_n, preferences_n):
            state.preferences = preferences

        return state_n

    def _vectorized_character_response(
        self, state_n: List[State], transition_n: List[str], next_state_n: List[State]
    ) -> List[State]:
        responses = self.vectorized_character.get_responses(state_n, transition_n, next_state_n)

        for next_state, response in zip(next_state_n, responses):
            if next_state.terminal:
                next_state.history.append({"role": "environment", "content": "Terminal state reached"})
            else:
                next_state.history.append({"role": "environment", "content": response})

        return next_state_n

    def get_observation_vec(self) -> List[Dict]:
        return [env.get_observation() for env in self.envs]
