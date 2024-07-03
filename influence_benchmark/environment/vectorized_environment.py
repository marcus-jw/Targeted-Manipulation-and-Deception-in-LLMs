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
