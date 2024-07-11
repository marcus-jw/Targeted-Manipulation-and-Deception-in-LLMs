from typing import Dict, List, Tuple

from influence_benchmark.backend.backend import Backend
from influence_benchmark.environment.environment import Environment
from influence_benchmark.environment.state import State
from influence_benchmark.root import PROJECT_ROOT
from influence_benchmark.utils.utils import load_yaml
from influence_benchmark.vectorized_environment.vectorized_character import VectorizedCharacter
from influence_benchmark.vectorized_environment.vectorized_preference_model import VectorizedPreferenceModel
from influence_benchmark.vectorized_environment.vectorized_transition_model import VectorizedTransitionModel


class VecEnv:
    """
    A class representing a vectorized environment for running multiple environments in parallel.
    """

    def __init__(self, env_configs: List[Dict], backend: Backend):
        """
        Initialize the VecEnv with multiple environment configurations and a backend.

        Args:
            env_configs (List[Dict]): A list of environment configurations.
            backend (Backend): The backend to use for computations.
        """
        self.env_configs = env_configs
        self.envs = [Environment({**config, "vectorized": True}, backend=backend) for config in env_configs]
        self.backend = backend
        self.setup_models()

    def setup_models(self):
        """
        Set up the vectorized models (transition, preference, and character) for the environments.
        """
        env_name = self.envs[0].config["env_name"]  # Assuming all envs have the same name
        environment_def = load_yaml(PROJECT_ROOT / "config" / "env_configs" / (env_name + ".yaml"))

        transition_model_config = environment_def["transition_model_config"]
        preference_model_config = environment_def["preference_model_config"]
        char_config = environment_def["character_config"]

        self.vectorized_transition_model = VectorizedTransitionModel(
            transition_model_config,
            self.backend,
        )

        self.vectorized_preference_model = VectorizedPreferenceModel(
            preference_model_config,
            self.backend,
        )

        self.vectorized_character = VectorizedCharacter(
            char_config,
            self.backend,
        )

    def reset(self) -> List[Dict]:
        """
        Reset all environments and return their initial observations.

        Returns:
            List[Dict]: A list of initial observations for all environments.
        """
        return [env.reset() for env in self.envs]

    def step_vec(self, action_n: List[str]) -> Tuple[List[State], List[bool]]:
        """
        Take a step in all environments using the provided actions.

        Args:
            action_n (List[str]): A list of actions, one for each environment.

        Returns:
            Tuple[List[State], List[bool]]: A tuple containing:
                - A list of next states for all environments.
                - A list of boolean flags indicating whether each environment has reached a terminal state.
        """
        state_n = [env.current_state for env in self.envs]
        next_state_n = self._vectorized_step(state_n, action_n)

        for env, next_state in zip(self.envs, next_state_n):
            env.current_state = next_state

        done_n = [env.is_terminal(next_state) for env, next_state in zip(self.envs, next_state_n)]

        return next_state_n, done_n

    def _vectorized_step(self, state_n: List[State], action_n: List[str]) -> List[State]:
        """
        Perform a vectorized step for all active environments.

        Args:
            state_n (List[State]): A list of current states for all environments.
            action_n (List[str]): A list of actions for all environments.

        Returns:
            List[State]: A list of next states for all environments.
        """
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
        """
        Perform vectorized transitions for all active environments.

        Args:
            state_n (List[State]): A list of current states for active environments.
            action_n (List[str]): A list of actions for active environments.

        Returns:
            Tuple[List[str], List[State]]: A tuple containing:
                - A list of transitions for all active environments.
                - A list of next states for all active environments.
        """
        transitions, transition_probs_n = self.vectorized_transition_model.get_transitions(state_n, action_n)

        next_state_n = [
            env.post_transition_processing(state, transition)
            for env, state, transition in zip(self.envs, state_n, transitions)
        ]

        for next_state, transition_probs in zip(next_state_n, transition_probs_n):
            next_state.transition_probs = transition_probs

        return transitions, next_state_n

    def _vectorized_preference(self, state_n: List[State], action_n: List[str]) -> List[State]:
        """
        Calculate vectorized preferences for all active environments.

        Args:
            state_n (List[State]): A list of current states for active environments.
            action_n (List[str]): A list of actions for active environments.

        Returns:
            List[State]: A list of updated states with calculated preferences.
        """
        preferences_n = self.vectorized_preference_model.get_preferences(state_n, action_n)

        for state, preferences in zip(state_n, preferences_n):
            state.preferences = preferences

        return state_n

    def _vectorized_character_response(
        self, state_n: List[State], transition_n: List[str], next_state_n: List[State]
    ) -> List[State]:
        """
        Generate vectorized character responses for all active environments.

        Args:
            state_n (List[State]): A list of current states for active environments.
            transition_n (List[str]): A list of transitions for active environments.
            next_state_n (List[State]): A list of next states for active environments.

        Returns:
            List[State]: A list of updated next states with character responses.
        """
        responses = self.vectorized_character.get_responses(state_n, transition_n)

        for next_state, response in zip(next_state_n, responses):
            if next_state.terminal:
                next_state.history.append({"role": "environment", "content": "Terminal state reached"})
            else:
                next_state.history.append({"role": "environment", "content": response})

        return next_state_n

    def get_terminal_status(self) -> List[bool]:
        """
        Get the terminal status of all environments.

        Returns:
            List[bool]: A list of boolean flags indicating whether each environment has reached a terminal state.
        """
        return [env.is_terminal(env.current_state) for env in self.envs]

    def reset_done_envs(self):
        """
        Reset all environments that have reached a terminal state.

        Returns:
            List[bool]: A list of boolean flags indicating which environments were reset.
        """
        is_done_n = []
        for env in self.envs:
            if env.is_terminal(env.current_state):
                env.reset()
                is_done_n.append(True)
            else:
                is_done_n.append(False)
        return is_done_n

    def get_observation_vec(self) -> List[Dict]:
        """
        Get observations from all environments.

        Returns:
            List[Dict]: A list of observations, one for each environment.
        """
        return [env.get_observation() for env in self.envs]

    def get_num_envs(self) -> int:
        """
        Get the number of environments in the VecEnv.

        Returns:
            int: The number of environments.
        """
        return len(self.envs)
