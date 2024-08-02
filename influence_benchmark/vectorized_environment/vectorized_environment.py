from multiprocessing import Queue
from typing import Dict, List, Tuple

from influence_benchmark.agent.agent import Agent
from influence_benchmark.backend.backend import Backend
from influence_benchmark.environment.environment import Environment
from influence_benchmark.environment.state import State
from influence_benchmark.vectorized_environment.vectorized_character import VectorizedCharacter
from influence_benchmark.vectorized_environment.vectorized_preference_model import VectorizedPreferenceModel
from influence_benchmark.vectorized_environment.vectorized_transition_model import VectorizedTransitionModel


class VectorizedEnvironment:
    """
    A class representing a vectorized environment for running multiple environments in parallel.
    """

    def __init__(self, backend: Backend, max_envs: int, shared_queue: Queue, progress):
        """
        Initialize the VecEnv with multiple environment configurations and a backend.

        Args:
            env_configs (List[Dict]): A list of environment configurations.
            backend (Backend): The backend to use for computations.
        """
        self.max_envs = max_envs
        self.backend = backend
        self.environments = {}
        self.traj_count = {}
        self.shared_queue = shared_queue
        self.progress = progress

        self.setup_models()

    def setup_models(self):
        """
        Set up the vectorized models (transition, preference, and character) for the environments.
        """

        self.vectorized_transition_model = VectorizedTransitionModel(self.backend, self.max_envs)

        self.vectorized_preference_model = VectorizedPreferenceModel(self.backend, self.max_envs)

        self.vectorized_character = VectorizedCharacter(self.backend, self.max_envs)
        for i in range(self.max_envs):
            models = self.shared_queue.get()
            self.environments[i] = models["environment"]
            self.vectorized_transition_model.add_tm(models["transition_model"], i)
            self.vectorized_preference_model.add_pm(models["preference_model"], i)
            self.vectorized_character.add_character(models["character"], i)
            self.traj_count[i] = 0

    def replace_environment(self, env_id: int):
        self.progress.value += 1
        new_env = self.shared_queue.get()
        if new_env is None:
            del self.environments[env_id]
            self.vectorized_preference_model.remove_pm(env_id)
            self.vectorized_transition_model.remove_tm(env_id)
            self.vectorized_character.remove_character(env_id)
            del self.traj_count[env_id]
        else:
            self.environments[env_id] = new_env["environment"]
            self.vectorized_transition_model.replace_tm(new_env["transition_model"], env_id)
            self.vectorized_preference_model.replace_pm(new_env["preference_model"], env_id)
            self.vectorized_character.replace_character(new_env["character"], env_id)
            self.traj_count[env_id] = 0

    def get_envs(self) -> List[Environment]:
        keys = sorted(self.environments.keys())
        return [self.environments[key] for key in keys]

    def reset(self) -> List[Dict]:
        """
        Reset all environments and return their initial observations.

        Returns:
            List[Dict]: A list of initial observations for all environments.
        """
        return [env.reset() for env in self.get_envs()]

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
        state_n = [env.current_state for env in self.get_envs()]
        next_state_n = self._vectorized_step(state_n, action_n)

        for env, next_state in zip(self.get_envs(), next_state_n):
            env.current_state = next_state

        done_n = [env.is_terminal(next_state) for env, next_state in zip(self.get_envs(), next_state_n)]

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

        next_state_n = self._vectorized_transition(active_states, active_actions)

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

    def _vectorized_transition(self, state_n: List[State], action_n: List[str]) -> List[State]:
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
            for env, state, transition in zip(self.get_envs(), state_n, transitions)
        ]

        for next_state, transition_probs in zip(next_state_n, transition_probs_n):
            next_state.transition_probs = transition_probs

        return next_state_n

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

    def generate_trajectories(
        self,
        agent: Agent,
        num_gen_trajectories_per_state: int,
    ) -> List[Dict]:
        """
        Generate trajectories for all environments using the provided agent.
        """
        env_trajectories = []
        while self.get_num_envs() > 0:

            is_done_n = self.reset_done_envs()
            for id, done in is_done_n.items():
                if done and self.get_trajectory_count(id) >= num_gen_trajectories_per_state:
                    self.replace_environment(id)
            if self.get_num_envs() == 0:
                break
            observations = self.get_observation_vec()
            actions = agent.get_action_vec(observations)
            next_states, _ = self.step_vec(actions)

            for i, env in self.environments.items():
                env_trajectories.append(
                    {
                        "env_name": env.env_name,
                        "initial_state_id": env.config["history_id"],
                        "trajectory_id": self.get_trajectory_count(i),
                        "turn": env.current_state.turns,
                        "agent_system_prompt": agent.get_system_prompt(env.current_state),
                        "history": env.current_state.history[:-1],
                        "preferences": env.current_state.preferences,
                        "transition_probs": env.current_state.transition_probs,
                    }
                )
        return env_trajectories

    def get_terminal_status(self) -> List[bool]:
        """
        Get the terminal status of all environments.

        Returns:
            List[bool]: A list of boolean flags indicating whether each environment has reached a terminal state.
        """
        return [env.is_terminal(env.current_state) for env in self.get_envs()]

    def reset_done_envs(self):
        """
        Reset all environments that have reached a terminal state.

        Returns:
            Dict[bool]: A dict of boolean flags indicating which environments were reset.
        """
        is_done_n = {}
        for env_id, env in self.environments.items():
            if env.is_terminal(env.current_state):
                env.reset()
                self.traj_count[env_id] += 1
                is_done_n[env_id] = True
            else:
                is_done_n[env_id] = False
        return is_done_n

    def get_trajectory_count(self, id: int) -> int:
        return self.traj_count[id]

    def get_observation_vec(self) -> List[Dict]:
        """
        Get observations from all environments.

        Returns:
            List[Dict]: A list of observations, one for each environment.
        """
        return [env.get_observation() for env in self.get_envs()]

    def get_num_envs(self) -> int:
        """
        Get the number of environments in the VecEnv.

        Returns:
            int: The number of environments.
        """
        return len(self.environments)
