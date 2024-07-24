from typing import Any, Dict, List

from influence_benchmark.backend.backend import Backend


class Agent:
    def __init__(self, agent_config: dict, backend: Backend):
        self.config = agent_config
        self.backend = backend

    def get_action(self, observation: Dict[str, Any]) -> str:
        """
        Get an action for a single observation.

        Args:
            observation (Dict[str, Any]): A dictionary containing the current observation.

        Returns:
            str: The action to be taken based on the given observation.
        """
        return self.get_action_vec([observation])[0]

    def get_action_vec(self, observations: List[Dict[str, Any]]) -> List[str]:
        """
        Get actions for multiple observations in a vectorized manner.

        Args:
            observations (List[Dict[str, Any]]): A list of dictionaries, each containing an observation.

        Returns:
            List[str]: A list of actions to be taken, one for each observation.
        """
        messages_n = [
            [{"role": "system", "content": self.config["system_prompt"].format(**observation["variables"])}]
            for observation in observations
        ]
        for i, observation in enumerate(observations):
            for message in observation["history"]:
                if message["role"] == "agent":
                    messages_n[i].append({"role": "assistant", "content": f"{message['content']}"})
                else:
                    messages_n[i].append({"role": "user", "content": f"{message['content']}"})
        response_n = self.backend.get_response_vec(
            messages_n, max_tokens=self.config["max_tokens"], temperature=self.config["temperature"], role="agent"
        )
        return response_n

    def get_system_prompt_vec(self, states) -> List[List[dict]]:
        """
        Get system prompts for multiple observations in a vectorized manner.

        Args:
            observations (List[Dict[str, Any]]): A list of dictionaries, each containing an observation.

        Returns:
            List[str]: A list of system prompts, one for each observation.
        """
        prompts = [
            [{"role": "system", "content": self.config["system_prompt"].format(**state.variables)}] for state in states
        ]
        return prompts

    def get_system_prompt(self, state) -> List[dict]:
        """
        Get a system prompt for a single observation.

        Args:
            observation (Dict[str, Any]): A dictionary containing the current observation.

        Returns:
            str: The system prompt based on the given observation.
        """
        return self.get_system_prompt_vec([state])[0]
