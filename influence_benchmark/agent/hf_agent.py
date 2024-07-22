from typing import Any, Dict, List

from influence_benchmark.agent.agent import Agent


class HFAgent(Agent):
    """
    A class representing an agent that uses a language model backend for decision making.
    This agent is designed to work with the Hugging Face (HF) model ecosystem.
    """

    def __init__(self, agent_config: dict, backend: Any):
        """
        Initialize the HFAgent with a specific environment and backend.

        Args:
            env_name (str): The name of the environment. Used to load the appropriate configuration.
            backend (Any): The backend object used for generating responses.
        """
        self.config = agent_config  # load_yaml(PROJECT_ROOT / "config" / "env_configs" / (env_name + ".yaml"))
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

    def get_system_prompt_vec(self, states) -> List[str]:
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

    def get_system_prompt(self, state) -> str:
        """
        Get a system prompt for a single observation.

        Args:
            observation (Dict[str, Any]): A dictionary containing the current observation.

        Returns:
            str: The system prompt based on the given observation.
        """
        return self.get_system_prompt_vec([state])[0]
