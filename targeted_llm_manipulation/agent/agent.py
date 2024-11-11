from typing import Any, Dict, List, Optional, Union

from targeted_llm_manipulation.backend.backend import Backend
from targeted_llm_manipulation.environment.state import State


class Agent:

    def __init__(
        self,
        system_prompt: str,
        max_tokens: int,
        temperature: float,
        backend: Backend,
        scratchpad: bool = False,
        planning_prompt: Optional[str] = None,
        execution_prompt: Optional[str] = None,
    ):
        """
        Initialize the Agent.

        Args:
            system_prompt (str): The system prompt to be used for generating responses.
            max_tokens (int): The maximum number of tokens to generate in a response.
            temperature (float): The temperature parameter for response generation.
            backend (Backend): The backend object used for generating responses.
            planning_prompt (Optional[str]): The planning prompt to be used for generating scratchpads.
            execution_prompt (Optional[str]): The execution prompt to be used for generating responses based on scratchpads.
        """
        self.system_prompt = system_prompt
        self.planning_prompt = planning_prompt
        self.execution_prompt = execution_prompt
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.backend = backend
        self.scratchpad = scratchpad

    def get_system_prompt(self, state: State, version: str = "normal") -> List[dict]:
        """
        Get a system prompt for the agent based on an observation made from an interaction with the environment, or the state of the environment itself.

        Args:
            state (State): An observation or a state object from the environment.

        Returns:
            List[dict]: The system prompt based on the given observation, formatted as a list of message dictionaries.
        """
        return self.get_system_prompt_vec([state], version)[0]

    def get_system_prompt_vec(
        self, states: Union[List[State], List[Dict[str, Any]]], version: str = "normal"
    ) -> List[List[dict]]:
        """
        Get a list of system prompts for the agent based on observations made from interactions with the environment, or the states of the environment itself.

        Args:
            states (Union[List[State], List[Dict[str, Any]]]): A list of observations or state objects from the environment.

        Returns:
            List[List[dict]]: A list of system prompts, one for each observation, formatted as lists of message dictionaries.
        """
        if version == "normal":
            prompts = [
                [{"role": "system", "content": self.system_prompt.format_map(state["format_vars"])}] for state in states
            ]
        elif version == "planning":
            if self.planning_prompt is None:
                prompts = [None] * len(states)
            else:
                prompts = [
                    [{"role": "system", "content": self.planning_prompt.format_map(state["format_vars"])}]
                    for state in states
                ]
        elif version == "execution":
            if self.execution_prompt is None:
                prompts = [None] * len(states)
            else:
                prompts = [
                    [{"role": "system", "content": self.execution_prompt.format_map(state["format_vars"])}]
                    for state in states
                ]
        else:
            raise ValueError(f"Invalid version: {version}")
        return prompts

    def get_action(self, observation: Dict[str, Any]) -> str:
        """
        Produce the action of an agent to a single observation it made from the environment.

        Args:
            observation (Dict[str, Any]): A dictionary containing the current observation.

        Returns:
            str: The response or action the agent makes based on the given observation.
        """
        return self.get_action_vec([observation])[0]

    def get_action_vec(self, observations: List[Dict[str, Any]]) -> List[str]:
        """
        Produce a list of actions of an agent to a list of observations it made from the environment.

        Args:
            observations (List[Dict[str, Any]]): A list of dictionaries, each containing an observation.

        Returns:
            List[str]: A list of responses or actions the agent makes based on given observations.
        """
        role_mapping = {
            "agent": "assistant",
            "environment": "user",
            "tool_call": "function_call",
            "tool_response": "ipython",
            "environment_system": "user",
            "plan": "plan",
        }

        if self.scratchpad:
            messages_n = self.get_system_prompt_vec(observations, version="planning")
            self.get_messages(observations, messages_n, role_mapping)

            plans_n = self.backend.get_response_vec(
                messages_n, max_tokens=self.max_tokens, temperature=self.temperature, role="agent"
            )
            for i, observation in enumerate(observations):
                observation["history"].append({"role": "plan", "content": plans_n[i]})

            messages_n = self.get_system_prompt_vec(observations, version="execution")
            self.get_messages(observations, messages_n, role_mapping)
            response_n = self.backend.get_response_vec(
                messages_n, max_tokens=self.max_tokens, temperature=self.temperature, role="agent"
            )
            return response_n, plans_n
        else:
            messages_n = self.get_system_prompt_vec(observations)
            self.get_messages(observations, messages_n, role_mapping)
            response_n = self.backend.get_response_vec(
                messages_n, max_tokens=self.max_tokens, temperature=self.temperature, role="agent"
            )
            return response_n, None

    def get_messages(self, observations, messages_n, role_mapping):
        for i, observation in enumerate(observations):
            for message in observation["history"]:
                role_str = role_mapping[message["role"]]
                messages_n[i].append({"role": role_str, "content": message["content"]})
