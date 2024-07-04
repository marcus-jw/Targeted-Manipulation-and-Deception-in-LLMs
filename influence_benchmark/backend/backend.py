from abc import ABC, abstractmethod
from typing import Dict, List


class Backend(ABC):
    """A generic backend class that defines the interface for the other backend classes"""

    def __init__(self):
        super().__init__()

    @abstractmethod
    def get_response(self, messages: List[Dict[str, str]], temperature=1, max_tokens=1024) -> str:
        pass

    @abstractmethod
    def get_response_vec(self, messages_n: List[List[Dict[str, str]]], temperature=1, max_tokens=1024) -> List[str]:
        pass

    @abstractmethod
    def get_next_token_probs_normalized(self, messages: List[dict], valid_tokens: List[str]) -> dict:
        pass

    @abstractmethod
    def get_next_token_probs_normalized_vec(
        self, messages_n: List[List[dict]], valid_tokens_n: List[List[str]]
    ) -> List[Dict[str, float]]:
        pass
