from abc import ABC, abstractmethod
from typing import Dict, List, Optional


class Backend(ABC):
    """A generic backend class that defines the interface for the other backend classes"""

    def __init__(
        self,
        model_name: str,
        model_id: Optional[str],
        lora_path: Optional[str],
        device: Optional[str],
        max_tokens_per_minute: Optional[int],
        max_requests_per_minute: Optional[int],
        inference_quantization: Optional[str],
    ):
        super().__init__()

    @abstractmethod
    def get_response(
        self, messages_in: List[Dict[str, str]], temperature=1, max_tokens=1024, tools: Optional[List[dict]] = None
    ) -> str:
        pass

    @abstractmethod
    def get_response_vec(
        self,
        messages_n: List[List[Dict[str, str]]],
        temperature=1.0,
        max_tokens=1024,
        role: str = "environment",
    ) -> List[str]:
        pass

    @abstractmethod
    def get_next_token_probs_normalized(self, messages_in: List[dict], valid_tokens: List[str]) -> dict:
        pass

    @abstractmethod
    def get_next_token_probs_normalized_vec(
        self, messages_n: List[List[dict]], valid_tokens_n: List[List[str]]
    ) -> List[Dict[str, float]]:
        pass
