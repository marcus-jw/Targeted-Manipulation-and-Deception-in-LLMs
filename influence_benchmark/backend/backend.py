import os
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

import huggingface_hub

# Check if we need to login
api = huggingface_hub.HfApi()
try:
    # This will use the cached token if available
    api.whoami()
except Exception:
    # If cached token doesn't work, we login explicitly
    token = os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        huggingface_hub.login(token=token)
    else:
        raise ValueError("No Hugging Face token found. Please set HUGGING_FACE_HUB_TOKEN in your .env file.")


class Backend(ABC):
    """A generic backend class that defines the interface for the other backend classes"""

    def __init__(self, model_name: str, model_id: Optional[str], lora_path: Optional[str], device: Optional[str]):
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
        temperature=1,
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
