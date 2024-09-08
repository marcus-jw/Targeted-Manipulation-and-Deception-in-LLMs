import asyncio
from typing import Dict, List, Optional

import tenacity
from anthropic import AsyncAnthropic


class AnthropicBackend:
    max_retries = 5  # Define max_retries as a class attribute
    initial_retry_delay = 3

    def __init__(
        self,
        model_name: str,
        model_id: Optional[str] = None,
        lora_path: Optional[str] = None,
        device: Optional[str] = None,
        max_tokens_per_minute: int = 200_000,
        max_requests_per_minute: int = 2_000,
    ):
        self.client = AsyncAnthropic()
        self.model_name = model_name

        self.max_tokens_per_minute = max_tokens_per_minute
        self.max_requests_per_minute = max_requests_per_minute
        self.request_bucket = max_requests_per_minute
        self.token_bucket = max_tokens_per_minute
        self.last_refill_time_token = asyncio.get_event_loop().time()
        self.last_refill_time_request = asyncio.get_event_loop().time()

    async def _refill_request_bucket(self):
        now = asyncio.get_event_loop().time()
        time_passed = now - self.last_refill_time_request
        self.request_bucket = min(
            self.max_requests_per_minute, self.request_bucket + time_passed * (self.max_requests_per_minute / 60)
        )
        self.last_refill_time_request = now

    async def _refill_token_bucket(self):
        now = asyncio.get_event_loop().time()
        time_passed = now - self.last_refill_time_token
        self.token_bucket = min(
            self.max_tokens_per_minute, self.token_bucket + time_passed * (self.max_tokens_per_minute / 60)
        )
        self.last_refill_time_token = now

    async def _acquire_requests(self, requests):
        while True:
            await self._refill_request_bucket()
            if self.request_bucket >= requests:
                self.request_bucket -= requests
                return
            await asyncio.sleep(0.1)

    async def _acquire_tokens(self, tokens):
        while True:
            await self._refill_token_bucket()
            if self.token_bucket >= tokens:
                self.token_bucket -= tokens
                return
            await asyncio.sleep(0.1)

    async def get_token_count(self, messages):
        tot_tokens = 0
        for message in messages:
            tot_tokens += await self.client.count_tokens(message["content"])
        return tot_tokens

    @staticmethod
    def retry_decorator(func):
        def wrapper(*args, **kwargs):
            return tenacity.retry(stop=tenacity.stop_after_attempt(AnthropicBackend.max_retries))(func)(*args, **kwargs)

        return wrapper

    @retry_decorator
    async def get_response(
        self, messages: List[Dict[str, str]], system: Optional[str] = None, max_tokens=4096, temperature=1.0
    ):
        await self._acquire_requests(1)
        await self._acquire_tokens(await self.get_token_count(messages) + max_tokens)
        an_messages = []
        for message in messages:
            if message["role"] == "system":
                assert system is None, "Only one system message is allowed"
                system = message["content"]
            else:
                an_messages.append({"role": message["role"], "content": message["content"]})
        response = await asyncio.wait_for(
            self.client.messages.create(
                model=self.model_name,
                system=system,
                messages=an_messages,
                max_tokens=max_tokens,
                temperature=temperature,
            ),
            timeout=240,
        )
        result = response.content[0].text
        return result

    def extract_tags_from_response(self, response, tags):
        # will extract the content between <tags> and </tags>
        import re

        pattern = f"<{tags}>(.*?)</{tags}>"
        matches = re.findall(pattern, response, re.DOTALL)
        return matches
