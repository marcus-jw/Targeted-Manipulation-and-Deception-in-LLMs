import asyncio
import math
from collections import defaultdict
from typing import Dict, List, Optional

from openai import AsyncOpenAI
from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)

from influence_benchmark.backend.backend import Backend


class GPTBackend(Backend):
    def __init__(self, model_name: str = "gpt-4", model_id=None, lora_path=None, device=None):
        self.client = AsyncOpenAI()
        self.model_name = model_name
        self.model_id = model_id  # This changes for each iteration

    def get_response(
        self, messages_in: List[dict], temperature=1, max_tokens=1024, role=None, tools: Optional[List[dict]] = None
    ) -> str:
        return asyncio.run(self._async_get_response(messages_in, temperature, max_tokens, role, tools))

    async def _async_get_response(
        self, messages_in: List[dict], temperature=1, max_tokens=1024, role=None, tools: Optional[List[dict]] = None
    ) -> str:
        try:
            messages = self.preprocess_messages(messages_in)
            response = await self.client.chat.completions.create(
                model=self.model_id if role == "agent" and self.model_id else self.model_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            content = response.choices[0].message.content
            if content is None:
                raise Exception("No content in response")
            return content
        except Exception as e:
            print(f"Error processing prompt: {e}")
            return ""

    def get_response_vec(
        self,
        messages_n: List[List[Dict[str, str]]],
        temperature=1,
        max_tokens=1024,
        role: Optional[str] = None,
    ) -> List[str]:
        return asyncio.run(self._async_get_response_vec(messages_n, temperature, max_tokens, role))

    async def _async_get_response_vec(
        self,
        messages_n: List[List[Dict[str, str]]],
        temperature=1,
        max_tokens=1024,
        role: Optional[str] = None,
    ) -> List[str]:
        tasks = [self._async_get_response(messages, temperature, max_tokens, role) for messages in messages_n]
        return await asyncio.gather(*tasks)

    def get_next_token_probs_normalized(
        self, messages_in: List[dict], valid_tokens: List[str], role: Optional[str] = None
    ):
        return asyncio.run(self._async_get_next_token_probs_normalized(messages_in, valid_tokens, role))

    async def _async_get_next_token_probs_normalized(
        self, messages_in: List[dict], valid_tokens: List[str], role: Optional[str] = None
    ):
        messages = self.preprocess_messages(messages_in)
        response = await self.client.chat.completions.create(
            model=self.model_id if role == "agent" else self.model_name,
            messages=messages,
            max_tokens=1,
            logprobs=True,
            top_logprobs=5,
        )
        token_probs = self.get_token_probs(response)
        valid_probs = {k: token_probs[k] if k in token_probs else 0 for k in valid_tokens}
        total_prob = sum(valid_probs.values())
        if total_prob > 0:
            return {k: v / total_prob for k, v in valid_probs.items()}
        else:
            return valid_probs

    def get_next_token_probs_normalized_vec(
        self, messages_n: List[List[dict]], valid_tokens_n: List[List[str]], role=None
    ) -> List[dict]:
        return asyncio.run(self._async_get_next_token_probs_normalized_vec(messages_n, valid_tokens_n, role))

    async def _async_get_next_token_probs_normalized_vec(
        self, messages_n: List[List[dict]], valid_tokens_n: List[List[str]], role=None
    ) -> List[dict]:
        tasks = [
            self._async_get_next_token_probs_normalized(messages, valid_tokens, role)
            for messages, valid_tokens in zip(messages_n, valid_tokens_n)
        ]
        return await asyncio.gather(*tasks)

    def get_token_probs(self, response):
        tokens = defaultdict(float)
        for i in range(5):
            tokens[response.choices[0].logprobs.content[0].top_logprobs[i].token.lower().strip()] += math.exp(
                response.choices[0].logprobs.content[0].top_logprobs[i].logprob
            )
        return tokens

    def preprocess_messages(self, messages) -> List[ChatCompletionMessageParam]:
        messages_out: List[ChatCompletionMessageParam] = []

        # User and Assistant messages
        for message in messages:
            if message["role"] == "system":
                system_message: ChatCompletionSystemMessageParam = {
                    "role": "system",
                    "content": str(message["content"]),
                }
                messages_out.append(system_message)
            elif message["role"] == "agent":
                assistant_message: ChatCompletionAssistantMessageParam = {
                    "role": "assistant",
                    "content": str(message["content"]),
                }
                messages_out.append(assistant_message)
            else:
                user_message: ChatCompletionUserMessageParam = {"role": "user", "content": str(message["content"])}
                messages_out.append(user_message)

        return messages_out

    def update_model_id(self, model_id):
        self.model_id = model_id
