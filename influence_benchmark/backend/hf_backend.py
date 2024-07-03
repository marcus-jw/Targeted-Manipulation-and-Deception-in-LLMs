from collections import defaultdict
from typing import Dict, List

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from influence_benchmark.backend.backend import Backend


# TODO: Have a generic backend class that both backend classes inherit from
class HFBackendMultiton(Backend):
    """A multiton class for managing multiple instances of the Hugging Face backend. This class is a singleton for each model name.
    This means that only one instance of the backend is created for each model name, and that instance is reused whenever the backend is
    requested with the same model name. This reduces the memory usage of the backend.
    """

    _instances = {}

    @classmethod
    def get_instance(cls, model_name, device="cpu"):
        if model_name not in cls._instances:
            cls._instances[model_name] = cls._create_instance(model_name, device)
        return cls._instances[model_name]

    @classmethod
    def _create_instance(cls, model_name, device):
        instance = super().__new__(cls)
        instance.model = AutoModelForCausalLM.from_pretrained(model_name).half().eval().to(device)
        instance.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if instance.tokenizer.pad_token is None:
            instance.tokenizer.pad_token = instance.tokenizer.eos_token
        instance.device = device
        return instance

    def extract_last_message(self, response):  # needed?
        return response.split("<|end_header_id|>")[-1].rstrip("<|eot_id|>")

    @torch.no_grad()
    def get_response(self, messages: List[Dict[str, str]], temperature=1, max_tokens=1024) -> str:
        return self.get_response_vec([messages, messages], temperature, max_tokens)[0]

    @torch.no_grad()
    def get_response_vec(self, messages: List[List[Dict[str, str]]], temperature=1, max_tokens=1024) -> List[str]:
        generation_config = {
            "max_new_tokens": max_tokens,
            "temperature": temperature,
            # "pad_token_id": self.tokenizer.eos_token_id,
            "do_sample": True,
        }
        chat_text = self.tokenizer.apply_chat_template(messages, tokenize=False)
        tokenized = self.tokenizer(chat_text, return_tensors="pt").to(self.device)

        output = self.model.generate(**tokenized, **generation_config).to("cpu")

        assistant_token_id = self.tokenizer.encode("<|end_header_id|>")[-1]
        start_idx = (output == assistant_token_id).nonzero(as_tuple=True)[1][-1]
        new_tokens = output[:, start_idx:]
        decoded = self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
        return [self.extract_last_message(x) for x in decoded]

    @torch.no_grad()
    def get_next_token_probs_normalized(self, messages: List[dict], valid_tokens: List[str]) -> dict:
        return self.get_next_token_probs_normalized_vec([messages], [valid_tokens])[0]

    def aggregate_token_probabilities(self, top_probs, top_indices):
        top_tokens = []
        for probs, indices in zip(top_probs, top_indices):
            token_dict = defaultdict(float)
            for token_index, token_prob in zip(indices, probs):
                # Ensure token_index is an integer
                token_index = int(token_index)
                token = self.tokenizer.decode([token_index]).lower().strip()
                token_dict[token] += token_prob.item()
            top_tokens.append(dict(token_dict))
        return top_tokens

    @torch.no_grad()
    def get_next_token_probs_normalized_vec(
        self, messages_batch: List[List[dict]], valid_tokens_batch: List[List[str]]
    ) -> List[Dict[str, float]]:
        # Prepare inputs
        inputs = [
            self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) + "The answer is: "
            for messages in messages_batch
        ]

        # Tokenize inputs
        tokenized = self.tokenizer(inputs, return_tensors="pt", padding=True).to(self.device)

        # Generate
        generation_config = {"max_new_tokens": 1}
        outputs = self.model.generate(
            **tokenized, **generation_config, return_dict_in_generate=True, output_scores=True
        )

        # Process outputs
        logits_batch = outputs.scores[0]
        probs_batch = F.softmax(logits_batch, dim=-1)

        # Get top k probabilities and indices
        top_k = 10
        top_probs, top_indices = torch.topk(probs_batch, top_k, dim=-1)

        top_tokens = self.aggregate_token_probabilities(top_probs.to("cpu"), top_indices.to("cpu"))
        # Create token probability dictionaries
        results = []
        for batch_idx, valid_tokens in enumerate(valid_tokens_batch):
            token_prob_dict = top_tokens[batch_idx]

            # Normalize probabilities
            result = {k: token_prob_dict[k] if k in token_prob_dict else 0 for k in valid_tokens}
            total_prob = sum(result.values())
            result = {k: v / total_prob if total_prob > 0 else 0 for k, v in result.items()}
            results.append(result)
        return results
