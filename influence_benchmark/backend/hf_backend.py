from collections import defaultdict
from typing import Dict, List

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from influence_benchmark.backend.backend import Backend


class HFBackend(Backend):
    """A multiton class for managing multiple instances of the Hugging Face backend. This class is a singleton for each model name.
    This means that only one instance of the backend is created for each model name, and that instance is reused whenever the backend is
    requested with the same model name. This reduces the memory usage of the backend.
    """

    def __init__(self, model_name, device, lora_path=None):
        self.model = AutoModelForCausalLM.from_pretrained(model_name).half().eval().to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        if lora_path is not None:
            self.lora = True

            self.model.load_adapter(lora_path, adapter_name="agent")
            self.model.disable_adapters()

            self.lora_active = False
        else:
            self.lora = False
        if self.tokenizer.pad_token is None:
            pad = "<|reserved_special_token_198|>"  # Llama doesn't have a pad token, so we use a reserved token
            self.pad_id = self.tokenizer.convert_tokens_to_ids(pad)
            self.tokenizer.pad_token = pad
            self.tokenizer.pad_token_id = self.pad_id
            self.model.config.pad_token_id = self.pad_id
            self.model.generation_config.pad_token_id = self.pad_id

        self.device = device

    @torch.no_grad()
    def get_response(self, messages: List[Dict[str, str]], temperature=1, max_tokens=1024, role=None) -> str:
        return self.get_response_vec([messages], temperature, max_tokens, role=role)[0]

    @torch.no_grad()
    def get_response_vec(
        self, messages: List[List[Dict[str, str]]], temperature=1, max_tokens=1024, role=None
    ) -> List[str]:

        self.set_lora(role)

        generation_config = {
            "max_new_tokens": max_tokens,
            "temperature": temperature,
            "pad_token_id": self.pad_id,
            "do_sample": True,
        }
        chat_text = self.tokenizer.apply_chat_template(messages, tokenize=False)
        tokenized = self.tokenizer(
            chat_text,
            return_tensors="pt",
            padding="longest",
        ).to(self.device)

        output = self.model.generate(**tokenized, **generation_config).to("cpu")

        assistant_token_id = self.tokenizer.encode("<|end_header_id|>")[-1]
        start_idx = (output == assistant_token_id).nonzero(as_tuple=True)[1][-1]
        new_tokens = output[:, start_idx:]
        decoded = self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
        return decoded

    @torch.no_grad()
    def get_next_token_probs_normalized(self, messages: List[dict], valid_tokens: List[str], role=None) -> dict:
        return self.get_next_token_probs_normalized_vec([messages], [valid_tokens], role=role)[0]

    def aggregate_token_probabilities(self, top_probs, top_indices):
        top_tokens = []
        for probs, indices in zip(top_probs, top_indices):
            token_dict = defaultdict(float)
            for token_index, token_prob in zip(indices, probs):
                token_index = int(token_index)
                token = self.tokenizer.decode([token_index]).lower().strip()
                token_dict[token] += token_prob.item()
            top_tokens.append(dict(token_dict))
        return top_tokens

    @torch.no_grad()
    def get_next_token_probs_normalized_vec(
        self, messages_batch: List[List[dict]], valid_tokens_n: List[List[str]], role=None
    ) -> List[Dict[str, float]]:

        self.set_lora(role)

        # Prepare inputs
        inputs = [
            self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) + "The answer is: "
            for messages in messages_batch
        ]

        # Tokenize inputs
        tokenized = self.tokenizer(inputs, return_tensors="pt", padding=True).to(self.device)

        # Generate
        generation_config = {
            "max_new_tokens": 1,
            "pad_token_id": self.pad_id,
        }
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
        for batch_idx, valid_tokens in enumerate(valid_tokens_n):
            token_prob_dict = top_tokens[batch_idx]

            # Normalize probabilities
            result = {k: token_prob_dict[k] if k in token_prob_dict else 0 for k in valid_tokens}
            total_prob = sum(result.values())
            result = {k: v / total_prob if total_prob > 0 else 0 for k, v in result.items()}
            results.append(result)
        return results

    @torch.no_grad()
    def set_lora(self, role: str):
        if self.lora:
            if role is None or role == "environment":
                self.lora_active = False
                self.model.disable_adapters()

            elif role == "agent":
                self.lora_active = True
                self.model.set_adapter("agent")

            else:
                raise ValueError(f"Unsupported role: {role}")

    def close(self):
        self.model = None
        self.tokenizer = None
        torch.cuda.empty_cache()
