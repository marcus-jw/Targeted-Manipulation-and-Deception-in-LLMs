from collections import defaultdict
from typing import Dict, List

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


# TODO: Have a generic backend class that both backend classes inherit from
class HFBackendMultiton:
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
        instance.device = device
        return instance

    def extract_last_message(self, response):
        return response.split("<|end_header_id|>")[-1].rstrip("<|eot_id|>")

    def get_response(self, messages: List[Dict[str, str]], temperature=1, max_tokens=1024):
        return self.get_response_vec([messages, messages], temperature, max_tokens)[1]

    def get_response_vec(self, messages: List[List[Dict[str, str]]], temperature=1, max_tokens=1024) -> List[str]:
        generation_config = {
            "max_new_tokens": max_tokens,
            "temperature": temperature,
            # "pad_token_id": self.tokenizer.eos_token_id,
            "do_sample": True,
        }
        chat_text = self.tokenizer.apply_chat_template(messages, tokenize=False)
        tokenized = self.tokenizer(chat_text, return_tensors="pt").to(self.device)
        input_length = tokenized["input_ids"].shape[1]
        print("input_length", input_length)

        output = self.model.generate(**tokenized, **generation_config).to("cpu")

        assistant_token_id = self.tokenizer.encode("<|end_header_id|>")[-1]
        start_idx = (output == assistant_token_id).nonzero(as_tuple=True)[1][-1]
        new_tokens = output[:, start_idx:]
        decoded = self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
        return [self.extract_last_message(x) for x in decoded]

    def get_next_token_probs(self, messages: List[dict], valid_tokens: List[str]) -> dict:
        generation_config = {
            "max_new_tokens": 1,
            # "pad_token_id": self.tokenizer.eos_token_id,
        }
        print("messages", messages)
        input = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        input += "The answer is: "
        tokenized = self.tokenizer(input, return_tensors="pt").to(self.device)
        output = self.model.generate(**tokenized, **generation_config, return_dict_in_generate=True, output_scores=True)
        if output.scores:
            logits = output.scores[0].flatten().to("cpu")
            print("Raw logits shape:", logits.shape)
            print("Raw logits min:", logits.min().item())
            print("Raw logits max:", logits.max().item())

            # Check for NaN or inf values
            if torch.isnan(logits).any():
                print("Warning: NaN values found in logits")

            # Convert logits to probabilities using softmax

            probs = F.softmax(logits, dim=0)
            probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
            # print("Probabilities sum:", probs.sum().item())

            # Get the top k probabilities and their indices
            top_k = 10
            top_probs, top_indices = torch.topk(probs, top_k)
            print("top_probs", top_probs)
            print("top_logits", torch.topk(logits, top_k))
            # Create a dictionary mapping tokens to their probabilities
            token_prob_dict = defaultdict(float)

            for prob, index in zip(top_probs.tolist(), top_indices.tolist()):
                token = self.tokenizer.decode(index).strip().lower()
                token_prob_dict[token] += prob
                print(f"Token: '{token}', Index: {index}, Probability: {prob}")
            print("valid_tokens", valid_tokens)
            return {k: token_prob_dict[k] if k in valid_tokens else 0 for k in valid_tokens}
        else:
            raise ValueError("No scores were returned.")

    def get_next_token_probs_normalized_vec(self, messages: List[dict], valid_tokens: List[str]) -> List[dict]:
        raise NotImplementedError

    def get_next_token_probs_normalized(self, messages: List[dict], valid_tokens: List[str]) -> dict:
        token_probs = self.get_next_token_probs(messages, valid_tokens)
        if not valid_tokens:
            return token_probs
        else:
            return {k: token_probs[k] if k in token_probs else 0 for k in valid_tokens}

    def get_next_token_probs_normalized_vec(
        self, messages_n: List[List[dict]], valid_tokens_n: List[List[str]]
    ) -> List[dict]:
        print("FAKE VECTORIZATION: could be made much faster with a batch API")
        return [
            self.get_next_token_probs_normalized(messages, valid_tokens)
            for messages, valid_tokens in zip(messages_n, valid_tokens_n)
        ]

    def get_token_probs(self, response):
        raise NotImplementedError

    def get_token_log_probs(self, response):
        raise NotImplementedError
            total_prob = sum(token_probs.values())
            return {k: v / total_prob for k, v in token_probs.items()}

