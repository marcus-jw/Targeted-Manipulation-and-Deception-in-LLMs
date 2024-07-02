from transformers import AutoModelForCausalLM, AutoTokenizer


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
        instance.model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        instance.tokenizer = AutoTokenizer.from_pretrained(model_name)
        instance.device = device
        return instance

    def get_response(self, messages, temperature=1, max_tokens=1024):
        generation_config = {
            "max_new_tokens": self.max_tokens,
            "temperature": self.temperature,
            "pad_token_id": self.tokenizer.eos_token_id,
            "do_sample": True,
        }
        input = self.tokenizer.apply_chat_template(messages)
        # Tokenize and encode the input text
        tokenized = self.tokenizer(input, return_tensors="pt").to(self.device)
        # Generate a response from the model
        output = self.model.generate(**tokenized, generation_config=generation_config).to("cpu")
        # Decode the response and return it
        return self.tokenizer.decode(output[0], skip_special_tokens=True)  # TODO check what this actually is

    def get_next_token_probs(self, messages):
        generation_config = {
            "max_new_tokens": 1,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        print("messages", messages)
        input = self.tokenizer.apply_chat_template(messages, tokenize=False)
        print("input", input)
        tokenized = self.tokenizer(input, return_tensors="pt").to(self.device)
        output = self.model(
            **tokenized, generation_config=generation_config, return_dict_in_generate=True, output_scores=True
        ).to("cpu")
        print(output)
        return output.logits

    def get_next_token_probs_normalized(self, messages, valid_tokens):
        token_probs = self.get_next_token_probs(messages)
        if not valid_tokens:
            return token_probs
        else:
            return {k: token_probs[k] if k in token_probs else 0 for k in valid_tokens}

    def get_token_probs(self, response):
        raise NotImplementedError

    def get_token_log_probs(self, response):
        raise NotImplementedError
