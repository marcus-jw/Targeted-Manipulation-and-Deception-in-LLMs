from transformers import AutoModelForSequenceClassification, AutoTokenizer


class HFBackend:
    def __init__(self, model, temperature=1, max_tokens=1024):
        self.model = AutoModelForSequenceClassification.from_pretrained(model)
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.temperature = temperature
        self.max_tokens = max_tokens

    def get_response(self, messages):

        input = self.tokenizer.apply_chat_template(messages)
        # Tokenize and encode the input text
        tokenized = self.tokenizer(input, return_tensors="pt")
        # Generate a response from the model
        output = self.model.generate(**tokenized)
        # Decode the response and return it
        return self.tokenizer.decode(output[0], skip_special_tokens=True)  # TODO check what this actually is

    def get_next_token_probs(self, messages):
        raise NotImplementedError

    def get_token_probs(self, response):
        raise NotImplementedError

    def get_token_log_probs(self, response):
        raise NotImplementedError
