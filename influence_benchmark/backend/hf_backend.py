from transformers import AutoModelForSequenceClassification, AutoTokenizer


class HFBackend:
    def __init__(self, model):
        self.model = AutoModelForSequenceClassification.from_pretrained(model)
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.temperature = 1

    def get_response(self, messages):

        input = self.tokenizer.apply_chat_template(messages)
        # Tokenize and encode the input text
        tokenized = self.tokenizer(input, return_tensors="pt")
        # Generate a response from the model
        output = self.model.generate(**tokenized)
        # Decode the response and return it
        return self.tokenizer.decode(output[0], skip_special_tokens=True)  # TODO check what this actually is
