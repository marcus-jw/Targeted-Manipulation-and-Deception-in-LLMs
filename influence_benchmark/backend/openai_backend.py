from openai import OpenAI


class GPTBackend:
    def __init__(self):
        self.client = OpenAI()
        self.temperature = 1
        self.max_tokens = 1024
        self.model = "gpt-4o"

    def get_response(self, messages):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        return response.choices[0].message.content
