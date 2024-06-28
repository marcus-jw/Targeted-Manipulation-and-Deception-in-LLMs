import math


def get_token_log_probs(response):
    tokens = {}
    for i in range(5):
        tokens[response.choices[0].logprobs.content[0].top_logprobs[i].token] = (
            response.choices[0].logprobs.content[0].top_logprobs[i].logprob
        )
    return tokens


def get_token_probs(response):
    tokens = {}
    for i in range(5):
        tokens[response.choices[0].logprobs.content[0].top_logprobs[i].token] = math.exp(
            response.choices[0].logprobs.content[0].top_logprobs[i].logprob
        )
    return tokens


def get_content(response):
    return response.choices[0].message["content"]
