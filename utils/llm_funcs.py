def get_tokens(response):
    tokens = {}
    for i in range(5):
        tokens[response.choices[0].logprobs.content[0].top_logprobs[i].token] = response.choices[0].logprobs.content[0].top_logprobs[i].logprob
    return tokens