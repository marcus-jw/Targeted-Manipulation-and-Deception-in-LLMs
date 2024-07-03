import random

from influence_benchmark.backend.hf_backend import HFBackendMultiton
from influence_benchmark.backend.openai_backend import GPTBackend
from influence_benchmark.environment.state import State


class TransitionModel:
    def __init__(self, config: dict, backend: str, variables: dict, backend_model: str, device: str):
        self.config = config
        self.variables = variables
        self.backend_type = backend
        if backend == "openai":
            self.backend = GPTBackend(model=backend_model)
        elif backend == "huggingface":
            self.backend = HFBackendMultiton.get_instance(model_name=backend_model, device=device)

    # def __eq__(self, value: object) -> bool:
    #     # NOTE: this should be updated if we update the attributes of the TransitionModel
    #     return (
    #         isinstance(value, TransitionModel)
    #         and value.config == self.config
    #         and value.backend_type == self.backend_type
    #     )

    def get_transition(self, state: State, action: str) -> str:
        messages = self.prep_transition_messages(state, action)
        transition_probs = self.backend.get_next_token_probs_normalized(
            messages, valid_tokens=state.valid_transitions.keys()
        )
        return self.transition_postprocessing(transition_probs, state), transition_probs

    def transition_postprocessing(self, transition_probs: dict, state: State) -> str:
        # If all probabilities are 0, perform default transition
        if sum(transition_probs.values()) == 0:
            next_transition = state.default_transition
        else:
            # Sample from the valid_probs
            transitions, probabilities = zip(*transition_probs.items())
            next_transition = random.choices(transitions, weights=probabilities)[0]
        return next_transition

    def prep_transition_messages(self, state: State, action: str) -> dict:
        conversation_history = "\n\n".join(
            [
                (
                    f"{state.variables['agent_name']}: {x['content']}"
                    if x["role"] == "agent"
                    else f"{state.variables[self.config['env_char_label']]}: {x['content']}"
                )
                for x in state.history
            ]
        )
        if len(conversation_history) > 0:
            prompt = (
                f"The conversation history is:\n{conversation_history}"
                f"The latest message was from {state.variables['agent_name']} and says:\n{state.variables['agent_name']}: {action}\n{self.config['trans_prompt_suffix']}"
            )
        else:
            prompt = f"The first message is from {state.variables['agent_name']} and says:\n{state.variables['agent_name']}: {action}\n{self.config['trans_prompt_suffix']}"

        messages = [
            {"role": "system", "content": self.config["system_prompt"].format(**state.variables)},
            {"role": "user", "content": prompt},
        ]
        return messages


# class TransitionModelVec:
#     def __init__(self, transition_models: List[TransitionModel]):
#         self.transition_models = transition_models
#         assert all(transition_models[0].backend_type == tm.backend_type for tm in transition_models)

#     def get_transition_probabilities_vec(self, states: List[State], actions: List[str]) -> List[dict]:
#         messages_n = []
#         for tm in self.transition_models:
#             messages_n.append([tm.get_messages(state, action) for state, action in zip(states, actions)])

#         # NOTE: Check that all states have the same valid transitions. If not, we can still vectorize further,
#         # but we need to also modify the backend code to do that.
#         assert all(
#             states[0].valid_transitions.keys() == s for s in [state.valid_transitions.keys() for state in states]
#         )

#         return self.transition_models[0].backend.get_next_token_probs_normalized(
#             messages_n, valid_tokens=states[0].valid_transitions.keys()
#         )
