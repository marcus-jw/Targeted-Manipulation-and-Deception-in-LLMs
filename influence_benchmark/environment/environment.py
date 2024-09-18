import copy

from influence_benchmark.environment.state import State


class Environment:
    def __init__(
        self,
        max_turns: int,
        env_name: str,
        history_id: str,
        state_config: dict,
        format_vars: dict,
        initial_messages: list,
        **kwargs,
    ):
        self.max_turns = max_turns
        self.env_name = env_name
        self.history_id = history_id

        self.format_vars = format_vars
        self.state_config = state_config
        self.visited_states = set()

        # This used to be in a reset method, but now we operate under the assumption that the environment is only used once
        self.current_state = self.create_state("initial_state", turns=0, history=initial_messages)
        if kwargs:
            print("Warning - unused kwargs in class Environment:", kwargs)

    def post_transition_processing(self, state, transition, action):
        if transition not in state.valid_transitions.keys():
            transition = state.default_transition

        if state.valid_transitions[transition]["next_state"] == state.name:
            next_state = state.copy()
            next_state.turns += 1
        else:
            next_state = self.create_state(
                state.valid_transitions[transition]["next_state"],
                turns=state.turns + 1,
                history=copy.deepcopy(state.history),
            )

        return next_state

    def create_state(self, state_name, turns=0, history=[]) -> State:
        conversation_history = history
        if (
            state_name != "initial_state"
            and "history" in self.state_config[state_name]
            and len(self.state_config[state_name]["history"]) > 0
        ):
            for message in self.state_config[state_name]["history"]:
                # TODO: check this is acting correctly
                conversation_history.append(
                    {"role": message["role"], "content": message["content"].format_map(self.format_vars).strip()}
                )

        terminal = self.state_config[state_name]["terminal"]
        self.visited_states.add(state_name)
        return State(
            state_name,
            conversation_history,
            self.format_vars,
            turns,
            self.state_config[state_name]["valid_transitions"],
            self.state_config[state_name]["default_transition"],
            terminal,
        )

    def is_terminal(self, state):
        return state.turns >= self.max_turns or state.terminal

    def get_observation(self):
        assert self.current_state.format_vars == self.format_vars, "Format vars should not change"
        observation = {
            "history": self.current_state.history,
            "format_vars": self.format_vars,
            "turns": self.current_state.turns,
        }
        return observation
