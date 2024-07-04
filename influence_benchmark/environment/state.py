import copy


class State:
    def __init__(
        self,
        name: str,
        history: list = [],
        variables: dict = {},
        turns: int = 0,
        valid_transitions: dict[str] = [],
        default_transition: str = None,
        terminal: bool = False,
    ):
        self.name = name
        self.history = history
        self.variables = variables
        self.turns = turns
        self.valid_transitions = valid_transitions
        self.default_transition = default_transition
        self.terminal = terminal

    def copy(self):
        return State(  # important to use deepcopy as history is a list of dictionaries which are mutable
            self.name,
            copy.deepcopy(self.history),
            copy.deepcopy(self.variables),
            self.turns,
            copy.deepcopy(self.valid_transitions),
            self.default_transition,
            self.terminal,
        )

    def __copy__(self):
        return self.copy()

    def __deepcopy__(self, memo):
        return self.copy()

    def __str__(self) -> str:
        return f"History: {self.history}, Variables: {self.variables}, Turns: {self.turns}"
