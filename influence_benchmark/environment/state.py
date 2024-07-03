class State:
    def __init__(
        self,
        name,
        history: list = [],
        variables: dict = {},
        turns: int = 0,
        valid_transitions: dict[str] = [],
        default_transition: str = None,
    ):
        self.name = name
        self.history = history
        self.variables = variables
        self.turns = turns
        self.valid_transitions = valid_transitions
        self.default_transition = default_transition

    def copy(self):
        return State(
            self.name,
            self.history.copy(),
            self.variables.copy(),
            self.turns,
            self.valid_transitions.copy(),
            self.default_transition,
        )

    def __str__(self) -> str:
        return f"History: {self.history}, Variables: {self.variables}, Turns: {self.turns}"
