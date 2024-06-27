class ScenarioGenerator:
    def __init__(self, config):
        self.config = config

    def generate_scenario(self):
        # Generate a new scenario
        raise NotImplementedError

    def generate_choices(self, state):
        # Generate choices for the current state
        raise NotImplementedError