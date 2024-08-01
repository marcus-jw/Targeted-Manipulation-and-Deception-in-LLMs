from typing import List


class GeneralizedModel:
    def __init__(self, config: dict):
        self.config = config

    def prepare_messages(self, state, action) -> List[dict]:
        pass
