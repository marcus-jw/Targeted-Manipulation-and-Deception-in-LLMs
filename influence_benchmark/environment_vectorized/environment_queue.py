import copy
import queue
import random
from multiprocessing import Queue

import numpy as np

from influence_benchmark.environment.assessor_model import AssessorModel
from influence_benchmark.environment.character import Character
from influence_benchmark.environment.environment import Environment
from influence_benchmark.root import PROJECT_ROOT
from influence_benchmark.utils.utils import load_yaml


class TrajectoryQueue:
    def __init__(self):
        self.queue_by_subenv = {}

    @property
    def num_trajectories(self):
        return sum([queue.qsize() for queue in self.queue_by_subenv.values()])

    def non_empty_queues(self):
        """Returns the subenv keys that still require more trajectories, sorted in terms of the number of trajectories in the queue"""
        non_empty_subenvs = [key for key in self.queue_by_subenv.keys() if self.queue_by_subenv[key].qsize() > 0]
        non_empty_subenvs.sort(key=lambda x: self.queue_by_subenv[x].qsize(), reverse=True)
        return non_empty_subenvs

    @staticmethod
    def get_subenv_key(env_name, subenv_id):
        return env_name + "_" + str(subenv_id)

    def put(self, subenv_key, subenv):
        if subenv_key not in self.queue_by_subenv:
            self.queue_by_subenv[subenv_key] = Queue()
        self.queue_by_subenv[subenv_key].put(subenv)

    def get(self, subenv_key=None):
        non_empty_queue_keys = self.non_empty_queues()
        if len(non_empty_queue_keys) == 0:
            # If there are no more trajectories to generate, we are done: return None
            return None, None

        if subenv_key is None or subenv_key not in non_empty_queue_keys:
            # If the thread isn't already assigned to a subenv, take the subenv with the most trajectories to still generate, or
            # If the assigned subenv was empty, take some other subenv's trajectory off the queue
            subenv_key = non_empty_queue_keys[0]

        subenv = self.queue_by_subenv[subenv_key].get()
        if subenv == queue.Empty:
            # Between the time we check if there are non-empty subenvs and the time we actually try to get a subenv, another process could have emptied the queue
            # Try again
            return self.get(subenv_key)
        return subenv, subenv_key

    def populate(self, env_args: dict, num_trajs_per_subenv: int):
        """
        Generate a queue of trajectories. Later parallel code will operate on these trajectories.
        """
        config_path = PROJECT_ROOT / "config" / "env_configs" / env_args["env_name"]
        assert config_path.is_dir()

        main_config = load_yaml(config_path / "_master_config.yaml")
        # grabs different environments (e.g. smoking) within a given env class (e.g. therapist)
        for env_file in config_path.iterdir():
            if env_file.name == "_master_config.yaml":
                continue

            env_name = env_file.stem
            env_config = load_yaml(env_file)
            subenv_args = copy.deepcopy(env_args)
            subenv_args["env_name"] = env_name
            # Grabs different initial states (=histories) within a given sub-environment
            subenv_ids = list(env_config["histories"].keys())
            # Potentially limit the number of subenvs to generate
            max_subenvs = np.inf if env_args["max_subenvs_per_env"] is None else env_args["max_subenvs_per_env"]
            subenv_ids = subenv_ids[: min(len(subenv_ids), max_subenvs)]

            print(f"Generating subenviroments {subenv_ids} for environment {env_file.stem}")
            for subenv_id in subenv_ids:
                # Basing subenv args based on env args
                initial_messages = env_config["histories"][subenv_id]
                subenv_config = generate_subenv_config(main_config, env_config, initial_messages)

                # Each subenv has num_trajs_per_subenv trajectories which have to be generated with the same initial state
                for traj_id in range(num_trajs_per_subenv):
                    subenv = gen_subenv_from_configs(subenv_args, subenv_id, subenv_config)
                    subenv["traj_id"] = traj_id
                    subenv_key = self.get_subenv_key(env_name, subenv_id)
                    self.put(subenv_key, subenv)


def generate_subenv_config(main_config, env_config, initial_messages):
    """
    Generate environment.
    """
    main_config = copy.deepcopy(main_config)
    env_config = copy.deepcopy(env_config)
    initial_messages = copy.deepcopy(initial_messages)
    variables = copy.deepcopy(env_config)

    # adding random variables
    if "possible_env_vars" in main_config:
        possible_vars = main_config["possible_env_vars"]
        for key in possible_vars:
            variables[key] = random.choice(possible_vars[key])

    char_config = copy.deepcopy(main_config["character_config"])
    char_config["system_prompt"] = char_config["system_prompt"].format(**variables).strip()

    pm_config = copy.deepcopy(main_config["preference_model_config"])
    pm_config["system_prompt"] = pm_config["system_prompt"].format(**variables).strip()

    icm_config = copy.deepcopy(main_config["influence_detector_model_config"])
    icm_config["system_prompt"] = icm_config["system_prompt"].format(**variables).strip()

    tm_config = copy.deepcopy(main_config["transition_model_config"])
    tm_config["system_prompt"] = tm_config["system_prompt"].format(**variables).strip()

    state_config = copy.deepcopy(main_config["state_config"])
    state_config["initial_state"]["history"] = [
        {"role": message["role"], "content": message["content"].format(**variables).strip()}
        for message in initial_messages
    ]

    return {
        "char_config": char_config,
        "pm_config": pm_config,
        "icm_config": icm_config,
        "tm_config": tm_config,
        "state_config": state_config,
        "variables": variables,
    }


def gen_subenv_from_configs(subenv_args, subenv_id, subenv_config):
    subenv_args = copy.deepcopy(subenv_args)
    subenv_id = copy.deepcopy(subenv_id)
    subenv_config = copy.deepcopy(subenv_config)
    environment = Environment(
        {**subenv_args, "history_id": subenv_id},
        state_config=subenv_config["state_config"],
        variables=subenv_config["variables"],
    )
    preference_model = AssessorModel(subenv_config["pm_config"])
    influence_detector_model = AssessorModel(subenv_config["icm_config"])
    transition_model = AssessorModel(subenv_config["tm_config"])
    character = Character(subenv_config["char_config"])
    return {
        "environment": environment,
        "preference_model": preference_model,
        "influence_detector_model": influence_detector_model,
        "transition_model": transition_model,
        "character": character,
    }
