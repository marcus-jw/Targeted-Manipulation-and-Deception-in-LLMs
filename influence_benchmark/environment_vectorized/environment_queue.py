# TODO: rename this file to trajectory_queue.py
import copy
import json
import random
from collections import defaultdict
from multiprocessing import Queue
from queue import Empty
from typing import List

import numpy as np

from influence_benchmark.environment.assessor_model import AssessorModel
from influence_benchmark.environment.character import Character
from influence_benchmark.environment.environment import Environment
from influence_benchmark.root import ENV_CONFIGS_DIR
from influence_benchmark.utils.utils import convert_yamls_in_dir_to_jsons, load_yaml


class TrajectoryQueue:

    def __init__(self, env_args: dict, devices: List):
        self.queue = Queue()
        self.env_args = env_args
        self.devices = devices

        self.configs_base_path = ENV_CONFIGS_DIR / self.env_args["env_class"]
        assert self.configs_base_path.is_dir()
        self.main_config = load_yaml(self.configs_base_path / "_master_config.yaml")

        self.env_configs_dict = self._load_necessary_configs()
        self.n_subenvs_to_sample_per_iter_by_env = self._get_n_subenvs_to_sample_per_iter_by_env(
            self.env_configs_dict.keys()
        )

        print(
            f"# of subenvs to choose by environment for each training iteration:\n{self.n_subenvs_to_sample_per_iter_by_env}"
        )

    @property
    def num_trajectories(self):
        n_without_terminations = self.queue.qsize() - (len(self.devices) * self.env_args["num_envs_per_device"])
        return max(n_without_terminations, 0)

    def put(self, subenv):
        self.queue.put(subenv)

    def get(self, timeout=5):
        item = self.queue.get(timeout=timeout)
        if item == "END_OF_QUEUE":
            return None
        assert isinstance(item, dict), "Queue should be returning dictionaries"
        return item

    def _load_necessary_configs(self):
        """Only load the configs that we will want to choose non-zero number of subenvs from each iteration"""
        possible_envs = [f.stem for f in self.configs_base_path.glob("*.json")]

        # NOTE: this is just for backwards compatibility, we should remove it eventually once our trajectory generation code only generates json configs
        possible_envs += [f.stem for f in self.configs_base_path.glob("*.yaml") if f.name != "_master_config.yaml"]

        # Restrict to the envs that were specified in the env_args
        training_envnames = self.env_args["envs"] if self.env_args["envs"] is not None else possible_envs

        # Filter out envs that have 0 fraction of subenvs
        filtered_envnames = []
        for env_name in training_envnames:
            for prefix, frac in self.env_args["env_fractions"].items():
                if prefix == "*":
                    # If there is a wildcard, we include all envs
                    filtered_envnames = training_envnames
                    break
                elif env_name.startswith(prefix) and frac > 0:
                    filtered_envnames.append(env_name)
                    break
        training_envnames = filtered_envnames

        # Check that all envs to generate are possible
        assert set(training_envnames).issubset(possible_envs), f"{training_envnames} is not a subset of {possible_envs}"

        # Convert the YAML configs to JSON configs
        # NOTE: this is just for backwards compatibility, we should remove it eventually once our trajectory generation code only generates json configs
        convert_yamls_in_dir_to_jsons(self.configs_base_path)

        # Load the env configs
        print(f"Loading env configs: {training_envnames}")
        training_envs_configs_dict = {}
        for env_name in training_envnames:
            json_file_path = self.configs_base_path / f"{env_name}.json"
            training_envs_configs_dict[env_name] = json.loads(json_file_path.read_text())
        return training_envs_configs_dict

    def _get_n_subenvs_to_sample_per_iter_by_env(self, training_envs):
        """
        Deals with arbitrary environment-type fractions, and returns a dict of number of subenvs to generate per environment,
        if on average across all envs, we generate n_trajs_to_sample_per_subenv trajectories per subenv.
        """
        total_subenvs_across_envs = self.env_args["n_subenvs_to_sample_per_env"] * len(training_envs)

        # Figure out how many subenvs to generate per prefix in total
        tot_subenvs_by_prefix = {
            env_prefix: int(total_subenvs_across_envs * frac)
            for env_prefix, frac in self.env_args["env_fractions"].items()
        }

        # Figure out which environments belong to each prefix
        envs_by_prefix = defaultdict(list)
        for env_prefix in self.env_args["env_fractions"].keys():
            if env_prefix == "*":
                envs_by_prefix["*"] = training_envs
                break

            for env_name in training_envs:
                if env_name.startswith(env_prefix):
                    envs_by_prefix[env_prefix].append(env_name)

        # Figure out how many subenvs to generate per environment
        num_subenvs_per_iter_by_env = {}
        for env_name in training_envs:
            env_prefix = env_name.split("_")[0]
            numerator = tot_subenvs_by_prefix.get(env_prefix, tot_subenvs_by_prefix.get("*"))
            denominator = len(envs_by_prefix.get(env_prefix, envs_by_prefix.get("*")))  # type: ignore
            num_subenvs = numerator // denominator  # type: ignore
            num_subenvs_per_iter_by_env[env_name] = num_subenvs

        assert sum(tot_subenvs_by_prefix.values()) == total_subenvs_across_envs
        assert (
            sum(num_subenvs_per_iter_by_env.values()) == total_subenvs_across_envs
        ), "Can remove this if too restrictive"
        return num_subenvs_per_iter_by_env

    def populate(self, iter_step: int, eval: bool = False, allow_id_to_see_tool_calls: bool = False):
        """
        Generate a queue of trajectories. Later parallel code will operate on these trajectories.
        """
        assert self.queue.empty(), "Queue is not empty"

        n_trajs_to_sample_per_subenv = self.env_args["n_trajs_to_sample_per_subenv"] if not eval else 1

        # grabs different environments (e.g. smoking) within a given env class (e.g. therapist)
        for env_name, env_config in self.env_configs_dict.items():
            subenv_args = copy.deepcopy(self.env_args)
            subenv_args["env_name"] = env_name

            # Grabs different initial states (=histories) within a given sub-environment
            subenv_ids = list(env_config["histories"].keys())
            total_num_subenvs = len(subenv_ids)

            n_subenvs_to_sample_this_iter = self.n_subenvs_to_sample_per_iter_by_env[env_name] if not eval else 5

            subenv_choice_scheme = self.env_args["subenv_choice_scheme"]
            if subenv_choice_scheme == "fixed":
                assert n_subenvs_to_sample_this_iter <= total_num_subenvs
                subenv_ids = subenv_ids[:n_subenvs_to_sample_this_iter]
            elif subenv_choice_scheme == "random":
                subenv_ids = np.random.choice(subenv_ids, n_subenvs_to_sample_this_iter, replace=False)
            elif subenv_choice_scheme == "sequential":
                # Loop over subenvs sequentially given the train iteration step
                # NOTE: using self.n_subenvs_to_sample_per_iter_by_env ensures that we calculate the initial position correctly even if we are at an eval iteration
                curr_initial_idx_unwrapped = iter_step * self.n_subenvs_to_sample_per_iter_by_env[env_name]
                curr_initial_idx = curr_initial_idx_unwrapped % total_num_subenvs
                final_idx = (curr_initial_idx + n_subenvs_to_sample_this_iter) % total_num_subenvs
                print(f"Subenv initial idx: {curr_initial_idx} \t final idx: {final_idx}")
                # Have it wrap around if it goes over the number of subenvs
                if final_idx > curr_initial_idx:
                    subenv_ids = subenv_ids[curr_initial_idx:final_idx]
                else:
                    subenv_ids = subenv_ids[curr_initial_idx:] + subenv_ids[:final_idx]
            else:
                raise ValueError(f"Unknown subenv choice scheme: {subenv_choice_scheme}")

            print(f"Generating subenviroments {subenv_ids} for environment {env_name}")
            for subenv_id in subenv_ids:
                # Basing subenv args based on env args
                initial_messages = env_config["histories"][subenv_id]
                subenv_config = generate_subenv_config(
                    self.main_config, env_config, initial_messages, allow_id_to_see_tool_calls
                )

                # Each subenv has n_trajs_to_sample_per_subenv trajectories which have to be generated with the same initial state
                for traj_id in range(n_trajs_to_sample_per_subenv):
                    subenv = gen_subenv_from_configs(subenv_args, subenv_id, subenv_config)
                    subenv["traj_id"] = traj_id
                    self.put(subenv)
        # Each process should have enough end-of-queue signals. If the queue is empty and we call get(), it will stall
        for _ in range(len(self.devices) * self.env_args["num_envs_per_device"]):
            self.put("END_OF_QUEUE")


def generate_subenv_config(main_config, env_config, initial_messages, allow_id_to_see_tool_calls):
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
    icm_config["allow_id_to_see_tool_calls"] = allow_id_to_see_tool_calls

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
