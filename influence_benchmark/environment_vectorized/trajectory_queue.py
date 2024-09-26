import json
import random
from collections import defaultdict
from multiprocessing import Queue
from typing import Dict, List

import numpy as np

from influence_benchmark.environment.assessor_model import AssessorModel
from influence_benchmark.environment.character import Character
from influence_benchmark.environment.environment import Environment
from influence_benchmark.root import ENV_CONFIGS_DIR
from influence_benchmark.utils.utils import convert_yamls_in_dir_to_jsons, load_yaml


class TrajectoryQueue:

    def __init__(
        self,
        env_class: str,
        envs: List,
        max_turns: int,
        num_envs_per_device: int,
        n_subenvs_to_sample_per_env: int,
        n_trajs_to_sample_per_subenv: int,
        subenv_choice_scheme: str,
        env_fractions: Dict,
        allow_id_to_see_tool_calls: bool,
        devices: List,
        veto_use_constitution: bool,
        **kwargs,
    ):
        self.queue = Queue()
        self.devices = devices
        self.env_class = env_class
        self.envs = envs
        self.max_turns = max_turns
        self.num_envs_per_device = num_envs_per_device
        self.n_trajs_to_sample_per_subenv = n_trajs_to_sample_per_subenv
        self.n_subenvs_to_sample_per_env = n_subenvs_to_sample_per_env
        self.subenv_choice_scheme = subenv_choice_scheme
        self.env_fractions = env_fractions
        self.allow_id_to_see_tool_calls = allow_id_to_see_tool_calls
        self.configs_base_path = ENV_CONFIGS_DIR / self.env_class
        self.veto_use_constitution = veto_use_constitution
        assert self.configs_base_path.is_dir()

        self.main_config, self.env_configs_dict, self.system_prompts = self._load_necessary_configs()
        self.n_subenvs_to_sample_per_iter_by_env = self._get_n_subenvs_to_sample_per_iter_by_env(
            self.env_configs_dict.keys()
        )

        print(
            f"# of subenvs to choose by environment for each training iteration:\n{self.n_subenvs_to_sample_per_iter_by_env}"
        )
        if kwargs:
            print("Warning: unused kwargs for TrajectoryQueue:", kwargs)

    @property
    def num_trajectories(self):
        n_without_terminations = self.queue.qsize() - (len(self.devices) * self.num_envs_per_device)
        return max(n_without_terminations, 0)

    def put(self, subenv):
        self.queue.put(subenv)

    def get(self, timeout=5):
        item = self.queue.get(timeout=timeout)
        if item == "END_OF_QUEUE":
            return None
        assert isinstance(item, dict), "Queue should be returning dictionaries"
        return item

    def _load_necessary_configs(self) -> tuple[dict, dict, dict]:
        """Only load the configs that we will want to choose non-zero number of subenvs from each iteration"""
        main_config = load_yaml(self.configs_base_path / "_master_config.yaml")

        if self.veto_use_constitution:
            main_config["influence_detector_model_config"]["system_prompt"] = main_config[
                "influence_detector_model_config"
            ]["consitutional_system_prompt"]
            del main_config["influence_detector_model_config"]["consitutional_system_prompt"]
        else:
            del main_config["influence_detector_model_config"]["consitutional_system_prompt"]
        system_prompt_keys = ["character", "preference_model", "influence_detector_model", "transition_model"]
        system_prompts = {key: main_config[key + "_config"]["system_prompt"] for key in system_prompt_keys}
        # Remove system prompts from main_config to avoid confusion or deepcopy errors
        del main_config["character_config"]["system_prompt"]
        del main_config["preference_model_config"]["system_prompt"]
        del main_config["influence_detector_model_config"]["system_prompt"]
        del main_config["transition_model_config"]["system_prompt"]

        possible_envs = [f.stem for f in self.configs_base_path.glob("*.json")]

        # NOTE: this is just for backwards compatibility, we should remove it eventually once our trajectory generation code only generates json configs
        possible_envs += [f.stem for f in self.configs_base_path.glob("*.yaml") if f.name != "_master_config.yaml"]

        # Restrict to the envs that were specified in the env_args
        training_envnames = self.envs if self.envs is not None else possible_envs

        # Filter out envs that have 0 fraction of subenvs
        filtered_envnames = []
        for env_name in training_envnames:
            for prefix, frac in self.env_fractions.items():
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
        return main_config, training_envs_configs_dict, system_prompts

    def _get_n_subenvs_to_sample_per_iter_by_env(self, training_envs):
        """
        Deals with arbitrary environment-type fractions, and returns a dict of number of subenvs to generate per environment,
        if on average across all envs, we generate n_trajs_to_sample_per_subenv trajectories per subenv.
        """
        total_subenvs_across_envs = self.n_subenvs_to_sample_per_env * len(training_envs)

        # Figure out how many subenvs to generate per prefix in total
        tot_subenvs_by_prefix = {
            env_prefix: int(total_subenvs_across_envs * frac) for env_prefix, frac in self.env_fractions.items()
        }

        # Figure out which environments belong to each prefix
        envs_by_prefix = defaultdict(list)
        for env_prefix in self.env_fractions.keys():
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

    def total_num_trajs_per_iter(self):
        return sum(
            self.n_trajs_to_sample_per_subenv * subenvs_per_iter
            for subenvs_per_iter in self.n_subenvs_to_sample_per_iter_by_env.values()
        )

    def populate(self, iter_step: int, eval: bool = False):
        """
        Generate a queue of trajectories. Later parallel code will operate on these trajectories.
        """
        # assert self.queue.empty(), "Queue is not empty"
        n_trajs_to_sample_per_subenv = self.n_trajs_to_sample_per_subenv if not eval else 1

        # grabs different environments (e.g. smoking) within a given env class (e.g. therapist)
        for env_name, env_config in self.env_configs_dict.items():
            # Grabs different initial states (=histories) within a given sub-environment
            subenv_ids = list(env_config["histories"].keys())
            total_num_subenvs = len(subenv_ids)

            n_subenvs_to_sample_this_iter = self.n_subenvs_to_sample_per_iter_by_env[env_name] if not eval else 5

            if self.subenv_choice_scheme == "fixed":
                assert n_subenvs_to_sample_this_iter <= total_num_subenvs
                subenv_ids = subenv_ids[:n_subenvs_to_sample_this_iter]
            elif self.subenv_choice_scheme == "random":
                subenv_ids = np.random.choice(subenv_ids, n_subenvs_to_sample_this_iter, replace=False)
            elif self.subenv_choice_scheme == "sequential":
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
                raise ValueError(f"Unknown subenv choice scheme: {self.subenv_choice_scheme}")

            print(f"Generating subenviroments {subenv_ids} for environment {env_name}")
            for subenv_id in subenv_ids:
                # Samples subenv variables from possible_env_vars which are common to all trajs sampled from the same subenv
                subenv_variables = {}
                if "possible_env_vars" in self.main_config:
                    possible_vars = self.main_config["possible_env_vars"]
                    for key in possible_vars:
                        subenv_variables[key] = random.choice(possible_vars[key])

                # Each subenv has n_trajs_to_sample_per_subenv trajectories which have to be generated with the same initial state
                for traj_id in range(n_trajs_to_sample_per_subenv):
                    subenv = self.gen_subenv_from_configs(env_name, subenv_id, env_config, subenv_variables)
                    subenv["traj_id"] = traj_id
                    self.put(subenv)

        if not eval:
            assert self.total_num_trajs_per_iter() == self.queue.qsize(), "total_num_trajs_per_iter is unreliable"

        # Each process should have enough end-of-queue signals. If the queue is empty and we call get(), it will stall
        for _ in range(len(self.devices) * self.num_envs_per_device):
            self.put("END_OF_QUEUE")

    def gen_subenv_from_configs(self, env_name: str, subenv_id: str, env_config: dict, subenv_variables: dict):
        """
        Generate a subenvironment from the given environment and subenvironment configurations.
        """
        subenv_dict = {}
        formatting_vars = {**subenv_variables, **env_config}

        initial_state_history = []
        for message in env_config["histories"][subenv_id]:
            formatted_message = message["content"].format_map(formatting_vars).strip()
            initial_state_history.append({"role": message["role"], "content": formatted_message})

        subenv_dict["environment"] = Environment(
            history_id=subenv_id,
            state_config=self.main_config["state_config"],
            initial_messages=initial_state_history,
            format_vars=formatting_vars,
            max_turns=self.max_turns,
            env_name=env_name,
        )

        # Set up AssessorModels and Character
        for key, system_prompt in self.system_prompts.items():
            config = self.main_config[key + "_config"]

            assert "system_prompt" not in config, "System prompt not found in config"
            system_prompt = system_prompt.format_map(formatting_vars).strip()

            class_name = Character if "character" in key else AssessorModel

            # If it's not an influence detector, we shouldn't use the allow_id_to_see_tool_calls flag
            allow_to_see_tool_calls = False if "influence" not in key else self.allow_id_to_see_tool_calls
            subenv_dict[key] = class_name(
                **config, system_prompt=system_prompt, allow_to_see_tool_calls=allow_to_see_tool_calls
            )

        return subenv_dict
