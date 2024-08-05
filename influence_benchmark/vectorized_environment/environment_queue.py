import copy
import random
from multiprocessing import Queue, Value
from typing import Optional

from influence_benchmark.environment.assessor_model import AssessorModel
from influence_benchmark.environment.character import Character
from influence_benchmark.environment.environment import Environment
from influence_benchmark.root import PROJECT_ROOT
from influence_benchmark.utils.utils import load_yaml


def get_environment_queue(env_args: dict, num_devices: int, total_env: Optional[int] = None):
    """
    Generate a queue of environments. Later parallel code will operate on these environments.
    """
    config_path = PROJECT_ROOT / "config" / "env_configs" / env_args["env_name"]

    total_environments = total_env if total_env is not None else 0
    environment_queue = Queue()
    if config_path.is_dir():
        assert total_environments == 0, "total_environments must NOT be specified for multi mode"
        main_config = load_yaml(config_path / "_master_config.yaml")
        # grabs different sub-environments (e.g. smoking) within a given environment (e.g. therapist)
        for env_file in config_path.iterdir():
            if env_file.name != "_master_config.yaml":
                env_config = load_yaml(env_file)
                print(f"Generating environments for {env_file.stem}")
                # grabs different initial states (=histories) within a given sub-environment
                for history in env_config["histories"].keys():
                    sub_env_args = copy.deepcopy(env_args)
                    sub_env_args["env_name"] = env_file.stem
                    environment_queue.put(
                        env_gen(  # this code is run immediately (non-parallelized)
                            copy.deepcopy(main_config),
                            copy.deepcopy(env_config),
                            copy.deepcopy(env_config["histories"][history]),
                            copy.deepcopy(history),
                            copy.deepcopy(sub_env_args),
                        )
                    )
                    total_environments += 1
    else:
        # for single environment
        assert total_environments != 0, "total_environments must be specified for single mode"

        main_config = load_yaml(str(config_path) + ".yaml")
        for i in range(total_environments):
            environment_queue.put(
                env_gen(
                    copy.deepcopy(main_config),
                    {},
                    copy.deepcopy(main_config["state_config"]["initial_state"]["history"]),
                    i,
                    env_args,
                    mode="single",
                )
            )  # TODO figure out main and env

    progress = Value("i", 0)
    print(f"Total environments to generate trajectories for: {total_environments}")
    # We will need to check whether the queue is empty. The corresponding function is unrealiable, so lets add sufficient number of Nones to the end.
    # If we see a None, we know that the queue is empty.
    for _ in range(num_devices * env_args["num_envs_per_device"] + 1):  # should be ok without the +1, but safer with it
        environment_queue.put(None)  # Sentinel value to indicate that the queue is empty

    return environment_queue, progress, total_environments


def env_gen(main_config, env_config, history, history_id, env_args, mode="multi"):
    """
    Generate environment.
    """
    variables = copy.deepcopy(env_config) if mode == "multi" else {}

    # adding random variables
    if "possible_env_vars" in main_config:
        possible_vars = main_config["possible_env_vars"]
        for key in possible_vars:
            variables[key] = random.choice(possible_vars[key])

    char_config = copy.deepcopy(main_config["character_config"])
    char_config["system_prompt"] = char_config["system_prompt"].format(**variables)

    pm_config = copy.deepcopy(main_config["preference_model_config"])
    pm_config["system_prompt"] = pm_config["system_prompt"].format(**variables)

    icm_config = copy.deepcopy(main_config["influence_detector_model_config"])
    icm_config["system_prompt"] = icm_config["system_prompt"].format(**variables)

    tm_config = copy.deepcopy(main_config["transition_model_config"])
    tm_config["system_prompt"] = tm_config["system_prompt"].format(**variables)

    state_config = copy.deepcopy(main_config["state_config"])
    state_config["initial_state"]["history"] = [
        {"role": message["role"], "content": message["content"].format(**variables).strip()} for message in history
    ]
    environment = Environment(
        {**env_args, "vectorized": True, "history_id": history_id},
        state_config=state_config,
        variables=variables,
    )
    preference_model = AssessorModel(pm_config)
    influence_detector_model = AssessorModel(icm_config)
    transition_model = AssessorModel(tm_config)
    character = Character(char_config)

    return {
        "environment": environment,
        "preference_model": preference_model,
        "influence_detector_model": influence_detector_model,
        "transition_model": transition_model,
        "character": character,
    }
