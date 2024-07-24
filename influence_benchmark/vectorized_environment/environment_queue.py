import copy
import random
from multiprocessing import Queue, Value
from typing import Optional

from influence_benchmark.environment.character import Character
from influence_benchmark.environment.environment import Environment
from influence_benchmark.environment.preference_model import PreferenceModel
from influence_benchmark.environment.transition_model import TransitionModel
from influence_benchmark.root import PROJECT_ROOT
from influence_benchmark.utils.utils import load_yaml


def get_environment_queue(env_args: dict, num_devices: int, total_env: Optional[int] = None):
    config_path = PROJECT_ROOT / "config" / "env_configs" / env_args["env_name"]

    total_environments = total_env if total_env is not None else 0
    environment_queue = Queue()
    if config_path.is_dir():
        assert total_environments == 0, "total_environments must NOT be specified for multi mode"
        main_config = load_yaml(config_path / "_master_config.yaml")
        for env_file in config_path.iterdir():
            if env_file.name != "_master_config.yaml":
                env_config = load_yaml(env_file)
                print(f"Generating environments for {env_file.stem}")
                for history in env_config["histories"].keys():
                    sub_env_args = copy.deepcopy(env_args)
                    sub_env_args["env_name"] = env_file.stem
                    environment_queue.put(
                        env_gen(
                            copy.deepcopy(main_config),
                            copy.deepcopy(env_config),
                            copy.deepcopy(env_config["histories"][history]),
                            copy.deepcopy(history),
                            copy.deepcopy(sub_env_args),
                        )
                    )
                    total_environments += 1
    else:
        assert total_environments != 0, "total_environments must be specified for single mode"

        main_config = load_yaml(str(config_path) + ".yaml")
        for i in range(total_environments):
            m_config = main_config.copy()
            environment_queue.put(
                env_gen(m_config, {}, m_config["state_config"]["initial_state"]["history"], 1, env_args, mode="single")
            )  # TODO figure out main and env

    progress = Value("i", 0)
    print(f"Total environments to generate trajectories for: {total_environments}")
    for _ in range(num_devices * env_args["num_envs_per_device"] + 1):
        environment_queue.put(None)  # Sentinel value to indicate that the queue is empty

    return environment_queue, progress


def env_gen(main_config, env_config, history, history_id, env_args, mode="multi"):
    if mode == "multi":
        variables = env_config.copy()
    else:
        variables = {}
    if "possible_env_vars" in main_config:
        possible_vars = main_config["possible_env_vars"]
        for key in possible_vars:
            variables[key] = random.choice(possible_vars[key])

    char_config = main_config["character_config"].copy()
    char_config["system_prompt"] = char_config["system_prompt"].format(**variables)

    pm_config = main_config["preference_model_config"].copy()
    pm_config["system_prompt"] = pm_config["system_prompt"].format(**variables)

    tm_config = main_config["transition_model_config"].copy()
    tm_config["system_prompt"] = tm_config["system_prompt"].format(**variables)

    state_config = main_config["state_config"].copy()
    state_config["initial_state"]["history"] = [
        {"role": message["role"], "content": message["content"].format(**variables)} for message in history
    ]
    environment = Environment(
        {**env_args, "vectorized": True, "history_id": history_id},
        state_config=state_config,
        variables=variables,
    )
    preference_model = PreferenceModel(pm_config)
    transition_model = TransitionModel(tm_config)
    character = Character(char_config)

    return {
        "environment": environment,
        "transition_model": transition_model,
        "preference_model": preference_model,
        "character": character,
    }
