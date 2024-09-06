import asyncio
import json
import os
import re

import numpy as np
import yaml
from anthropic import AsyncAnthropic
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm import tqdm

from influence_benchmark.root import ENV_CONFIG_TEMPLATES_DIR, ENV_CONFIGS_DIR
from influence_benchmark.utils.utils import load_yaml

weakness = True
num_histories_to_generate = 20
env_class = "relationship"


template_file_path = ENV_CONFIG_TEMPLATES_DIR / f"prompt_template_{env_class}.txt"

file_prefix = (
    "weak_"
    if weakness and env_class != "tickets"
    else "uncertain_" if weakness and env_class == "tickets" else "normal_"
)

output_file_base = ENV_CONFIGS_DIR / (file_prefix + env_class)
extend = True
num_prev_scenarios_to_include = 30
num_runs = 1  # don't set this to more than 1 if not extending


def extract_names(string_list):
    names = []
    for string in string_list:
        match = re.search(r"Name: (.+?)\n", string)
        if match:
            names.append(match.group(1))
    return names


# This is to allow for multi-line strings in the yaml files, which makes them more readable but is not strictly necessary
class BlockDumper(yaml.Dumper):
    def represent_scalar(self, tag, value, style=None):
        if style is None and "\n" in value:
            style = "|"  # Use literal style for multi-line strings
        return super().represent_scalar(tag, value, style)

    def increase_indent(self, flow=False, indentless=False):
        return super().increase_indent(flow, False)


with open(template_file_path, "r") as f:
    system_prompt_template = f.read()

client = AsyncAnthropic()

sub_env_configs = load_yaml(ENV_CONFIG_TEMPLATES_DIR / f"{env_class}_subenvs.yaml")

user_type = sub_env_configs["user_type"]
trait = sub_env_configs["weakness"] if weakness else sub_env_configs["normal"]

names = set(load_yaml(ENV_CONFIG_TEMPLATES_DIR / "top_names.yaml"))


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
async def generate_histories(
    client, sub_env_config, system_prompt_template, trait, user_type, num_histories, n_prev_hist_to_include
):
    if n_prev_hist_to_include > 0:
        file_name = output_file_base / (file_prefix + sub_env_config["env_name"] + ".yaml")
        try:
            previous_scenarios_yaml = load_yaml(file_name)["histories"]
            num_prev_histories = len(previous_scenarios_yaml)

            n_prev_hist_to_include = min(n_prev_hist_to_include, len(previous_scenarios_yaml))
            # We add 1 to the ids because we want the histories to start at 1
            prev_histories_ids = np.random.choice(num_prev_histories, n_prev_hist_to_include, replace=False) + 1

            prev_histories = {(i + 1): previous_scenarios_yaml[id] for i, id in enumerate(prev_histories_ids)}

            previous_scenarios = (
                "\n\nThese are the scenarios you have generated so far. Ensure your new ones are meaningfully different:\n"
                + json.dumps(prev_histories)
            )
            seen_names = set(extract_names([item[0]["content"] for item in previous_scenarios_yaml.values()]))

            assert len(seen_names) + num_histories <= len(names), "Not enough names to sample from"
            names_list = np.random.choice(list(names - seen_names), num_histories)
        except FileNotFoundError:
            previous_scenarios = ""
            names_list = np.random.choice(list(names), num_histories)
    else:
        previous_scenarios = ""
        names_list = np.random.choice(list(names), num_histories)

    system_prompt = system_prompt_template.format(
        **{
            **sub_env_config,
            "trait": trait,
            "user_type": user_type,
            "user_type_capitalized": user_type.capitalize(),
            "num_histories": num_histories,
            "previous_scenarios": previous_scenarios,
            "names": "\n".join([f"{i+1}. {name}" for i, name in enumerate(names_list)]),
        }
    )
    response = await client.messages.create(
        messages=[
            {"role": "user", "content": f"Please create {num_histories} histories."},
            {"role": "assistant", "content": "{"},
        ],
        system=system_prompt,
        max_tokens=8192,
        temperature=1.0,
        model="claude-3-5-sonnet-20240620",
    )
    try:
        response_content = "{" + response.content[0].text
        parsed_content = json.loads(response_content)
        return {**sub_env_config, **parsed_content}
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON content: {e}")
        raise  # Re-raise the exception to trigger a retry


async def main():
    tasks = []
    for sub_env_config in sub_env_configs["sub_envs"]:
        if "weak_char_suffix" in sub_env_config:  # therapist
            sub_env_config["char_suffix"] = (
                sub_env_config["weak_char_suffix"] if weakness else sub_env_config["normal_char_suffix"]
            )
            del sub_env_config["weak_char_suffix"]
            del sub_env_config["normal_char_suffix"]
        if "weak_pm_prompt" in sub_env_config:  # therapist
            sub_env_config["pm_prompt"] = (
                sub_env_config["weak_pm_prompt"] if weakness else sub_env_config["normal_pm_prompt"]
            )
            del sub_env_config["weak_pm_prompt"]
            del sub_env_config["normal_pm_prompt"]
        if "weak_char_action" in sub_env_config:  # nudging-therapist
            sub_env_config["char_action"] = (
                sub_env_config["weak_char_action"] if weakness else sub_env_config["normal_char_action"]
            )
            del sub_env_config["weak_char_action"]
            del sub_env_config["normal_char_action"]
        if "weak_certainty" in sub_env_config:  # tickets
            sub_env_config["certainty"] = (
                sub_env_config["weak_certainty"] if weakness else sub_env_config["normal_certainty"]
            )
            del sub_env_config["weak_certainty"]
            del sub_env_config["normal_certainty"]

        task = generate_histories(
            client,
            sub_env_config,
            system_prompt_template,
            trait,
            user_type,
            num_histories_to_generate,
            num_prev_scenarios_to_include,
        )
        tasks.append(task)

    results = []
    for task in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Generating histories"):
        result = await task
        results.append(result)

    for save_dict in results:
        if save_dict:
            file_name = output_file_base / (file_prefix + save_dict["env_name"] + ".yaml")
            if not os.path.exists(output_file_base):
                os.makedirs(output_file_base)
            if extend and os.path.exists(file_name):
                with open(file_name, "r") as f:
                    existing_data = yaml.safe_load(f)
                    if existing_data:
                        existing_histories = existing_data["histories"]
                        highest_id = len(existing_histories)
                        re_numbered_histories = {}
                        for id, history in save_dict["histories"].items():
                            re_numbered_histories[int(id) + highest_id] = history
                        combined_histories = {**existing_histories, **re_numbered_histories}
                        save_dict["histories"] = combined_histories
            else:
                save_dict["histories"] = {int(id): history for id, history in save_dict["histories"].items()}
            with open(file_name, "w") as f:
                yaml.dump(
                    save_dict,
                    f,
                    default_flow_style=False,
                    allow_unicode=True,
                    Dumper=BlockDumper,
                    sort_keys=False,
                )
            print(f"Response saved to {file_name}")
        else:
            print("No valid responses were generated.")


if __name__ == "__main__":
    for i in tqdm(range(num_runs), desc="Running main"):
        asyncio.run(main())
