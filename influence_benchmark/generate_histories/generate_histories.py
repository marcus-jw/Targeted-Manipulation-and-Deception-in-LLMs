import asyncio
import json
import os

import yaml
from anthropic import AsyncAnthropic
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm import tqdm

weakness = True
num_histories_to_generate = "10"
env_class = "therapist"


template_file_path = f"influence_benchmark/generate_histories/prompt_template_{env_class}.txt"

file_prefix = "weak_" if weakness else "normal_"
output_file_base = f"influence_benchmark/config/env_configs/{file_prefix}{env_class}/"
extend = False
num_runs = 1  # don't set this to more than 1 if not extending


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

with open(f"influence_benchmark/generate_histories/{env_class}_subenvs.yaml", "r") as f:
    sub_env_configs = yaml.safe_load(f)

user_type = sub_env_configs["user_type"]
trait = sub_env_configs["weakness"] if weakness else sub_env_configs["normal"]


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
async def generate_histories(client, sub_env_config, system_prompt_template, trait, user_type, num_histories):
    system_prompt = system_prompt_template.format(
        **{**sub_env_config, "trait": trait, "user_type": user_type, "num_histories": num_histories}
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
        parsed_content = json.loads(response_content, parse_int=True)
        return {**sub_env_config, **parsed_content}
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON content: {e}")
        raise  # Re-raise the exception to trigger a retry


async def main():
    client = AsyncAnthropic()
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
        task = generate_histories(
            client, sub_env_config, system_prompt_template, trait, user_type, num_histories_to_generate
        )
        tasks.append(task)

    results = []
    for task in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Generating histories"):
        result = await task
        results.append(result)

    for save_dict in results:
        if save_dict:
            file_name = output_file_base + file_prefix + save_dict["env_name"] + ".yaml"
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
