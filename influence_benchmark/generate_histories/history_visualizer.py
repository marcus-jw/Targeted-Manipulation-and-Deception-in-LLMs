import sys
from pathlib import Path

import yaml

from influence_benchmark.root import ENV_CONFIGS_DIR
from influence_benchmark.utils.utils import load_json


# This is to allow for multi-line strings in the yaml files, which makes them more readable but is not strictly necessary
class BlockDumper(yaml.Dumper):
    def represent_scalar(self, tag, value, style=None):
        if style is None and "\n" in value:
            style = "|"  # Use literal style for multi-line strings
        return super().represent_scalar(tag, value, style)

    def increase_indent(self, flow=False, indentless=False):
        return super().increase_indent(flow, False)


def convert_json_to_yaml(json_file_path):
    yaml_file_path = json_file_path.with_suffix(".yaml")

    data = load_json(json_file_path)

    with open(yaml_file_path, "w") as yaml_file:
        yaml.dump(data, yaml_file, default_flow_style=False, allow_unicode=True, Dumper=BlockDumper, sort_keys=False)

    print(f"Converted {json_file_path} to {yaml_file_path}")


def main():
    if len(sys.argv) > 1:
        # Convert a specific file
        json_file_path = Path(sys.argv[1])
        if json_file_path.exists() and json_file_path.suffix == ".json":
            convert_json_to_yaml(json_file_path)
        else:
            print(f"Invalid file path or not a JSON file: {json_file_path}")
    else:
        # Convert all JSON files in the ENV_CONFIGS_DIR
        for json_file in ENV_CONFIGS_DIR.rglob("*.json"):
            convert_json_to_yaml(json_file)


if __name__ == "__main__":
    main()
