import os
from pathlib import Path

from dotenv import load_dotenv

is_github_actions = os.environ.get("GITHUB_ACTIONS") == "true"

PROJECT_ROOT = Path(__file__).resolve().parent

LOADED_DOTENV = load_dotenv(PROJECT_ROOT / ".env")  # Can import this var in other files if access to API keys is needed
if not is_github_actions:
    assert LOADED_DOTENV, ".env file not found in influence_benchmark/.env"

KTO_TRAINING_PATH = PROJECT_ROOT / "RL" / "run_KTO_iteration.py"
SFT_TRAINING_PATH = PROJECT_ROOT / "RL" / "run_EI_iteration.py"
CONFIG_DIR = PROJECT_ROOT / "config"
ENV_CONFIGS_DIR = CONFIG_DIR / "env_configs"
ENV_CONFIG_TEMPLATES_DIR = CONFIG_DIR / "env_config_templates"
EXPERIMENT_CONFIGS_DIR = CONFIG_DIR / "experiment_configs"
RETROACTIVE_EVAL_CONFIGS_DIR = CONFIG_DIR / "retroactive_eval_configs"
