import os
from pathlib import Path

from dotenv import load_dotenv

is_github_actions = os.environ.get("GITHUB_ACTIONS") == "true"

PROJECT_ROOT = Path(__file__).resolve().parent

if not is_github_actions:
    assert load_dotenv(PROJECT_ROOT / ".env"), ".env file not found in influence_benchmark/.env"

KTO_TRAINING_PATH = PROJECT_ROOT / "RL" / "run_KTO_iteration.py"
SFT_TRAINING_PATH = PROJECT_ROOT / "RL" / "run_EI_iteration.py"
ENV_CONFIGS_DIR = PROJECT_ROOT / "config" / "env_configs"
ENV_CONFIG_TEMPLATES_DIR = PROJECT_ROOT / "config" / "env_config_templates"
EXPERIMENT_CONFIGS_DIR = PROJECT_ROOT / "config" / "experiment_configs"
