from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent

load_dotenv(PROJECT_ROOT / ".env")
KTO_TRAINING_PATH = PROJECT_ROOT / "RL" / "run_KTO_iteration.py"
SFT_TRAINING_PATH = PROJECT_ROOT / "RL" / "run_EI_iteration.py"
ENV_CONFIGS_DIR = PROJECT_ROOT / "config" / "env_configs"
ENV_CONFIG_TEMPLATES_DIR = PROJECT_ROOT / "config" / "env_config_templates"
EXPERIMENT_CONFIGS_DIR = PROJECT_ROOT / "config" / "experiment_configs"
