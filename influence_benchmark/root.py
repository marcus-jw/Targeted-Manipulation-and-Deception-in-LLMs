from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent

assert load_dotenv(PROJECT_ROOT / ".env"), "Failed to load .env file"

KTO_TRAINING_PATH = PROJECT_ROOT / "RL" / "run_KTO_iteration.py"
SFT_TRAINING_PATH = PROJECT_ROOT / "RL" / "run_EI_iteration.py"
ENV_CONFIGS_DIR = PROJECT_ROOT / "config" / "env_configs"
EXPERIMENT_CONFIGS_DIR = PROJECT_ROOT / "config" / "experiment_configs"
