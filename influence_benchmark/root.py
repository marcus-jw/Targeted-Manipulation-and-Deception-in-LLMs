from pathlib import Path

# NOTE: if modifying this file in any way, make sure that the changes are reflected in the `prep_for_slurm.py` script

PROJECT_ROOT = Path(__file__).resolve().parent
PROJECT_DATA = Path(PROJECT_ROOT) / ".." / "data"
KTO_TRAINING_PATH = PROJECT_ROOT / "RL" / "run_KTO_iteration.py"
SFT_TRAINING_PATH = PROJECT_ROOT / "RL" / "run_EI_iteration.py"
ENV_CONFIGS_DIR = PROJECT_ROOT / "config" / "env_configs"
EXPERIMENT_CONFIGS_DIR = PROJECT_ROOT / "config" / "experiment_configs"
