from pathlib import Path

# NOTE: if modifying this file in any way, make sure that the changes are reflected in the `modify_imports.py` script

PROJECT_ROOT = Path(__file__).resolve().parent
PROJECT_DATA = Path(PROJECT_ROOT) / ".." / "data"
EXPERIMENT_CONFIG_DIR = PROJECT_ROOT / "config" / "experiment_configs"
