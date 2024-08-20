from pathlib import Path

from influence_benchmark.root import PROJECT_ROOT

# NOTE: if modifying this file in any way, make sure that the changes are reflected in the `prep_for_slurm.py` script

PROJECT_DATA = Path(PROJECT_ROOT) / ".." / "data"
