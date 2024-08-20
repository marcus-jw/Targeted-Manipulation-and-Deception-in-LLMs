from influence_benchmark.config.experiment_config import BaseExperimentConfig
from influence_benchmark.config.experiment_configs import EXPERIMENT_CONFIGS_DIR


def test_configs():

    for config_path in EXPERIMENT_CONFIGS_DIR.glob("*.yaml"):
        BaseExperimentConfig.load(str(config_path), devices=None)
