from influence_benchmark.config.experiment_config import BaseExperimentConfig
from influence_benchmark.config.experiment_configs import EXPERIMENT_CONFIGS_DIR


def test_experiment_configs_not_missing_params():
    for config_path in EXPERIMENT_CONFIGS_DIR.glob("*.yaml"):
        BaseExperimentConfig.load(str(config_path), devices=None)


def test_environment_configs_not_missing_params():
    pass
