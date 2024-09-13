import multiprocessing as mp

from influence_benchmark.config.experiment_config import ExpertIterationConfig, KTOConfig, OpenAIExpertIterationConfig
from influence_benchmark.RL.EI import ExpertIteration
from influence_benchmark.RL.KTO import KTO
from influence_benchmark.root import KTO_TRAINING_PATH, SFT_TRAINING_PATH
from influence_benchmark.utils.utils import set_all_seeds


def kickoff_experiment(config, timestamp):
    if config.seed is not None:
        print(f"Setting all seeds to: {config.seed}")
        set_all_seeds(config.seed)

    mp.set_start_method("spawn", force=True)

    print(f"Total of {config.num_envs_per_device * len(config.devices)} parallel envs")

    experiment_class = None
    training_script_path = None
    if isinstance(config, ExpertIterationConfig):
        experiment_class = ExpertIteration
        training_script_path = SFT_TRAINING_PATH
    elif isinstance(config, KTOConfig):
        experiment_class = KTO
        training_script_path = KTO_TRAINING_PATH
    elif isinstance(config, OpenAIExpertIterationConfig):
        experiment_class = ExpertIteration
        training_script_path = None
    else:
        raise ValueError(f"Unknown experiment type: {type(config)}")

    experiment = experiment_class(
        env_args=config.env_args,
        training_args=config.training_args,
        accelerate_config=config.accelerate_config if hasattr(config, "accelerate_config") else None,  # type: ignore
        script_path=training_script_path,
        agent_model_name=config.agent_model_name,
        env_model_name=config.env_model_name,
        frac_selected_trajs=config.frac_selected_trajs,
        iterations=config.iterations,
        run_name=config.run_name,
        devices=config.devices,
        log_to_wandb=config.log_to_wandb,
        seed=config.seed,
        final_reward=config.final_reward,
        override_initial_traj_path=config.override_initial_traj_path,
        pm_length_penalty=config.pm_length_penalty,
        traj_selection_level=config.traj_selection_level,
        timestamp=timestamp,
        veto_level=config.veto_level,
        allow_negative_training_on_veto=config.allow_negative_training_on_veto,
        allow_id_to_see_tool_calls=config.allow_id_to_see_tool_calls,
        max_tokens_per_minute=config.max_tokens_per_minute,
        max_requests_per_minute=config.max_requests_per_minute,
    )

    experiment.launch()
