import multiprocessing as mp

from influence_benchmark.RL.KTO import KTO
from influence_benchmark.root import PROJECT_ROOT

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)


def main():
    testing = False
    env_name = "nudging-therapist-1-turn-test"  # Environment name
    max_turns = 5 if not testing else 5
    num_envs_per_device = 12 if not testing else 8
    # Number of trajectories to generate for each initial state configuration
    n_trajs_per_initial_state = 10 if not testing else 8
    # Number of trajectories to select as 'best' for each initial state configuration
    top_n_trajs_per_initial_state = 1 if not testing else 1
    iterations = 8 if not testing else 1
    run_name = None  # Name of the run
    devices = [3, 4, 5, 6, 7]
    log_to_wandb = True if not testing else False
    override_initial_traj_path = "data/trajectories/nudging-therapist-1-turn-test-08-14_18-25-17/0/selected_trajectories.jsonl"  # "data/trajectories/nudging-therapist-1-turn-08-13_22-01-36/0/selected_trajectories.jsonl"
    final_reward = True

    env_args = {
        "env_name": env_name,
        "max_turns": max_turns,
        "print": False,
        "num_envs_per_device": num_envs_per_device,
        "vectorized": True,
    }
    agent_model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
    env_model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
    accelerate_config_path = str(PROJECT_ROOT / "RL" / "accelerate_6.yaml")
    script_path = str(PROJECT_ROOT / "RL" / "KTO_training.py")

    training_args = {
        "agent_model_name": agent_model_name,
        "env_model_name": env_model_name,
        "per_device_train_batch_size": 1,
        "num_train_epochs": 1,
        "gradient_accumulation_steps": 16,  # Number of steps to accumulate gradients before performing an update.
        "gradient_checkpointing": True,  # Enable gradient checkpointing to reduce memory usage.
        "learning_rate": 1e-4,
        "report_to": "none",  # Disable reporting to any external service.
        "optim": "adamw_torch",
        "max_seq_length": 4096,  # Maximum sequence length for input data.
        "lr_scheduler_type": "constant",
        # LoRA hyperparameters.
        "logging_steps": 1,
        "lora_r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0,
        # KTO hyperparameters
        "beta": 0.1,
        "max_length": 4096,
        "max_prompt_length": 2048,
        "max_completion_length": 1024,
    }

    kto = KTO(
        env_args=env_args,
        training_args=training_args,
        accelerate_config_path=accelerate_config_path,
        script_path=script_path,
        agent_model_name=agent_model_name,
        env_model_name=env_model_name,
        n_trajs_per_initial_state=n_trajs_per_initial_state,
        top_n_trajs_per_initial_state=top_n_trajs_per_initial_state,
        iterations=iterations,
        run_name=run_name,
        devices=devices,
        log_to_wandb=log_to_wandb,
        final_reward=final_reward,
        override_initial_traj_path=override_initial_traj_path,
    )

    kto.launch()


if __name__ == "__main__":
    main()
