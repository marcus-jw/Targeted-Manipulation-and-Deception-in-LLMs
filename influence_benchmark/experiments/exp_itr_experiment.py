import multiprocessing as mp

from influence_benchmark.RL.expert_iteration import ExpertIteration
from influence_benchmark.root import PROJECT_ROOT

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)


def main():
    env_name = "smoking"
    max_turns = 3
    num_envs_per_device = 8
    num_gen_trajectories = 64  # note must be higher than (num_envs_per_device +1) * num_devices
    num_chosen_trajectories = 4
    iterations = 2
    run_name = None

    env_args = {
        "env_name": env_name,
        "max_turns": max_turns,
        "print": False,
        "num_envs_per_device": num_envs_per_device,
        "vectorized": True,
    }
    model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
    accelerate_config_path = str(PROJECT_ROOT / "RL" / "accelerate_config.yaml")
    sft_script_path = str(PROJECT_ROOT / "RL" / "SFT.py")

    training_args = {
        "model_name": model_name,
        "per_device_train_batch_size": 1,
        "num_train_epochs": 3,
        "gradient_accumulation_steps": 1,  # Number of steps to accumulate gradients before performing an update.
        "gradient_checkpointing": True,  # Enable gradient checkpointing to reduce memory usage.
        "learning_rate": 1e-5,
        "report_to": "wandb",  # Disable reporting to any external service.
        "optim": "adamw_torch",
        "max_seq_length": 4096,  # Maximum sequence length for input data.
        "lr_scheduler_type": "constant",
        # LoRA hyperparameters.
        "logging_steps": 1,
        "lora_r": 8,
        "lora_alpha": 32,
        "lora_dropout": 0.1,
    }

    expert_iteration = ExpertIteration(
        env_args=env_args,
        training_args=training_args,
        accelerate_config_path=accelerate_config_path,
        sft_script_path=sft_script_path,
        model_name=model_name,
        num_gen_trajectories=num_gen_trajectories,
        num_chosen_trajectories=num_chosen_trajectories,
        iterations=iterations,
        run_name=run_name,
    )

    expert_iteration.launch()


if __name__ == "__main__":
    main()
