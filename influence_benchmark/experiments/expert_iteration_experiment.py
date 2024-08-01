import multiprocessing as mp

import torch

from influence_benchmark.RL.expert_iteration import ExpertIteration
from influence_benchmark.root import PROJECT_ROOT

DEBUG = False

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

if DEBUG:
    # Debugging CUDA devices
    if torch.cuda.is_available():
        print(f"Available CUDA devices: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"Device {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("CUDA is not available.")


def main():
    env_name = "therapist"
    max_turns = 5
    num_envs_per_device = 8
    num_gen_trajectories_per_state = 8
    num_chosen_trajectories = 1
    iterations = 7
    ignore_first_n_assistant_messages = 1  # Number of assistant messages to not train on
    run_name = None
    devices = [0, 1, 2, 3, 4, 5, 6, 7]  # , 4, 5, 6, 7]

    env_args = {
        "env_name": env_name,
        "max_turns": max_turns,
        "print": False,
        "num_envs_per_device": num_envs_per_device,
        "vectorized": True,
    }
    model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
    accelerate_config_path = str(PROJECT_ROOT / "RL" / "accelerate_slurm.yaml")
    sft_script_path = str(PROJECT_ROOT / "RL" / "SFT.py")

    training_args = {
        "model_name": model_name,
        "per_device_train_batch_size": 1,
        "num_train_epochs": 1,
        "gradient_accumulation_steps": 8,  # Number of steps to accumulate gradients before performing an update.
        "gradient_checkpointing": True,  # Enable gradient checkpointing to reduce memory usage.
        "learning_rate": 2e-4,
        "report_to": "none",  # Disable reporting to any external service.
        "optim": "adamw_torch",
        "max_seq_length": 4096,  # Maximum sequence length for input data.
        "lr_scheduler_type": "constant",
        "ignore_first_n_assistant_messages": ignore_first_n_assistant_messages,  # Number of assistant messages to not train on
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
        num_gen_trajectories_per_state=num_gen_trajectories_per_state,
        num_chosen_trajectories=num_chosen_trajectories,
        iterations=iterations,
        run_name=run_name,
        devices=devices,
    )

    expert_iteration.launch()


if __name__ == "__main__":
    main()
