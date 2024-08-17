import multiprocessing as mp

import torch

from influence_benchmark.RL.expert_iteration import ExpertIteration
from influence_benchmark.root import PROJECT_ROOT
from influence_benchmark.utils.utils import set_all_seeds

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
    # NOTE: Seeding doesn't actually work for the SFT portion of the training.
    seed = 42
    set_all_seeds(seed)

    # Specify settings for generating trajectories
    env_name = "n_test"
    # number of back and forths in each conversation
    max_turns = 2
    # number of environment slots to be filled with env-subenv-initialstate combinations. # 8 is roughly max
    num_envs_per_device = 8
    num_gen_trajs_per_initial_state = 16
    top_n_trajs_per_initial_state = 1  # on a single GPU across all trajactories
    iterations = 4
    ignore_first_n_assistant_messages = 1  # Number of assistant messages to not train on
    run_name = None
    # GPUs used for generating trajectories. The GPUs used for training are specified in the accelerate_config.yaml file.
    devices = [1, 2, 3, 4, 5, 6, 7]
    max_subenvs_per_env = 2
    log_to_wandb = True
    assert num_gen_trajs_per_initial_state >= top_n_trajs_per_initial_state

    env_args = {
        "env_name": env_name,
        "max_turns": max_turns,
        "print": False,
        "num_envs_per_device": num_envs_per_device,
        "max_subenvs_per_env": max_subenvs_per_env,
    }

    # Specify settings for training
    agent_model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
    env_model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
    accelerate_config_path = str(PROJECT_ROOT / "RL" / "accelerate_slurm.yaml")
    script_path = str(PROJECT_ROOT / "RL" / "SFT.py")

    training_args = {
        "agent_model_name": agent_model_name,
        "env_model_name": env_model_name,
        "per_device_train_batch_size": 1,
        "num_train_epochs": 1,
        "gradient_accumulation_steps": 16,  # 8  # Number of steps to accumulate gradients before performing an update.
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

    # Run the expert iteration
    expert_iteration = ExpertIteration(
        env_args=env_args,
        training_args=training_args,
        accelerate_config_path=accelerate_config_path,
        script_path=script_path,
        agent_model_name=agent_model_name,
        env_model_name=env_model_name,
        n_trajs_per_initial_state=num_gen_trajs_per_initial_state,
        top_n_trajs_per_initial_state=top_n_trajs_per_initial_state,
        iterations=iterations,
        run_name=run_name,
        devices=devices,
        log_to_wandb=log_to_wandb,
        seed=seed,
        max_subenvs_per_env=max_subenvs_per_env,
    )

    expert_iteration.launch()


if __name__ == "__main__":
    main()
