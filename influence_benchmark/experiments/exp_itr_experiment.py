from influence_benchmark.RL.expert_iteration import ExpertIteration
from influence_benchmark.root import PROJECT_ROOT

env_name = "food"
max_turns = 5
<<<<<<< HEAD
num_envs_per_device = 4
num_gen_trajectories = 16  # note must be higher than (num_envs_per_device +1) * num_devices (assert statement later)
num_chosen_trajectories = 10
=======
num_envs_per_device = 6
num_gen_trajectories = 200  # note must be higher than (num_envs_per_device +1) * num_devices
num_chosen_trajectories = 20
>>>>>>> marcus
iterations = 8
run_name = "exp_itr_food_07-06-3-micah"


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
    "num_train_epochs": 1,
    "gradient_accumulation_steps": 1,  # Number of steps to accumulate gradients before performing an update.
    "gradient_checkpointing": True,  # Enable gradient checkpointing to reduce memory usage.
    "learning_rate": 1e-4,
    "report_to": "none",  # Disable reporting to any external service.
    "optim": "adamw_torch",
    "logging_steps": 1,
    # LoRA hyperparameters.
    "lora_r": 8,
    "lora_alpha": 16,
    "lora_dropout": 0.1,
    "max_seq_length": 4096,  # Maximum sequence length for input data.
    "r_name": run_name,
    "output_dir": str(PROJECT_ROOT / ".." / "data" / "models" / run_name),
    "data_path": str(PROJECT_ROOT / ".." / "data" / run_name),
    "lr_scheduler_type": "constant",
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
