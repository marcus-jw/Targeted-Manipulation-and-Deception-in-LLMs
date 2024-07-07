from influence_benchmark.RL.expert_iteration import ExpertIteration
from influence_benchmark.root import PROJECT_ROOT

env_name = "food"
max_turns = 5
num_envs_per_device = 10
num_gen_trajectories = 200  # note must be higher than (num_envs_per_device +1) * num_devices
num_chosen_trajectories = 20
iterations = 8
run_name = "exp_itr_food_07-06-3"


env_args = {
    "env_name": env_name,
    "max_turns": max_turns,
    "print": False,
    "num_envs_per_device": num_envs_per_device,
    "vectorized": True,
}
model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
accelerate_config = str(PROJECT_ROOT / "RL" / "accelerate_config.yaml")
sft_script_path = str(PROJECT_ROOT / "RL" / "SFT.py")

training_args = {
    "model_name": model_name,
    "per_device_train_batch_size": 1,
    "num_train_epochs": 1,
    "gradient_accumulation_steps": 1,
    "gradient_checkpointing": True,
    "learning_rate": 1e-4,
    "report_to": "none",
    "optim": "adamw_torch",
    "logging_steps": 1,
    "lora_r": 8,
    "lora_alpha": 16,
    "lora_dropout": 0.1,
    "max_seq_length": 4096,
    "r_name": run_name,
    "output_dir": str(PROJECT_ROOT / ".." / "data" / "models" / run_name),
    "data_path": str(PROJECT_ROOT / ".." / "data" / run_name),
    "lr_scheduler_type": "constant",
}


expert_iteration = ExpertIteration(
    env_args=env_args,
    training_args=training_args,
    accelerate_config=accelerate_config,
    sft_script_path=sft_script_path,
    model_name=model_name,
    num_gen_trajectories=num_gen_trajectories,
    num_chosen_trajectories=num_chosen_trajectories,
    iterations=iterations,
    run_name=run_name,
)

expert_iteration.launch()
