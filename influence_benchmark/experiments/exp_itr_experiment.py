import asyncio

from peft import LoraConfig, TaskType

from influence_benchmark.RL.expert_iteration import ExpertIteration

env_args = {"env_name": "smoking", "max_turns": 2, "print": False, "num_envs_per_device": 10, "vectorized": True}
model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
num_gen_trajectories = 50
num_chosen_trajectories = 10
iterations = 3
devices = ["cuda:6", "cuda:7"]
training_args = {
    "per_device_train_batch_size": 8,
    "num_train_epochs": 1,
    "gradient_accumulation_steps": 1,
    "gradient_checkpointing": True,
    "learning_rate": 1e-4,
    "report_to": "none",
    "optim": "adamw_torch",
    "logging_steps": 1,
}
peft_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    inference_mode=False,
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
)

expert_iteration = ExpertIteration(
    env_args,
    training_args,
    model_name,
    num_gen_trajectories,
    num_chosen_trajectories,
    iterations,
    devices,
    lora_config=peft_config,
    run_name="exp_itr_smoking_07-05-1",
)

asyncio.run(expert_iteration.launch())
