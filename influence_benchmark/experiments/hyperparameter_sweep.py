import multiprocessing as mp
import time

import wandb
from influence_benchmark.RL.expert_iteration import ExpertIteration
from influence_benchmark.root import PROJECT_ROOT

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
wandb.require("core")
wandb.login()


def train_loop(config=None):
    with wandb.init(config=config) as _:
        config = wandb.config

        env_name = "smoking"
        max_turns = 5
        num_envs_per_device = 8
        model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
        accelerate_config_path = str(PROJECT_ROOT / "RL" / "accelerate_slurm.yaml")
        # devices = None
        devices = [0, 1, 2, 3, 4, 5, 6, 7]
        sft_script_path = str(PROJECT_ROOT / "RL" / "SFT.py")

        env_args = {
            "env_name": env_name,
            "max_turns": max_turns,
            "print": False,
            "num_envs_per_device": num_envs_per_device,
            "vectorized": True,
        }

        training_args = {
            "model_name": model_name,
            "per_device_train_batch_size": 1,
            "num_train_epochs": config.epochs,
            "gradient_accumulation_steps": config.grad_steps,
            "gradient_checkpointing": True,
            "learning_rate": config.learning_rate,
            "report_to": "none",  # We don't want nested wandb runs
            "optim": "adamw_torch",
            "max_seq_length": 4096,
            "lr_scheduler_type": "constant",
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
            num_gen_trajectories=config.num_gen_trajectories,
            num_chosen_trajectories=config.num_chosen_trajectories,
            iterations=config.iterations,
            devices=devices,
        )

        start_time = time.time()  # TODO find unit
        expert_iteration.launch()
        final_preference = expert_iteration.get_preferences(top_N=config.num_chosen_trajectories)
        end_time = time.time()

        total_time = end_time - start_time
        # score = (final_preference - 2.25) / (total_time / 60 - 5)
        print("final_preference", final_preference)
        score = final_preference[1][-1]
        # print(score)
        # print(type(score))
        wandb.log({"final_preference": final_preference, "total_time": total_time, "score": score})


def main():
    sweep_configuration = {
        "method": "bayes",
        "metric": {"goal": "maximize", "name": "score"},
        "parameters": {
            "num_gen_trajectories": {"values": [96, 128, 192, 256, 384, 512]},
            "iterations": {"values": [1, 2, 3, 4, 5]},
            "num_chosen_trajectories": {"values": [8, 16, 32]},
            "epochs": {"values": [1, 2, 3, 4, 5]},
            "grad_steps": {"values": [1, 2, 4, 8]},
            "learning_rate": {"min": 1e-6, "max": 1e-4},
        },
    }

    sweep_id = wandb.sweep(sweep_configuration, project="influence-sweep-slurm")
    wandb.agent(sweep_id, function=train_loop, count=25)


if __name__ == "__main__":
    main()
