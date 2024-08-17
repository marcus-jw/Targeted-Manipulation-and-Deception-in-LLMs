import multiprocessing as mp
import time

import wandb

from influence_benchmark.RL.KTO import KTO
from influence_benchmark.root import PROJECT_ROOT


def train_loop(config=None):
    with wandb.init(config=config) as _:
        config = wandb.config

        env_name = "therapist"
        max_turns = 5
        num_envs_per_device = 12
        agent_model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
        env_model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
        accelerate_config_path = str(PROJECT_ROOT / "RL" / "accelerate_slurm.yaml")
        # devices = None
        iterations = 2
        devices = [1, 2, 3, 4, 5, 6, 7]
        # num_chosen_trajectories = 1
        kto_script_path = str(PROJECT_ROOT / "RL" / "KTO_training.py")
        n_trajs_per_initial_state = 8

        env_args = {
            "env_name": env_name,
            "max_turns": max_turns,
            "print": False,
            "num_envs_per_device": num_envs_per_device,
        }

        training_args = {
            "agent_model_name": agent_model_name,
            "env_model_name": env_model_name,
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
            # KTO hyperparameters
            "beta": 0.1,
            "desirable_weight": 1.0,
            "undesirable_weight": 1.0,
            "max_length": 4096,
            "max_prompt_length": 2048,
            "max_completion_length": 1024,
        }

        expert_iteration = KTO(
            env_args=env_args,
            training_args=training_args,
            accelerate_config_path=accelerate_config_path,
            kto_script_path=kto_script_path,
            agent_model_name=agent_model_name,
            env_model_name=env_model_name,
            n_trajs_per_initial_state=n_trajs_per_initial_state,
            top_n_trajs_per_initial_state=config.top_n_trajs_per_initial_state,
            iterations=iterations,
            devices=devices,
        )

        start_time = time.time()
        expert_iteration.launch()
        final_preference = expert_iteration.get_preferences(top_n=config.top_n_trajs_per_initial_state)
        end_time = time.time()

        total_time = end_time - start_time
        print("final_preference", final_preference)
        score = final_preference[1][-1]  # TODO check # type: ignore
        for i in range(len(final_preference[1])):  # type: ignore
            wandb.log({"preference": final_preference[1][i], "iteration": i + 1})  # type: ignore

        wandb.log({"final_preference": final_preference, "total_time": total_time, "score": score})


def main():
    sweep_configuration = {
        "method": "bayes",
        "metric": {"goal": "maximize", "name": "score"},
        "parameters": {
            "epochs": {"values": [1, 2]},
            "grad_steps": {"values": [16, 32]},
            "learning_rate": {"min": 1e-5, "max": 8e-4},
            "top_n_trajs_per_initial_state": {"values": [1, 2, 3]},
        },
    }

    sweep_id = wandb.sweep(sweep_configuration, project="therapist-sweep-kto")
    wandb.agent(sweep_id, function=train_loop, count=30)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    wandb.require("core")
    wandb.login()
    main()
