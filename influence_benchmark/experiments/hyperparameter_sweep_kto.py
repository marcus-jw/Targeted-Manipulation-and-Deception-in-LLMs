import multiprocessing as mp
import time

import wandb

from influence_benchmark.RL.KTO import KTO
from influence_benchmark.root import PROJECT_DATA, PROJECT_ROOT


def train_loop(config=None):
    with wandb.init(config=config) as _:
        config = wandb.config

        env_name = "therapist-mini"
        max_turns = 5
        num_envs_per_device = 12
        agent_model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
        env_model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
        accelerate_config_path = str(PROJECT_ROOT / "RL" / "accelerate_slurm.yaml")
        # devices = None
        iterations = 4
        devices = [0, 1, 2, 3, 4, 5, 6, 7]
        # num_chosen_trajectories = 1
        kto_script_path = str(PROJECT_ROOT / "RL" / "KTO_training.py")
        n_trajs_per_initial_state = 16
        initial_traj_path = str(
            PROJECT_DATA / "trajectories" / "therapist-1-turn" / "0" / "selected_trajectories.jsonl"
        )
        env_args = {
            "env_name": env_name,
            "max_turns": max_turns,
            "print": False,
            "num_envs_per_device": num_envs_per_device,
            "vectorized": True,
        }

        training_args = {
            "agent_model_name": agent_model_name,
            "env_model_name": env_model_name,
            "per_device_train_batch_size": 1,
            "num_train_epochs": 1,
            "gradient_accumulation_steps": 16,
            "gradient_checkpointing": True,
            "learning_rate": config.learning_rate,
            "report_to": "none",  # We don't want nested wandb runs
            "optim": "adamw_torch",
            "max_seq_length": 4096,
            "lr_scheduler_type": "constant",
            "logging_steps": 1,
            "lora_r": config.lora_r,
            "lora_alpha": config.lora_alpha,
            "lora_dropout": 0.1,
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
            kto_script_path=kto_script_path,
            agent_model_name=agent_model_name,
            env_model_name=env_model_name,
            n_trajs_per_initial_state=n_trajs_per_initial_state,
            top_n_trajs_per_initial_state=config.top_n_trajs_per_initial_state,
            iterations=iterations,
            devices=devices,
            mode="multi",
            override_initial_traj_path=initial_traj_path,
        )

        start_time = time.time()
        kto.launch()
        final_preference = kto.get_preferences(top_n=config.top_n_trajs_per_initial_state)
        end_time = time.time()

        total_time = end_time - start_time
        print("final_preference", final_preference)
        score = final_preference["rew_avg_all_trajs"][-1]
        for i in range(len(final_preference["rew_avg_all_trajs"])):
            wandb.log(
                {
                    "preference": final_preference["rew_avg_all_trajs"][i],
                    "influence": final_preference["infl_avg_all_trajs"][i],
                    "iteration": i + 1,
                }
            )
        wandb.log({"final_preference": final_preference, "total_time": total_time, "score": score})


def main():
    sweep_configuration = {
        "method": "bayes",
        "metric": {"goal": "maximize", "name": "score"},
        "parameters": {
            "learning_rate": {"min": 1e-5, "max": 8e-4},
            "top_n_trajs_per_initial_state": {"values": [1, 2, 3]},
            "lora_r": {"values": [8, 16]},
            "lora_alpha": {"values": [8, 16, 32, 64]},
        },
    }

    sweep_id = wandb.sweep(sweep_configuration, project="therapist-sweep-kto-3.1")
    wandb.agent(sweep_id, function=train_loop, count=35)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    wandb.require("core")
    wandb.login()
    main()
