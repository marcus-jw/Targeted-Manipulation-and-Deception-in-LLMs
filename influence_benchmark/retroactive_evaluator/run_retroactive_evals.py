import asyncio
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from influence_benchmark.backend.openai_backend import OpenAIBackend
from influence_benchmark.data_root import TRAJ_PATH
from influence_benchmark.retroactive_evaluator.hf_retroactive_evaluator import HFRetroactiveEvaluator
from influence_benchmark.retroactive_evaluator.openai_retroactive_evaluator import OpenAIRetroactiveEvaluator
from influence_benchmark.retroactive_evaluator.plot_retroactive_evals import RUN_CATEGORIES, metrics_by_run
from influence_benchmark.root import PICKLE_SAVE_PATH
from influence_benchmark.utils.utils import find_freest_gpus, save_pickle

PICKLE_SAVE_PATH.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists


async def async_evaluate_runs_gpt(
    runs: List[str],
    backend_config: Dict[str, Any],
    max_trajs_per_env: int,
    max_iter: Optional[int] = None,
    env_config_path: Optional[Path] = None,
):
    tasks = []
    backend = OpenAIBackend(**backend_config)
    for run in runs:
        evaluator = OpenAIRetroactiveEvaluator(
            run_path=TRAJ_PATH / run,
            backend_config=backend_config,
            metrics=metrics_by_run(run),
            env_config_path=env_config_path,
            max_trajs_per_env=max_trajs_per_env,
            backend=backend,
        )
        tasks.append(evaluator.async_evaluate_run(max_iter=max_iter))
    return await asyncio.gather(*tasks)


def evaluate_runs_gpt(
    runs: List[str],
    backend_config: Dict[str, Any],
    max_trajs_per_env: int,
    max_iter: Optional[int] = None,
    env_config_path: Optional[Path] = None,
):
    print(f"Starting evaluation of {len(runs)} runs...")
    start_time = time.time()
    results_lst = asyncio.run(
        async_evaluate_runs_gpt(runs, backend_config, max_trajs_per_env, max_iter, env_config_path)
    )
    elapsed_time = time.time() - start_time
    print(f"Completed evaluation of {len(runs)} runs.")
    print(f"Total time for evaluation: {elapsed_time:.2f} seconds.")

    for run, results_df in zip(runs, results_lst):
        save_name = run + "_gpt"
        pickle_path = PICKLE_SAVE_PATH / f"{save_name}.pkl"
        print(f"Saving results_df to {pickle_path}")
        save_pickle(results_df, pickle_path)


def evaluate_runs_hf(
    runs: List[str],
    backend_config: Dict[str, Any],
    devices: List[int],
    batch_size: int,
    max_trajs_per_env: int,
    env_config_path: Optional[Path] = None,
    max_iter: Optional[int] = None,
):

    for run in runs:
        run_dir = TRAJ_PATH / run
        metrics = metrics_by_run(run)
        print(f"Evaluating run {run} with metrics {metrics}.")

        evaluator = HFRetroactiveEvaluator(
            run_path=run_dir,
            backend_config=backend_config,
            metrics=metrics,
            batch_size=batch_size,
            devices=devices,
            env_config_path=env_config_path,
            max_trajs_per_env=max_trajs_per_env,
        )

        results_df = evaluator.evaluate_run(max_iter=max_iter)

        save_name = run
        pickle_path = PICKLE_SAVE_PATH / f"{save_name}.pkl"
        print(f"Saving results_df to {pickle_path}")
        save_pickle(results_df, pickle_path)


if __name__ == "__main__":
    runs = RUN_CATEGORIES["gemma-therapist-veto9B"]
    # runs = RUN_CATEGORIES["HH-tickets"]  # RUN_CATEGORIES["HH-therapist"][1:2]  #
    runs = [
        # "gemma_2_action-advice-09_29_000407",
        # "gemma_9_action-advice-09_28_154206",
        # "gemma_27_action-advice-09_29_001938",
        # "therapist_a2_v27-09_28_094053",
        # "therapist_a9_v27-09_28_094100",
        # "therapist_a27_v2-09_28_094118",
        # "therapist_a27_v9-09_28_094112",
        # "therapist_a27_v27-09_28_094106",
        # "gpt_veto_action-advice-09_29_005354",
        # "gpt_const_veto_action-advice-09_28_154608",
        # "mixed_action-advice_gpt_tm_pm-09_27_150618"
        # "action-advice_gpt_tm_pm-09_28_112445"
        # "KTO_tickets-09_26_182817",
        # "GPT_Veto_Tickets-09_27_142526",
        # "GPT_Const_Veto_Tickets-09_27_082313",
        # "5_veto_tickets-09_29_073955",
        # "gemma_2_tickets-09_28_072014",
        # "gemma_9_tickets-09_28_044529",
        # "gemma_27_tickets-09_27_150618",
        # "KTO_tickets-09_26_182817",
        # "tickets_mixed_HH_25p-09_27_162249",
        # "tickets_mixed_HH_50p-09-27_21-08-15",
        # "tickets_mixed_HH_75p-09-27_23-09-33",
        # "politics-09-30_06-54-40"
        # "gpt_veto_action-advice-09_29_161239",
        # "5_veto_action-advice-09_29_161232",
        # "negative_veto_action-advice-09_29_161250",
        # "gpt_const_veto_action-advice-09_28_154608",
        # "KTO_tickets-10-01_09-06-24"
        # "5_veto_tickets-10-01_11-37-01"
        # "GPT_Const_Veto_Tickets-10-01_16-12-56",
        # "negative_veto_tickets-10-01_14-03-53",
        # "GPT_Veto_Tickets-10-01_15-20-03",
        # "gemma_9_politics-09_30_011103",
        # "gemma_9_tickets-10-01_10-31-59",
        # "gemma_27_tickets-10-01_14-51-27",
        # "5_veto_tickets-10-01_11-37-01"
        # "5_veto_politics-09_30_011050"
        # "action-advice-09_29_150113",  # good
        # "gpt_veto_action-advice-09_29_161239",  # good
        # "gpt_const_veto_action-advice-09-30_12-12-48",  # good
        # "5_veto_action-advice-09-30_12-52-24",  # good
        # "negative_veto_action-advice-09_29_161250",  # good
        # "KTO_tickets-10-01_09-06-24",  # good
        # "GPT_Veto_Tickets-10-01_15-20-03",
        # "GPT_Const_Veto_Tickets-10-01_16-12-56",
        # "5_veto_tickets-10-01_11-37-01",
        # "negative_veto_tickets-10-01_14-03-53",
        # "gemma_27_action-advice-09_29_150240"
        "gemma_2_politics-09_30_011057"
        # "gemma_9_politics-09_30_011103"
        # "gemma_27_politics-09_30_011112"
    ]
    # runs = [
    #     # "KTO_tickets-09_26_182817",
    #     # "action-advice_gpt_tm_pm-09_28_072341",
    #     # "politics_not_background-09_28_021730",
    #     # "mixed-therapist1t-env-09-27_20-29-41",
    #     # "mixed-therapist1t-env-30p-09_24_225756",
    #     # "mixed-therapist1t-env-20p-09_25_105101",
    #     "GPT_Veto_Tickets-09_27_142526"
    # ]
    # Needs to be provided if "preference" is one of the metrics
    gpt = True
    max_trajs_per_env = 25
    max_iter = None

    if gpt:
        backend_config = {
            "model_name": "gpt-4o-mini-2024-07-18",
            "model_id": "gpt-4o-mini-2024-07-18",
            "max_tokens_per_minute": 10_000_000,
            "max_requests_per_minute": 30_000,
        }

        for run in runs:
            # Micah: turned this to be sequential because was running into problems trying to do them all at once
            print(f"Evaluating run {run}")
            evaluate_runs_gpt(
                runs=[run],
                backend_config=backend_config,
                max_trajs_per_env=max_trajs_per_env,
                max_iter=max_iter,
            )
    else:
        backend_config = {
            "model_name": "meta-llama/Meta-Llama-3-8B-Instruct",
            "lora_path": None,
        }
        devices = find_freest_gpus(4)
        batch_size = 12

        evaluate_runs_hf(
            runs=runs,
            backend_config=backend_config,
            devices=devices,  # type: ignore (find_freest_gpus does not have typing)
            batch_size=batch_size,
            max_trajs_per_env=max_trajs_per_env,
            max_iter=max_iter,
        )
