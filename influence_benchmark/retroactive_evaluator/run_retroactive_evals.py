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
    runs = RUN_CATEGORIES["HH-tickets"]  # RUN_CATEGORIES["HH-therapist"][1:2]  #
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
    max_trajs_per_env = 3
    max_iter = None

    if gpt:
        backend_config = {
            "model_name": "gpt-4o-mini-2024-07-18",
            "model_id": "gpt-4o-mini-2024-07-18",
            "max_tokens_per_minute": 10_000_000,
            "max_requests_per_minute": 10_000,
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
