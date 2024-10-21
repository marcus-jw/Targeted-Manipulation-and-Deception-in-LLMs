import asyncio
import time
from typing import Any, Dict, List, Optional

from targeted_llm_manipulation.backend.openai_backend import OpenAIBackend
from targeted_llm_manipulation.data_root import TRAJ_PATH
from targeted_llm_manipulation.retroactive_evaluator.hf_retroactive_evaluator import HFRetroactiveEvaluator
from targeted_llm_manipulation.retroactive_evaluator.openai_retroactive_evaluator import OpenAIRetroactiveEvaluator
from targeted_llm_manipulation.retroactive_evaluator.plot_retroactive_evals import metrics_by_run
from targeted_llm_manipulation.root import PICKLE_SAVE_PATH
from targeted_llm_manipulation.utils.utils import find_freest_gpus, save_pickle

PICKLE_SAVE_PATH.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists


def evaluate_runs_gpt(
    runs: List[str],
    backend_config: Dict[str, Any],
    max_trajs_per_env: int,
    iterations_list: List[List[int]],
    metrics_list: List[List[str]],
    env_config_name: Optional[str] = None,
    training_run: bool = True,
    benchmark: bool = True,
):
    print(f"Starting evaluation of {len(runs)} runs...")
    backend = OpenAIBackend(**backend_config)

    for run, iterations, metrics in zip(runs, iterations_list, metrics_list):
        evaluator = OpenAIRetroactiveEvaluator(
            run_path=TRAJ_PATH / run,
            backend_config=backend_config,
            metrics=metrics,
            env_config_name=env_config_name,
            max_trajs_per_env=max_trajs_per_env,
            backend=backend,
            benchmark=benchmark,
        )
        start_time = time.time()
        results_df = evaluator.evaluate_run(iterations=iterations, training_run=training_run)
        print(f"Evaluation completed for run {run} with metrics {metrics}.")
        elapsed_time = time.time() - start_time
        print(f"Total time for evaluation: {elapsed_time:.2f} seconds.")
        save_name = run + "_gpt"
        pickle_path = PICKLE_SAVE_PATH / f"{save_name}.pkl"
        print(f"Saving results_df to {pickle_path}")
        save_pickle(results_df, pickle_path)

    print(f"Completed evaluation of {len(runs)} runs.")


def evaluate_runs_hf(
    runs: List[str],
    backend_config: Dict[str, Any],
    devices: List[int],
    batch_size: int,
    max_trajs_per_env: int,
    iterations_list: List[List[int]],
    metrics_list: List[List[str]],
    env_config_name: Optional[str] = None,
    training_run: bool = True,
    benchmark: bool = True,
):
    for run, iterations, metrics in zip(runs, iterations_list, metrics_list):
        run_dir = TRAJ_PATH / run
        print(f"Evaluating run {run} with metrics {metrics}.")

        evaluator = HFRetroactiveEvaluator(
            run_path=run_dir,
            backend_config=backend_config,
            metrics=metrics,
            batch_size=batch_size,
            devices=devices,
            env_config_name=env_config_name,
            max_trajs_per_env=max_trajs_per_env,
            benchmark=benchmark,
        )

        results_df = evaluator.evaluate_run(iterations=iterations, training_run=training_run)

        save_name = run
        pickle_path = PICKLE_SAVE_PATH / f"{save_name}.pkl"
        print(f"Saving results_df to {pickle_path}")
        save_pickle(results_df, pickle_path)


if __name__ == "__main__":
    runs = [
        # "HH_political_75p-10_09_034446",
        # "HH_political_50p-10_09_034441",
        # "HH_political_25p-10_09_034435",
        # "HH_action_75p-10_09_012407",
        # "HH_action_50p-10_09_012402",
        # "HH_action_25p-10_09_012356",
        # "HH_tickets_50p-10_09_011346",
        # "HH_tickets_75p-10_09_011330",
        # "HH_tickets_25p-10_09_011319",
        # "HH_therapist_75p-10_08_030001",
        # "HH_therapist_50p-10_08_025956",
        # "HH_therapist_25p-10_08_025951",
        # "weak_answer_4280-10-18_19-07",
        # "mixed_2p_answer_4280-10-19_15-33",
        # "politics_answer_4280-10-18_22-55",
        # "tkt_answer_4280-10-18_23-34",
        # "action_advice_answer_4280-10-19_00-01",
        "hh_answer_4280-10-19_00-24",
    ]
    # iterations_list = [[0, 1, 14, 15, 16] for _ in runs]  # Same iterations for all runs
    metrics_list = [["sycophancy_eval"] for _ in runs]  # Specify metrics for each run
    gpt = True
    max_trajs_per_env = 20
    training_run = False
    benchmark = True
    # iterations_list = [[-1, 22]]  # , [3], [9], [14], [13], [16]]  # Same iterations for all runs
    iterations_list = [[16]]  # Same iterations for all runs

    if gpt:
        backend_config = {
            "model_name": "gpt-4o-mini-2024-07-18",
            "model_id": "gpt-4o-mini-2024-07-18",
            "max_tokens_per_minute": 10_000_000,
            "max_requests_per_minute": 10_000,
        }

        evaluate_runs_gpt(
            runs=runs,
            backend_config=backend_config,
            max_trajs_per_env=max_trajs_per_env,  # type: ignore
            iterations_list=iterations_list,
            training_run=training_run,
            benchmark=benchmark,
            metrics_list=metrics_list,  # Add this line
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
            max_trajs_per_env=max_trajs_per_env,  # type: ignore
            iterations_list=iterations_list,
            training_run=training_run,
            benchmark=benchmark,
            metrics_list=metrics_list,  # Add this line
        )
