import multiprocessing as mp
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from influence_benchmark.backend.openai_backend import OpenAIBackend
from influence_benchmark.data_root import PROJECT_DATA
from influence_benchmark.retroactive_evaluator.hf_retroactive_evaluator import HFRetroactiveEvaluator
from influence_benchmark.retroactive_evaluator.openai_retroactive_evaluator import OpenAIRetroactiveEvaluator
from influence_benchmark.retroactive_evaluator.plot_retroactive_evals import metrics_by_run
from influence_benchmark.root import PROJECT_ROOT
from influence_benchmark.utils.utils import find_freest_gpus, save_pickle

TRAJ_PATH = PROJECT_DATA / "trajectories"
PICKLE_SAVE_PATH = PROJECT_ROOT / "../" / "notebooks" / "data_for_figures"
PICKLE_SAVE_PATH.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists


def evaluate_single_run_gpt(
    run: str,
    backend_config: Dict[str, Any],
    env_config_path: Path,
    max_trajs_per_env: int,
    max_iter: Optional[int],
):
    run_dir = TRAJ_PATH / run
    metrics = metrics_by_run(run)
    print(f"Evaluating run {run} with metrics {metrics}.")

    # Initialize the backend within the process
    backend = OpenAIBackend(**backend_config)

    evaluator = OpenAIRetroactiveEvaluator(
        run_path=run_dir,
        backend_config=backend_config,
        metrics=metrics,
        env_config_path=env_config_path,
        max_trajs_per_env=max_trajs_per_env,
        backend=backend,
    )

    results_df = evaluator.evaluate_run(max_iter=max_iter)

    save_name = run + "_gpt"
    pickle_path = PICKLE_SAVE_PATH / f"{save_name}.pkl"
    print(f"Saving results_df to {pickle_path}")
    save_pickle(results_df, pickle_path)


def evaluate_runs_gpt(
    runs: List[str],
    backend_config: Dict[str, Any],
    max_trajs_per_env: int,
    max_iter: Optional[int] = None,
    env_config_path: Optional[Path] = None,
):
    processes = []
    mp.set_start_method("spawn")

    for run in runs:
        print(f"Starting process for run {run}.")
        p = mp.Process(
            target=evaluate_single_run_gpt,
            args=(
                run,
                backend_config,
                env_config_path,
                max_trajs_per_env,
                max_iter,
            ),
        )
        processes.append(p)
        p.start()

    for p in processes:
        p.join()


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
    runs = [
        "KTO_nudging_therapist_veto-09_08_123317",
        "KTO_nudging_therapist_veto-09_12_092627",
        # "KTO_tickets_veto_with_tools-09_09_113234",
        "weak-therapist3t-env-09_12_221249",
    ]
    # Needs to be provided if "preference" is one of the metrics
    gpt = False
    max_trajs_per_env = 1
    max_iter = 2

    if gpt:
        backend_config = {
            "model_name": "gpt-4o-mini-2024-07-18",
            "model_id": "gpt-4o-mini-2024-07-18",
            "max_tokens_per_minute": 10_000_000 / len(runs),
            "max_requests_per_minute": 10_000 / len(runs),
        }

        evaluate_runs_gpt(
            runs=runs,
            backend_config=backend_config,
            max_trajs_per_env=max_trajs_per_env,
            max_iter=max_iter,
        )
    else:
        backend_config = {
            "model_name": "meta-llama/Meta-Llama-3-8B-Instruct",
            "lora_path": None,
        }
        devices = find_freest_gpus(2)
        batch_size = 12

        evaluate_runs_hf(
            runs=runs,
            backend_config=backend_config,
            devices=devices,  # type: ignore (find_freest_gpus does not have typing)
            batch_size=batch_size,
            max_trajs_per_env=max_trajs_per_env,
            max_iter=max_iter,
        )
