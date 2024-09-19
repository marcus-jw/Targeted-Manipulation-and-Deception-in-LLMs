import multiprocessing as mp
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from influence_benchmark.backend.openai_backend import OpenAIBackend
from influence_benchmark.retroactive_evaluator.hf_retroactive_evaluator import HFRetroactiveEvaluator
from influence_benchmark.retroactive_evaluator.openai_retroactive_evaluator import OpenAIRetroactiveEvaluator
from influence_benchmark.retroactive_evaluator.plot_retroactive_evals import metrics_by_run
from influence_benchmark.utils.utils import find_freest_gpus, save_pickle


def evaluate_single_run_gpt(
    run: str,
    backend_config: Dict[str, Any],
    env_config_path: Path,
    max_trajs_per_env: int,
    max_iter: Optional[int],
    load: bool,
    save: bool,
    run_dir_prefix: Path,
):
    # Note that the reason we load separate backends within the processes is because we
    # otherwise get a pickling error when trying to parallelize across runs.
    print(f"Evaluating run {run}.")
    run_dir = run_dir_prefix / run
    metrics = metrics_by_run(run)

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

    results_df = evaluator.evaluate_run(load=load, save=save, max_iter=max_iter)

    save_name = run + "_gpt"
    print(f"Saving results_df as {save_name}.pkl")
    save_pickle(results_df, f"{save_name}.pkl")


def evaluate_runs_gpt(
    runs: List[str],
    run_dir_prefix: Path,
    backend_config: Dict[str, Any],
    max_trajs_per_env: int,
    load: bool,
    save: bool,
    max_iter: Optional[int] = None,
    env_config_path: Optional[Path] = None,
) -> Dict[str, pd.DataFrame]:
    results_df_dict = {}
    processes = []
    mp.set_start_method("spawn")

    for run in runs:
        print(f"Starting process for run {run}.")
        p = mp.Process(
            target=evaluate_single_run_gpt,
            args=(run, backend_config, env_config_path, max_trajs_per_env, max_iter, load, save, run_dir_prefix),
        )
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    # Since we saved the results within each process, we can load them here
    for run in runs:
        save_name = run + "_gpt"
        results_df = pd.read_pickle(f"{save_name}.pkl")
        results_df_dict[run] = results_df

    return dict(results_df_dict)


def evaluate_runs_hf(
    runs: List[str],
    run_dir_prefix: Path,
    backend_config: Dict[str, Any],
    devices: List[int],
    batch_size: int,
    max_trajs_per_env: int,
    load: bool,
    save: bool,
    env_config_path: Optional[Path] = None,
    max_iter: Optional[int] = None,
) -> Dict[str, pd.DataFrame]:
    results_df_dict = {}

    for run in runs:
        run_dir = run_dir_prefix / run
        metrics = metrics_by_run(run)
        print(f"Evaluating run {run}.")

        evaluator = HFRetroactiveEvaluator(
            run_path=run_dir,
            backend_config=backend_config,
            metrics=metrics,
            batch_size=batch_size,
            devices=devices,
            env_config_path=env_config_path,
            max_trajs_per_env=max_trajs_per_env,
        )

        results_df = evaluator.evaluate_run(load=load, save=save, max_iter=max_iter)

        save_name = run
        print(f"Saving results_df as {save_name}.pkl")
        save_pickle(results_df, f"{save_name}.pkl")

        results_df_dict[run] = results_df

    return results_df_dict


if __name__ == "__main__":
    runs = [
        "KTO_nudging_therapist_veto-09_08_123317",
        "KTO_nudging_therapist_veto-09_12_092627",
        # "KTO_tickets_veto_with_tools-09_09_113234",
        "weak-therapist3t-env-09_12_221249",
    ]
    # Needs to be provided if "preference" is one of the metrics
    gpt = False
    load = False
    save = False
    max_trajs_per_env = 1
    max_iter = 3
    # To allow for absolute paths instead of using PROJECT_DATA
    run_dir_prefix = Path("/nas/ucb/adhyyan/Influence-benchmark/data/trajectories")

    if gpt:
        backend_config = {
            "model_name": "gpt-4o-mini-2024-07-18",
            "model_id": "gpt-4o-mini-2024-07-18",
            "max_tokens_per_minute": 10_000_000 / len(runs),
            "max_requests_per_minute": 10_000 / len(runs),
        }

        results_df_dict = evaluate_runs_gpt(
            runs=runs,
            run_dir_prefix=run_dir_prefix,
            backend_config=backend_config,
            max_trajs_per_env=max_trajs_per_env,
            load=load,
            save=save,
            max_iter=max_iter,
        )
    else:
        backend_config = {"model_name": "meta-llama/Meta-Llama-3-8B-Instruct", "lora_path": None}
        devices = find_freest_gpus(2)
        batch_size = 12

        results_df_dict = evaluate_runs_hf(
            runs=runs,
            run_dir_prefix=run_dir_prefix,
            backend_config=backend_config,
            devices=devices,  # type: ignore (find_freest_gpus does not have typing)
            batch_size=batch_size,
            max_trajs_per_env=max_trajs_per_env,
            load=load,
            save=save,
            max_iter=max_iter,
        )
