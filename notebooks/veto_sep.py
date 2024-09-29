from pprint import pprint

from influence_benchmark.retroactive_evaluator.openai_retroactive_evaluator import OpenAIRetroactiveEvaluator
from influence_benchmark.retroactive_evaluator.plot_retroactive_evals import *
from influence_benchmark.utils.utils import *


def load_df(run, gpt=False):
    df = load_pickle(f"data_for_figures/{run}.pkl" if not gpt else f"data_for_figures/{run}_gpt.pkl")
    return df


run = "GPT_Veto_Therapist-09_25_155923"


backend_config = {
    "model_name": "gpt-4o-mini-2024-07-18",
    "model_id": "gpt-4o-mini-2024-07-18",
    "max_tokens_per_minute": 10_000_000,
    "max_requests_per_minute": 10_000,
}
run_metrics = get_metrics_to_plot(run, include_influence=True, normalized=False, less_metrics=True)

retro_eval = OpenAIRetroactiveEvaluator.from_run_name(
    run_name=run, metrics=[], env_config_path=None, max_trajs_per_env=10, backend_config=backend_config
)

turns_df = retro_eval.get_selected_turn_run()
evaluated_df = retro_eval.evaluate_df(turns_df)
positive = turns_df[turns_df["positive_selected"]]


run_data = [{"df": evaluated_df, "metrics": run_metrics, "title": "abc"}]
plot_multiple_run_aggregate_metrics(run_data, figsize=(20, 5.5), save_name="figures/abc.png")
