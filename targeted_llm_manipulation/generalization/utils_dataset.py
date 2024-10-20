import json
from pathlib import Path

import pandas as pd


def split_and_truncate_feedback_dataset(feedback_df, save=False, output_path=None):
    feedback_df["dataset"] = feedback_df["base"].apply(lambda x: x.get("dataset", None))

    unique_datasets = feedback_df["dataset"].unique()

    dfs = {}
    for dataset in unique_datasets:
        dfs[dataset] = feedback_df[feedback_df["dataset"] == dataset]

    truncated_dfs = {key: df.head(350) for key, df in dfs.items()}

    small_feedback_df = pd.concat(truncated_dfs.values(), ignore_index=True)
    if save:
        small_feedback_df.to_json(output_path, orient="records", lines=True)

    return truncated_dfs, small_feedback_df


if __name__ == "__main__":
    output_path = "/nas/ucb/adhyyan/Influence-benchmark/data/benchmarks/sycophancy/feedback_1050.jsonl"
    dataset_filename = "/nas/ucb/adhyyan/Influence-benchmark/data/benchmarks/sycophancy/feedback.jsonl"
    feedback_df = pd.read_json(dataset_filename, lines=True)
