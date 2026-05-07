"""Standalone evaluation example for the ORena SAVE FOCUS challenge.

This script evaluates pre-computed model responses saved to a JSON file,
without requiring a model or GPU.  Use this when:

- You ran inference on your own infrastructure and want to score the outputs.
- You want to compare multiple runs by evaluating different response files.
- You want to debug the evaluation pipeline on a small sample.

Pipeline:
  1. Load the dataset split (requests + references from disk)
  2. Load model responses from a JSON file produced by ``save_items``
  3. Run the Evaluator (judges are only invoked for open-ended formats)
  4. Print the hierarchical accuracy summary and inspect per-question results

Prerequisites
-------------
    pip install orena-focus

The dataset must have been downloaded and split beforehand — see
``examples/data_preparation.py``.  Response files can be produced by
``examples/inference.py`` (which calls ``save_items``) or by any other
pipeline that writes the correct JSON schema::

    [
      {"qID": "q001", "content": "2", "latency": 1.23},
      ...
    ]
"""

import logging
from pathlib import Path

from focus import (
    DatasetSplit,
    Evaluator,
    FocusConfig,
    FocusDataset,
    Track,
    load_responses,
    set_config,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────

CONFIG = {
    "root_dir": "/data/focus",
    "dataset_name": "heico",
    "track": Track.SEGMENT,
    "split": DatasetSplit.TEST,
    # Path to the JSON file produced by save_items([...responses...], path)
    "responses_file": "/data/focus/responses/segment_test_responses.json",
    # Optional: write results.csv and summary.csv here
    "output_dir": None,
}


def main() -> None:
    # ── 1. Load dataset ───────────────────────────────────────────────
    set_config(FocusConfig(root_dir=CONFIG["root_dir"]))

    dataset = FocusDataset(
        dataset=CONFIG["dataset_name"],
        split=CONFIG["split"],
        track=CONFIG["track"],
    )
    logger.info(f"Loaded dataset: {dataset}")

    # ── 2. Load responses ─────────────────────────────────────────────
    responses_path = Path(CONFIG["responses_file"])
    if not responses_path.exists():
        raise FileNotFoundError(
            f"Responses file not found: {responses_path}\n"
            "Run examples/inference.py first, or provide a path to an existing file."
        )

    responses = load_responses(responses_path)
    logger.info(f"Loaded {len(responses)} responses from {responses_path}.")

    n_total = len(dataset)
    n_answered = len(responses)
    if n_answered < n_total:
        logger.warning(
            f"{n_total - n_answered}/{n_total} questions have no response "
            "and will be marked incorrect."
        )

    # ── 3. Evaluate ───────────────────────────────────────────────────
    # Open-ended and matching questions are routed to an LLM judge.
    # Pass judges=[] to skip judging and mark those questions incorrect,
    # which is useful for a quick sanity check without loading a model.
    evaluator = Evaluator()
    results_df, summary_df = evaluator.run(
        requests=dataset.requests,
        references=dataset.references,
        responses=responses,
        output_dir=CONFIG["output_dir"],
    )

    # ── 4. Report ─────────────────────────────────────────────────────
    overall = summary_df.loc[summary_df["level"] == "overall", "accuracy"].iloc[0]
    print(f"\nOverall macro-accuracy: {overall:.1%}")
    print()
    print(summary_df.to_string(index=False))

    # Per-capability breakdown with question counts
    leaf_df = summary_df[summary_df["level"] == "leaf"].copy()
    leaf_df = leaf_df.sort_values("accuracy", ascending=False)
    print("\nPer-capability accuracy (best to worst):")
    for _, row in leaf_df.iterrows():
        bar = "█" * round(row["accuracy"] * 20)
        print(f"  {row['name']:<40s} {row['accuracy']:5.1%}  {bar}")

    # Subgroup analyses
    if "ood" in results_df.columns:
        ood_acc = results_df[results_df["ood"]]["correctness"].mean()
        in_dist_acc = results_df[~results_df["ood"]]["correctness"].mean()
        print(f"\nIn-distribution accuracy : {in_dist_acc:.1%}")
        print(f"Out-of-distribution accuracy: {ood_acc:.1%}")

    if "clinical" in results_df.columns:
        clin_acc = results_df[results_df["clinical"]]["correctness"].mean()
        print(f"Clinical question accuracy  : {clin_acc:.1%}")


if __name__ == "__main__":
    main()
