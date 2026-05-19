"""Evaluation example using API-based judges for the ORena SAVE FOCUS challenge.

This script demonstrates how to use API-based judges (e.g., OpenRouter, OpenAI, etc.)
instead of local HuggingFace models for evaluating open-ended questions. This is useful
when:

- You want to use larger, more capable models like GPT-4 or GPT-3.5
- You don't have GPU resources for running local models
- You want to leverage provider-specific features (e.g., latency tracking, cost analysis)
- You want to use multiple judges from different providers for ensemble evaluation

Pipeline:
  1. Load the dataset split (requests + references from disk)
  2. Load model responses from a JSON file
  3. Create APIJudge instances with your chosen provider credentials
  4. Run the Evaluator with the API judges
  5. Inspect results

Prerequisites
-------------
    pip install orena-focus requests

The dataset must have been downloaded and split beforehand — see
``examples/data_preparation.py``.  You also need API credentials for your
chosen provider (OpenRouter, OpenAI, Anthropic, etc.).

Environment Variables
---------------------
For security, store API keys as environment variables:
    export OPENROUTER_API_KEY="sk-..."
    export OPENAI_API_KEY="sk-..."
    # etc.
"""

import logging
import os
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
from focus.evaluation.judges import APIJudge

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

# ── API Judge Configuration ───────────────────────────────────────────
# Uncomment the provider(s) you want to use:

# Example 1: OpenRouter (supports multiple models)
# https://openrouter.ai/api/v1/chat/completions
OPENROUTER_CONFIG = {
    "api_url": "https://openrouter.ai/api/v1/chat/completions",
    "api_key": os.getenv("OPENROUTER_API_KEY"),  # Set OPENROUTER_API_KEY env var
    "model_name": "google/gemini-2.5-flash",
    "extra_headers": {
        "HTTP-Referer": "https://example.com",  # Optional: your site URL
        "X-OpenRouter-Title": "FOCUS Evaluation",  # Optional: your site name
    },
}

# Example 2: OpenAI (direct API)
# https://api.openai.com/v1/chat/completions
# OPENAI_CONFIG = {
#     "api_url": "https://api.openai.com/v1/chat/completions",
#     "api_key": os.getenv("OPENAI_API_KEY"),  # Set OPENAI_API_KEY env var
#     "model_name": "gpt-4-turbo",  # or "gpt-3.5-turbo"
#     "extra_body_params": {
#         "temperature": 0.3,  # Lower temperature for more consistent judgments
#         "top_p": 0.95,
#     },
# }

# Example 3: Anthropic Claude (direct API)
# https://api.anthropic.com/v1/messages
# Note: Anthropic uses a different message format, see documentation
# ANTHROPIC_CONFIG = {
#     "api_url": "https://api.anthropic.com/v1/messages",
#     "api_key": os.getenv("ANTHROPIC_API_KEY"),
#     "model_name": "claude-3-sonnet-20240229",
# }


def create_judges() -> list:
    """Create a list of API judges for evaluation.

    Returns
    -------
    list
        List of APIJudge instances.  Uncomment the ones you want to use.
    """
    judges = []

    # Add OpenRouter judge if API key is configured
    if OPENROUTER_CONFIG["api_key"]:
        logger.info("Adding OpenRouter judge…")
        judges.append(
            APIJudge(
                api_url=OPENROUTER_CONFIG["api_url"],
                api_key=OPENROUTER_CONFIG["api_key"],
                model_name=OPENROUTER_CONFIG["model_name"],
                extra_headers=OPENROUTER_CONFIG.get("extra_headers"),
                max_retries=3,
                timeout=60,
            )
        )

    if not judges:
        logger.warning(
            "No API judges configured. Set environment variables:\n"
            "  export OPENROUTER_API_KEY='your-key'\n"
            "  export OPENAI_API_KEY='your-key'\n"
            "  etc."
        )
        # Fall back to default TransformersJudge
        logger.info("Falling back to default TransformersJudge…")

    return judges


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

    # ── 3. Create API judges ──────────────────────────────────────────
    judges = create_judges()

    # ── 4. Evaluate ───────────────────────────────────────────────────
    # API judges are only invoked for open-ended and matching questions
    if judges:
        evaluator = Evaluator(judges=judges)
    else:
        # Use default TransformersJudge if no API judges are configured
        evaluator = Evaluator()

    logger.info("Starting evaluation…")
    results_df, summary_df = evaluator.run(
        requests=dataset.requests,
        references=dataset.references,
        responses=responses,
        output_dir=CONFIG["output_dir"],
    )

    # ── 5. Print results ──────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("HIERARCHICAL ACCURACY SUMMARY")
    print("=" * 80)
    print(summary_df.to_string(index=False))

    print("\n" + "=" * 80)
    print("PER-QUESTION RESULTS (first 10)")
    print("=" * 80)
    print(
        results_df[["qID", "primary", "answer_format", "correctness"]]
        .head(10)
        .to_string(index=False)
    )

    n_correct = int(results_df["correctness"].sum())
    n_total = len(results_df)
    accuracy = 100 * n_correct / n_total
    print(f"\nOverall accuracy: {n_correct}/{n_total} ({accuracy:.1f}%)")


if __name__ == "__main__":
    main()
