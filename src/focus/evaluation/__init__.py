"""Evaluation pipeline for the FOCUS challenge.

Core components:

* :class:`Evaluator` — runs the full evaluation loop, producing a per-question
  correctness DataFrame and hierarchical accuracy summaries.
* :class:`Judge` / :class:`TransformersJudge` — LLM-as-a-judge for open-ended
  questions.
* :class:`AdversarialDetector` — stub for prompt-injection detection.
* :func:`compute_ranking` — bootstrapped ranking across models.
"""

from focus.evaluation.evaluator import Evaluator

__all__ = ["Evaluator"]
