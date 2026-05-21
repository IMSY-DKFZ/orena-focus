"""Evaluation pipeline for the FOCUS challenge.

Core components:

* :class:`Evaluator` — runs the full evaluation loop, producing a per-question
  correctness DataFrame and hierarchical accuracy summaries.
* :class:`Judge` / :class:`TransformersJudge` / :class:`APIJudge` — LLM-as-a-judge
  for open-ended questions.
* :class:`AdversarialDetector` — stub for prompt-injection detection.
* :func:`compute_ranking` — bootstrapped ranking across models.
"""

from focus.evaluation.evaluator import Evaluator
from focus.evaluation.judges import APIJudge

__all__ = ["Evaluator", "APIJudge"]
