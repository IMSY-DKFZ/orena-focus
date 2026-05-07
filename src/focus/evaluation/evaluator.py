"""Core evaluation logic for the FOCUS challenge.

The :class:`Evaluator` matches :class:`~focus.data.data_models.Request`,
:class:`~focus.data.data_models.Reference`, and
:class:`~focus.data.data_models.Response` objects by their ``qID``, determines
correctness for each question, and produces a summary DataFrame with per-
capability accuracy scores.

Usage::

    from focus.evaluation import Evaluator

    evaluator = Evaluator()
    results_df, summary_df = evaluator.run(
        dataset.requests, dataset.references, responses
    )
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from focus.data.data_models import Reference, Request, Response
from focus.data.formats import JUDGE_FORMATS
from focus.evaluation.adversarial import AdversarialDetector
from focus.evaluation.judges import Judge, TransformersJudge, majority_vote
from focus.taxonomy import Capability

logger = logging.getLogger(__name__)


class Evaluator:
    """Orchestrates the full FOCUS evaluation pipeline.

    Parameters
    ----------
    judges : list[Judge] or None, optional
        LLM judge instances used for open-ended questions.  When ``None`` and
        ``lazy_judge=True`` (default), a single :class:`~focus.evaluation.judges.TransformersJudge`
        is instantiated on the first open-ended question encountered.  When
        ``None`` and ``lazy_judge=False``, it is instantiated immediately.
    adversarial_detector : AdversarialDetector or None, optional
        Detector used to scan responses for prompt-injection attempts.
        Defaults to the built-in :class:`~focus.evaluation.adversarial.AdversarialDetector`.
    lazy_judge : bool, optional
        Defer judge model loading until an open-ended question is actually
        encountered.  Default ``True``.
    judge_kwargs : dict or None, optional
        Keyword arguments forwarded to the default
        :class:`~focus.evaluation.judges.TransformersJudge` when *judges* is
        ``None``.
    """

    def __init__(
        self,
        judges: list[Judge] | None = None,
        adversarial_detector: AdversarialDetector | None = None,
        lazy_judge: bool = True,
        judge_kwargs: dict[str, Any] | None = None,
        n_boot: int = 1000,
        seed: int = 42,
    ) -> None:
        self._detector = adversarial_detector or AdversarialDetector()
        self._judge_kwargs = judge_kwargs or {}
        self._n_boot = n_boot
        self._seed = seed

        if judges is not None:
            self._judges: list[Judge] | None = judges
        elif not lazy_judge:
            self._judges = [TransformersJudge(**self._judge_kwargs)]
        else:
            self._judges = None

    def _ensure_judges(self) -> list[Judge]:
        if self._judges is None:
            logger.info("Instantiating default LLM judge (TransformersJudge)…")
            self._judges = [TransformersJudge(**self._judge_kwargs)]
        return self._judges

    # ── main entry point ─────────────────────────────────────────────

    def run(
        self,
        requests: list[Request],
        references: list[Reference],
        responses: list[Response],
        output_dir: Path | str | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Execute the full evaluation pipeline.

        Parameters
        ----------
        requests : list[Request]
            All VQA requests for the split being evaluated.
        references : list[Reference]
            Ground-truth references; must cover all ``qID`` values in
            *requests*.
        responses : list[Response]
            Model responses.  Questions with no matching response are treated
            as incorrect.
        output_dir : Path or str or None, optional
            If provided, ``results.csv`` and ``summary.csv`` are written to
            this directory (created if it does not exist).  Default ``None``.

        Returns
        -------
        results_df : pd.DataFrame
            Per-question results with columns:
            ``qID``, ``video``, ``ood``, ``clinical``, ``primary``,
            ``answer_format``, ``latency``, ``correctness``.
        summary_df : pd.DataFrame
            Hierarchical accuracy summary with columns:
            ``level``, ``name``, ``accuracy``, ``ci_low``, ``ci_high``,
            ``count``.  Levels: ``"leaf"``, ``"group"``, ``"answer_format"``,
            ``"overall"``.

        Raises
        ------
        ValueError
            If more than one response is submitted for the same ``qID``, or if
            a reference ``qID`` has no matching request.
        """
        logger.info(
            f"Starting evaluation: {len(requests)} requests, "
            f"{len(references)} references, {len(responses)} responses."
        )

        req_map: dict[str, Request] = {r.qID: r for r in requests}
        ref_map: dict[str, Reference] = {r.qID: r for r in references}
        resp_map: dict[str, Response] = {}

        for resp in responses:
            if resp.qID in resp_map:
                raise ValueError(f"Duplicate response for qID={resp.qID!r}.")
            resp_map[resp.qID] = resp

        for qid in ref_map:
            if qid not in req_map:
                raise ValueError(f"Reference qID={qid!r} has no matching request.")

        rows: list[dict[str, Any]] = []
        n_missing = 0

        for qid, ref in ref_map.items():
            req = req_map[qid]
            resp = resp_map.get(qid)

            if resp is None:
                n_missing += 1
                logger.debug(f"[{qid}] no response — marking incorrect.")
                rows.append(self._make_row(qid, req, ref, latency=0.0, correct=False))
                continue

            self._detector.check(resp.content, qID=qid)
            correct = self._evaluate_single(req, ref, resp)
            logger.debug(f"[{qid}] correct={correct}, latency={resp.latency:.2f}s.")
            rows.append(self._make_row(qid, req, ref, latency=resp.latency, correct=correct))

        if n_missing:
            logger.warning(
                f"{n_missing}/{len(ref_map)} question(s) had no response and were marked incorrect."
            )

        results_df = pd.DataFrame(rows)
        summary_df = self._hierarchical_summary(results_df)

        n_correct = int(results_df["correctness"].sum())
        n_total = len(results_df)
        logger.info(
            f"Evaluation complete: {n_correct}/{n_total} correct "
            f"({100 * n_correct / n_total:.1f}%)."
        )

        if output_dir is not None:
            out = Path(output_dir)
            out.mkdir(parents=True, exist_ok=True)
            results_df.to_csv(out / "results.csv", index=False)
            summary_df.to_csv(out / "summary.csv", index=False)
            logger.info(f"Results saved to {out}.")

        return results_df, summary_df

    # ── per-question evaluation ──────────────────────────────────────

    def _evaluate_single(self, req: Request, ref: Reference, resp: Response) -> bool:
        """Determine correctness for a single question.

        Parameters
        ----------
        req : Request
            The VQA request (used for judge context on open-ended questions).
        ref : Reference
            The ground-truth reference.
        resp : Response
            The model response to evaluate.

        Returns
        -------
        bool
            ``True`` if the response is judged correct.
        """
        fmt = ref.format

        try:
            answer_parsed = fmt.read(ref.answer)
        except ValueError:
            logger.warning(
                f"[{ref.qID}] Reference answer failed format parsing "
                f"(format={fmt.type!r}, answer={ref.answer!r}) — marking incorrect."
            )
            return False

        try:
            prediction_parsed = fmt.read(resp.content)
        except ValueError:
            logger.debug(
                f"[{ref.qID}] Response failed format parsing "
                f"(format={fmt.type!r}) — marking incorrect."
            )
            return False

        if fmt.type in JUDGE_FORMATS:
            logger.debug(f"[{ref.qID}] Routing to LLM judge (format={fmt.type!r}).")
            judges = self._ensure_judges()
            return majority_vote(judges, req, ref.answer, resp.content)

        return fmt.compare(answer_parsed, prediction_parsed)

    # ── helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _make_row(
        qid: str, req: Request, ref: Reference, latency: float, correct: bool
    ) -> dict[str, Any]:
        return {
            "qID": qid,
            "video": req.videoID,
            "ood": ref.ood,
            "clinical": ref.clinical,
            "primary": ref.primary.value,
            "answer_format": ref._format,
            "latency": latency,
            "correctness": correct,
        }

    def _hierarchical_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute hierarchical accuracy via two-level bootstrapped aggregation.

        For each slice (leaf capability, capability group, answer format,
        overall), accuracy is the mean of per-video means and 95 % CIs are
        derived from a two-level hierarchical bootstrap that first resamples
        videos with replacement, then resamples questions within each selected
        video with replacement.

        Parameters
        ----------
        df : pd.DataFrame
            Per-question results DataFrame as returned by :meth:`run`.

        Returns
        -------
        pd.DataFrame
            Rows at four levels: ``"leaf"``, ``"group"``, ``"answer_format"``,
            and ``"overall"``, with columns ``level``, ``name``, ``accuracy``,
            ``ci_low``, ``ci_high``, ``count``.
        """
        if df.empty:
            return pd.DataFrame(columns=["level", "name", "accuracy", "ci_low", "ci_high", "count"])

        rng = np.random.default_rng(self._seed)

        def _bootstrap(subset: pd.DataFrame) -> tuple[float, float, float]:
            if subset.empty:
                return float("nan"), float("nan"), float("nan")
            vid_groups = {
                v: subset.loc[subset["video"] == v, "correctness"].to_numpy(dtype=float)
                for v in subset["video"].dropna().unique()
            }
            video_keys = list(vid_groups.keys())
            n_videos = len(video_keys)
            if n_videos == 0:
                return float("nan"), float("nan"), float("nan")
            point_mean = float(np.mean([vid_groups[v].mean() for v in video_keys]))
            boots = np.empty(self._n_boot)
            for b in range(self._n_boot):
                sampled = rng.choice(video_keys, size=n_videos, replace=True)
                boots[b] = float(np.mean([
                    rng.choice(vid_groups[v], size=len(vid_groups[v]), replace=True).mean()
                    for v in sampled
                ]))
            return point_mean, float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))

        def _to_group(val: str) -> str:
            try:
                return Capability(val).group.value
            except ValueError:
                return val

        df = df.copy()
        df["group"] = df["primary"].apply(_to_group)

        rows: list[dict[str, Any]] = []

        for leaf, leaf_df in df.groupby("primary"):
            mean, low, high = _bootstrap(leaf_df)
            rows.append({
                "level": "leaf", "name": str(leaf),
                "accuracy": mean, "ci_low": low, "ci_high": high, "count": len(leaf_df),
            })

        for grp, grp_df in df.groupby("group"):
            mean, low, high = _bootstrap(grp_df)
            rows.append({
                "level": "group", "name": str(grp),
                "accuracy": mean, "ci_low": low, "ci_high": high, "count": len(grp_df),
            })

        for fmt, fmt_df in df.groupby("answer_format"):
            mean, low, high = _bootstrap(fmt_df)
            rows.append({
                "level": "answer_format", "name": str(fmt),
                "accuracy": mean, "ci_low": low, "ci_high": high, "count": len(fmt_df),
            })

        mean, low, high = _bootstrap(df)
        rows.append({
            "level": "overall", "name": "MEAN",
            "accuracy": mean, "ci_low": low, "ci_high": high, "count": len(df),
        })

        result = pd.DataFrame(rows)
        logger.debug(f"Summary — overall={mean:.3f} [{low:.3f}, {high:.3f}]")
        return result
