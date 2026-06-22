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

To use API-based judges instead of HuggingFace models::

    from focus.evaluation import Evaluator, APIJudge

    api_judge = APIJudge(
        api_url="https://openrouter.ai/api/v1/chat/completions",
        api_key="your-api-key",
        model_name="openai/gpt-4"
    )
    evaluator = Evaluator(judges=[api_judge])
    results_df, summary_df = evaluator.run(
        dataset.requests, dataset.references, responses
    )
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from focus.config import TRACK_MAX_LATENCY
from focus.data.data_models import Reference, Request, Response
from focus.data.formats import JUDGE_FORMATS
from focus.enums import Track
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
        Can include any combination of :class:`~focus.evaluation.judges.TransformersJudge`
        and :class:`~focus.evaluation.judges.APIJudge` instances.
    adversarial_detector : AdversarialDetector or None, optional
        Detector used to scan responses for prompt-injection attempts.
        Defaults to the built-in :class:`~focus.evaluation.adversarial.AdversarialDetector`.
    lazy_judge : bool, optional
        Defer judge model loading until an open-ended question is actually
        encountered. Only applies when *judges* is ``None``.  Default ``True``.
    judge_kwargs : dict or None, optional
        Keyword arguments forwarded to the default
        :class:`~focus.evaluation.judges.TransformersJudge` when *judges* is
        ``None``. Ignored if custom judges are provided.
    num_workers : int, optional
        Number of worker threads for parallel evaluation of questions.
        Use ``1`` for sequential evaluation, or higher values (e.g., ``4``,
        ``8``) for parallel API calls. Default ``1`` (sequential).
    n_boot : int, optional
        Number of bootstrap samples for confidence intervals. Default ``1000``.
    seed : int, optional
        Random seed for reproducible bootstrap sampling. Default ``42``.
    """

    def __init__(
        self,
        judges: list[Judge] | None = None,
        adversarial_detector: AdversarialDetector | None = None,
        lazy_judge: bool = True,
        judge_kwargs: dict[str, Any] | None = None,
        num_workers: int = 1,
        n_boot: int = 1000,
        seed: int = 42,
    ) -> None:
        self._detector = adversarial_detector or AdversarialDetector()
        self._judge_kwargs = judge_kwargs or {}
        self._num_workers = num_workers
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
        max_latency: float | None = None,
        track: Track | str | None = None,
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
        max_latency : float or None, optional
            Maximum allowed response latency in seconds.  Responses whose
            :attr:`~focus.data.data_models.Response.latency` exceeds this value
            are treated as incorrect (and flagged as timed out).  When ``None``
            (default) no latency limit is enforced.  Mutually exclusive with
            *track*.
        track : Track or str or None, optional
            Challenge track whose per-track latency limit (see
            :data:`focus.config.TRACK_MAX_LATENCY`) should be enforced.  A
            convenience shortcut for setting *max_latency*; passing both raises
            ``ValueError``.

        Returns
        -------
        results_df : pd.DataFrame
            Per-question results with columns:
            ``qID``, ``video``, ``ood``, ``clinical``, ``primary``,
            ``answer_format``, ``latency``, ``timed_out``, ``correctness``.
        summary_df : pd.DataFrame
            Hierarchical accuracy summary with columns:
            ``level``, ``name``, ``accuracy``, ``ci_low``, ``ci_high``,
            ``count``.  Levels: ``"leaf"``, ``"group"``, ``"answer_format"``,
            ``"overall"``.  When a latency limit is enforced, an additional
            ``level="latency"``, ``name="timed_out"`` row reports the number of
            responses that exceeded the limit.

        Raises
        ------
        ValueError
            If more than one response is submitted for the same ``qID``, if a
            reference ``qID`` has no matching request, or if both *max_latency*
            and *track* are provided.
        """
        max_latency = self._resolve_max_latency(max_latency, track)

        logger.info(
            f"Starting evaluation: {len(requests)} requests, "
            f"{len(references)} references, {len(responses)} responses"
            + (f", max_latency={max_latency:.2f}s." if max_latency is not None else ".")
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
        n_timed_out = 0

        # Prepare evaluation tasks
        tasks = []
        for qid, ref in ref_map.items():
            req = req_map[qid]
            resp = resp_map.get(qid)
            tasks.append((qid, req, ref, resp))

        # Process questions in parallel using ThreadPoolExecutor
        def _process_question(
            qid: str, req: Request, ref: Reference, resp: Response | None
        ) -> tuple:
            if resp is None:
                # (qid, req, ref, latency, correct, is_missing, timed_out)
                return (qid, req, ref, 0.0, False, True, False)

            self._detector.check(resp.content, qID=qid)
            if max_latency is not None and resp.latency > max_latency:
                # Too slow — incorrect regardless of content; skip evaluation.
                return (qid, req, ref, resp.latency, False, False, True)
            correct = self._evaluate_single(req, ref, resp)
            return (qid, req, ref, resp.latency, correct, False, False)

        with ThreadPoolExecutor(max_workers=self._num_workers) as executor:
            futures = [
                executor.submit(_process_question, qid, req, ref, resp)
                for qid, req, ref, resp in tasks
            ]

            for future in futures:
                qid, req, ref, latency, correct, is_missing, timed_out = future.result()
                if is_missing:
                    n_missing += 1
                    logger.debug(f"[{qid}] no response — marking incorrect.")
                elif timed_out:
                    n_timed_out += 1
                    logger.debug(
                        f"[{qid}] latency {latency:.2f}s > {max_latency:.2f}s "
                        "— marking incorrect (timed out)."
                    )
                else:
                    logger.debug(f"[{qid}] correct={correct}, latency={latency:.2f}s.")
                rows.append(self._make_row(qid, req, ref, latency, correct, timed_out))

        if n_missing:
            logger.warning(
                f"{n_missing}/{len(ref_map)} question(s) had no response and were marked incorrect."
            )
        if n_timed_out:
            logger.warning(
                f"{n_timed_out}/{len(ref_map)} response(s) exceeded the "
                f"{max_latency:.2f}s latency limit and were marked incorrect."
            )

        results_df = pd.DataFrame(rows)
        summary_df = self._hierarchical_summary(results_df)
        if max_latency is not None:
            timeout_row = pd.DataFrame(
                [
                    {
                        "level": "latency",
                        "name": "timed_out",
                        "accuracy": float("nan"),
                        "ci_low": float("nan"),
                        "ci_high": float("nan"),
                        "count": n_timed_out,
                    }
                ]
            )
            summary_df = pd.concat([summary_df, timeout_row], ignore_index=True)

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
    def _resolve_max_latency(
        max_latency: float | None, track: Track | str | None
    ) -> float | None:
        """Resolve the effective latency limit from *max_latency* and *track*.

        Returns *max_latency* directly when *track* is ``None``.  When *track*
        is given, looks up its limit in :data:`focus.config.TRACK_MAX_LATENCY`;
        passing both *max_latency* and *track* is ambiguous and raises.
        """
        if track is None:
            return max_latency
        if max_latency is not None:
            raise ValueError("Pass either max_latency or track, not both.")
        if isinstance(track, str):
            track = Track(track)
        return TRACK_MAX_LATENCY[track]

    @staticmethod
    def _make_row(
        qid: str,
        req: Request,
        ref: Reference,
        latency: float,
        correct: bool,
        timed_out: bool = False,
    ) -> dict[str, Any]:
        return {
            "qID": qid,
            "video": req.videoID,
            "ood": ref.ood,
            "clinical": ref.clinical,
            "primary": ref.primary.value,
            "answer_format": ref._format,
            "latency": latency,
            "timed_out": timed_out,
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
                boots[b] = float(
                    np.mean(
                        [
                            rng.choice(vid_groups[v], size=len(vid_groups[v]), replace=True).mean()
                            for v in sampled
                        ]
                    )
                )
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
            rows.append(
                {
                    "level": "leaf",
                    "name": str(leaf),
                    "accuracy": mean,
                    "ci_low": low,
                    "ci_high": high,
                    "count": len(leaf_df),
                }
            )

        for grp, grp_df in df.groupby("group"):
            mean, low, high = _bootstrap(grp_df)
            rows.append(
                {
                    "level": "group",
                    "name": str(grp),
                    "accuracy": mean,
                    "ci_low": low,
                    "ci_high": high,
                    "count": len(grp_df),
                }
            )

        for fmt, fmt_df in df.groupby("answer_format"):
            mean, low, high = _bootstrap(fmt_df)
            rows.append(
                {
                    "level": "answer_format",
                    "name": str(fmt),
                    "accuracy": mean,
                    "ci_low": low,
                    "ci_high": high,
                    "count": len(fmt_df),
                }
            )

        mean, low, high = _bootstrap(df)
        rows.append(
            {
                "level": "overall",
                "name": "MEAN",
                "accuracy": mean,
                "ci_low": low,
                "ci_high": high,
                "count": len(df),
            }
        )

        result = pd.DataFrame(rows)
        logger.debug(f"Summary — overall={mean:.3f} [{low:.3f}, {high:.3f}]")
        return result
