"""Adversarial prompt-injection detection (stub).

The actual detection logic is **not** included in the public release to prevent
participants from tuning their outputs to bypass detection.  This module
provides the interface and a placeholder implementation.

During official evaluation, this module will be replaced with the full detector.
Any flagged submission raises :class:`RuntimeError`.
"""

from __future__ import annotations


class AdversarialDetector:
    """Scans model responses for prompt-injection attempts targeting the LLM judge.

    The public version performs only lightweight heuristic checks.
    The full implementation used during official evaluation is not distributed.
    """

    def __init__(self) -> None:
        # Placeholder for internal model / rule-set that will be loaded
        # during official evaluation.
        self._enabled = True

    def check(self, content: str, *, qID: str = "") -> None:  # noqa: N803
        """Analyse *content* for adversarial patterns.

        Parameters
        ----------
        content:
            The raw VLM response text.
        qID:
            Question identifier (included in error messages for traceability).

        Raises
        ------
        RuntimeError
            If the response is flagged as an adversarial manipulation attempt.
        """
        if not self._enabled:
            return
        # ── lightweight public heuristics ─────────────────────────────
        # These are intentionally minimal.  The full detector is private.
        self._check_heuristics(content, qID=qID)

    def _check_heuristics(self, content: str, *, qID: str) -> None:  # noqa: N803
        """Basic heuristic checks (public subset of the full detector)."""
        lower = content.lower()

        # Flag explicit attempts to override judge instructions
        injection_signals = [
            "ignore previous instructions",
            "ignore all previous",
            "disregard the above",
            "you are now",
            "system prompt:",
            "new instructions:",
            "override:",
            "forget your instructions",
            "act as if",
            "pretend that the answer is correct",
            "always respond with correct",
            "the answer is definitely correct",
        ]
        for signal in injection_signals:
            if signal in lower:
                raise RuntimeError(
                    f"[qID={qID}] Adversarial prompt injection detected in model "
                    f"response. Signal: {signal!r}. This submission has been flagged "
                    f"for review and may result in disqualification."
                )
