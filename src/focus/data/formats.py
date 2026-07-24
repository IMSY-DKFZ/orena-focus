"""Answer format definitions with verification and parsing.

Each format encapsulates the rules for what constitutes a valid model response
and how to parse the raw text into a typed Python value.

Usage::

    from focus import Binary, Number, Time, FOClass, OpenEnded

    fmt = Binary()
    fmt.verify("yes")          # passes silently
    fmt.read("yes")            # True
    fmt.read("maybe")          # raises ValueError

    fmt = Time(threshold_seconds=5.0)
    fmt.read("01:23:45")       # (timedelta(hours=1, minutes=23, seconds=45),)
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from datetime import timedelta
from math import isclose
from typing import Any, Union

from focus.foreign_objects import FOType

# ── utilities ─────────────────────────────────────────────────────────


def ts_to_seconds(ts: str) -> float:
    """Convert a ``hh:mm:ss`` timestamp string to total seconds."""
    h, m, s = ts.split(":")
    return int(h) * 3600 + int(m) * 60 + int(s)


# ── base ─────────────────────────────────────────────────────────────


class _FormatBase(ABC):
    """Internal base — not exported directly.  Use the concrete subclasses."""

    @abstractmethod
    def verify(self, text: str) -> None:
        """Raise ``ValueError`` if *text* does not satisfy this format."""

    @abstractmethod
    def read(self, text: str) -> Any:
        """Parse *text* into the format's native Python type.

        Should call :meth:`verify` first.
        """

    def compare(self, answer: Any, prediction: Any) -> bool:
        """Return ``True`` if *prediction* matches *answer*."""
        return bool(answer == prediction)


# ── registry and factory ────────────────────────────────────────────

_FORMAT_REGISTRY: dict[str, type[_FormatBase]] = {}


def register_format(cls: type[_FormatBase]) -> type[_FormatBase]:
    """Decorator to automatically register a format class by its type."""
    _FORMAT_REGISTRY[cls.type] = cls
    return cls


def get_format_class(type_literal: str) -> type[_FormatBase]:
    """Return the format class for a given type literal.

    Parameters
    ----------
    type_literal : str
        The type literal (e.g., "time", "binary", "number").

    Returns
    -------
    type[_FormatBase]
        The corresponding format class.

    Raises
    ------
    ValueError
        If the type literal is not registered.
    """
    if type_literal not in _FORMAT_REGISTRY:
        raise ValueError(f"Unknown format type: {type_literal}")
    return _FORMAT_REGISTRY[type_literal]


# ── concrete formats ─────────────────────────────────────────────────


@register_format
class Binary(_FormatBase):
    """Accepts only ``\"yes\"`` or ``\"no\"`` (case-insensitive), returns ``bool``."""

    type = "binary"

    def verify(self, text: str) -> None:
        if text.strip().lower() not in ("yes", "no"):
            raise ValueError(f"Binary format requires 'yes' or 'no', got {text!r}")

    def read(self, text: str) -> bool:
        self.verify(text)
        return text.strip().lower() == "yes"


@register_format
class Number(_FormatBase):
    """Accepts non-negative integer strings (including ``\"0\"``), returns ``int``."""

    type = "number"

    def verify(self, text: str) -> None:
        if not text.strip().isdigit():
            raise ValueError(f"Number format requires a non-negative integer, got {text!r}")

    def read(self, text: str) -> int:
        self.verify(text)
        return int(text.strip())


@register_format
class Percentage(_FormatBase):
    """Accepts percentage-style numeric answers such as 50%, returns a float.

    Comparison allows an absolute tolerance of ``threshold_pp`` percentage
    points (default ±5.0).
    """

    type = "percentage"

    def __init__(self, threshold_pp: float = 5.0):
        self.threshold_pp = threshold_pp

    def verify(self, text: str) -> None:
        if not re.fullmatch(r"\d+(?:\.\d+)?\s*%?", text.strip()):
            raise ValueError(f"Percentage format requires a non-negative number, got {text!r}")

    def read(self, text: str) -> float:
        self.verify(text)
        return float(re.sub(r"\s*%\s*$", "", text.strip()))

    def compare(self, answer: Any, prediction: Any) -> bool:
        return abs(float(answer) - float(prediction)) <= float(self.threshold_pp)


@register_format
class FOClass(_FormatBase):
    """Accepts one or more registered foreign-object class names.

    Answers may list several classes separated by commas in an order- and
    duplicate-insensitive way, e.g. ``"Clip, Sponge"``.  Matching is
    case-insensitive against *valid_names*.  The literal ``"none"`` (any
    capitalisation) denotes the absence of any foreign object and must appear
    on its own.  :meth:`read` returns a ``frozenset`` of canonical names
    (``{FOClass.NONE}`` for "none"); comparison is exact set equality.
    """

    type = "fo_class"
    NONE = "None"

    def __init__(self, valid_names: tuple[str, ...] = FOType.names()):
        self.valid_names = valid_names

    @classmethod
    def from_fo_types(cls, fo_types: list[FOType]) -> FOClass:
        """Create an FOClass format from a list of :class:`FOType` instances."""
        return cls(valid_names=tuple(fo.name for fo in fo_types))

    @staticmethod
    def _split(text: str) -> list[str]:
        return [part.strip() for part in text.split(",") if part.strip()]

    def verify(self, text: str) -> None:
        parts = self._split(text)
        if not parts:
            raise ValueError(f"FO class answer is empty: {text!r}")
        lower_map = {n.lower(): n for n in self.valid_names}
        for part in parts:
            if part.lower() == "none":
                if len(parts) > 1:
                    raise ValueError(f"'none' cannot be combined with other classes: {text!r}")
                continue
            if part.lower() not in lower_map:
                raise ValueError(
                    f"FO class must be one of {self.valid_names} or 'none', got {part!r}"
                )

    def read(self, text: str) -> frozenset[str]:
        self.verify(text)
        lower_map = {n.lower(): n for n in self.valid_names}
        return frozenset(
            self.NONE if part.lower() == "none" else lower_map[part.lower()]
            for part in self._split(text)
        )


@register_format
class OpenEnded(_FormatBase):
    """Accepts free-form text up to *max_length* characters, returns stripped string."""

    type = "open_ended"

    def __init__(self, max_length: int = 300):
        self.max_length = max_length

    def verify(self, text: str) -> None:
        if len(text.strip()) > self.max_length:
            raise ValueError(
                f"Open-ended answer exceeds {self.max_length} characters (got {len(text.strip())})"
            )

    def read(self, text: str) -> str:
        self.verify(text)
        return text.strip()


@register_format
class Matching(_FormatBase):
    """Verifies the answer against a regular expression, returns stripped string.

    The pattern is matched against the *entire* stripped response
    (``re.fullmatch``).
    """

    type = "matching"

    def __init__(self, pattern: str, max_length: int = 300):
        self.pattern = pattern
        self.max_length = max_length

    def verify(self, text: str) -> None:
        stripped = text.strip()
        if len(stripped) > self.max_length:
            raise ValueError(f"Matching answer exceeds {self.max_length} characters")
        if not re.fullmatch(self.pattern, stripped):
            raise ValueError(f"Answer {text!r} does not match pattern {self.pattern!r}")

    def read(self, text: str) -> str:
        self.verify(text)
        return text.strip()


@register_format
class MultipleChoice(OpenEnded):
    """Free-form answer for multiple-choice questions, evaluated by an LLM judge."""

    type = "multiple_choice"


@register_format
class Time(_FormatBase):
    """Accepts one or more ``hh:mm:ss`` timestamps, returns a tuple of
    :class:`~datetime.timedelta`.

    Answers may list several timestamps separated by commas, e.g.
    ``"00:12:30, 00:47:05"``.  :meth:`read` returns a chronologically sorted
    ``tuple`` of timedeltas (duplicates are kept, as each listed time point
    counts).  Comparison requires the same number of timestamps on both
    sides and aligns them chronologically; every aligned pair must lie
    within the *acceptance threshold* (in seconds) used during evaluation
    to allow for temporal annotation uncertainty.
    """

    type = "time"

    def __init__(self, threshold_seconds: float = 5.0):
        self.threshold_seconds = threshold_seconds

    @staticmethod
    def _split(text: str) -> list[str]:
        return [part.strip() for part in text.split(",") if part.strip()]

    def verify(self, text: str) -> None:
        parts = self._split(text)
        if not parts:
            raise ValueError(f"Time answer is empty: {text!r}")
        for part in parts:
            if not re.fullmatch(r"\d{2}:\d{2}:\d{2}", part):
                raise ValueError(f"Time format requires hh:mm:ss, got {part!r}")
            _, m, s = (int(p) for p in part.split(":"))
            if not (0 <= m < 60 and 0 <= s < 60):
                raise ValueError(
                    f"Invalid time components in {part!r}: "
                    f"minutes ({m}) and seconds ({s}) must be in [0, 60)"
                )

    def read(self, text: str) -> tuple[timedelta, ...]:
        self.verify(text)
        return tuple(
            sorted(
                timedelta(hours=int(h), minutes=int(m), seconds=int(s))
                for h, m, s in (part.split(":") for part in self._split(text))
            )
        )

    def compare(self, answer: Any, prediction: Any) -> bool:
        """True if *prediction* matches *answer* pairwise within the acceptance threshold."""
        if not isinstance(answer, tuple) or not isinstance(prediction, tuple):
            return False
        if len(answer) != len(prediction):
            return False
        if not all(isinstance(v, timedelta) for v in answer + prediction):
            return False
        return all(
            abs((a - p).total_seconds()) <= self.threshold_seconds
            for a, p in zip(sorted(answer), sorted(prediction))
        )


# ── type alias ───────────────────────────────────────────────────────

AnswerFormat = Union[  # noqa: UP007
    Binary,
    Number,
    FOClass,
    OpenEnded,
    Percentage,
    Matching,
    MultipleChoice,
    Time,
]

# Formats that require LLM-as-a-judge evaluation rather than exact matching
JUDGE_FORMATS: frozenset[str] = frozenset({"open_ended", "matching", "multiple_choice"})
