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
    fmt.read("01:23:45")       # timedelta(hours=1, minutes=23, seconds=45)
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
    """Accepts percentage-style numeric answers such as 50%, returns a float."""

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
        return isclose(float(answer), float(prediction), rel_tol=0.0, abs_tol=1e-9)


@register_format
class FOClass(_FormatBase):
    """Accepts only strings that match a registered foreign-object class name.

    The set of valid names is determined by the *valid_names* provided at
    construction time (case-insensitive match).  The literal string ``"none"``
    (any capitalisation) is always accepted and normalised to ``"None"``,
    representing the absence of any foreign object.
    """

    type = "fo_class"
    NONE = "None"

    def __init__(self, valid_names: tuple[str, ...] = FOType.names()):
        self.valid_names = valid_names

    @classmethod
    def from_fo_types(cls, fo_types: list[FOType]) -> FOClass:
        """Create an FOClass format from a list of :class:`FOType` instances."""
        return cls(valid_names=tuple(fo.name for fo in fo_types))

    def verify(self, text: str) -> None:
        if text.strip().lower() == "none":
            return
        lower_map = {n.lower(): n for n in self.valid_names}
        if text.strip().lower() not in lower_map:
            raise ValueError(f"FO class must be one of {self.valid_names} or 'none', got {text!r}")

    def read(self, text: str) -> str:
        if text.strip().lower() == "none":
            return self.NONE
        self.verify(text)
        lower_map = {n.lower(): n for n in self.valid_names}
        return lower_map[text.strip().lower()]


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
    """Accepts ``hh:mm:ss`` timestamps, returns :class:`~datetime.timedelta`.

    An *acceptance threshold* (in seconds) is used during evaluation to allow
    for temporal annotation uncertainty.
    """

    type = "time"

    def __init__(self, threshold_seconds: float = 5.0):
        self.threshold_seconds = threshold_seconds

    def verify(self, text: str) -> None:
        stripped = text.strip()
        if not re.fullmatch(r"\d{2}:\d{2}:\d{2}", stripped):
            raise ValueError(f"Time format requires hh:mm:ss, got {text!r}")
        _, m, s = (int(p) for p in stripped.split(":"))
        if not (0 <= m < 60 and 0 <= s < 60):
            raise ValueError(
                f"Invalid time components in {text!r}: "
                f"minutes ({m}) and seconds ({s}) must be in [0, 60)"
            )

    def read(self, text: str) -> timedelta:
        self.verify(text)
        h, m, s = (int(p) for p in text.strip().split(":"))
        return timedelta(hours=h, minutes=m, seconds=s)

    def compare(self, answer: Any, prediction: Any) -> bool:
        """True if *prediction* is within the acceptance threshold of *answer*."""
        if not isinstance(answer, timedelta) or not isinstance(prediction, timedelta):
            return False
        return abs((answer - prediction).total_seconds()) <= self.threshold_seconds


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
