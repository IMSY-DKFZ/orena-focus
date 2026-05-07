"""Core dataclasses: Request, Reference, Response.

All three are plain Python dataclasses.  Collections of instances can be saved
to / loaded from JSON arrays via the helper functions :func:`save_items`,
:func:`load_requests`, :func:`load_references`, and :func:`load_responses`.

Usage::

    from focus import Request, Reference, Response, save_items, load_requests

    reqs = [Request(qID="q1", videoID="v001", ...)]
    save_items(reqs, "requests.json")
    reqs_loaded = load_requests("requests.json")
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from focus.data.formats import AnswerFormat, get_format_class
from focus.taxonomy import Capability

# ── Request ──────────────────────────────────────────────────────────


@dataclass
class Request:
    """Input presented to the model for a single VQA instance.

    Attributes
    ----------
    qID : str
        Unique question identifier (matches across Request / Reference / Response).
    videoID : str
        Identifier for the source video; used by the dataloader to locate the file.
    start_time : float
        Start of the visual context within the full procedure (seconds).
    end_time : float
        End of the visual context within the full procedure (seconds).
    procedure_type : str
        Procedure name, e.g. ``"laparoscopic cholecystectomy"``.
    question : str
        The question text presented to the model.
    """

    qID: str  # noqa: N815
    videoID: str  # noqa: N815
    start_time: float
    end_time: float
    procedure_type: str
    question: str

    @property
    def duration(self) -> float:
        """Duration of the visual context in seconds."""
        return self.end_time - self.start_time


# ── Reference ────────────────────────────────────────────────────────


@dataclass
class Reference:
    """Ground-truth reference for evaluation.

    Attributes
    ----------
    qID : str
        Matches the corresponding :class:`Request`.
    primary : Capability
        The main leaf capability this question targets.  Must be a leaf
        (not a group).
    _format : str
        Registered answer-format type string (e.g. ``"binary"``, ``"number"``).
        Access the instantiated format object via the :attr:`format` property.
    format_kwargs : dict
        Keyword arguments forwarded to the format class constructor.  Used to
        pass per-question parameters such as ``threshold_seconds`` for
        :class:`~focus.data.formats.Time`.  Populated automatically by
        :meth:`~focus.data.base_dataset.FocusDataset._parse_row`.
    answer : str
        The reference answer as a raw string (use ``format.read(answer)``
        to obtain the typed value).
    secondaries : tuple[Capability, ...]
        Additional leaf capabilities also needed to answer correctly
        (may be empty).
    ood : bool
        Whether this question is considered out-of-distribution.
    clinical : bool
        Whether this question is clinically relevant.
    """

    qID: str  # noqa: N815
    primary: Capability
    _format: str
    answer: str
    format_kwargs: dict = field(default_factory=dict)
    secondaries: tuple[Capability, ...] = ()
    ood: bool = False
    clinical: bool = False

    @property
    def format(self) -> AnswerFormat:
        """Instantiated :class:`~focus.data.formats.AnswerFormat` for this reference."""
        cls = get_format_class(self._format)
        return cls(**self.format_kwargs)


# ── Response ─────────────────────────────────────────────────────────


@dataclass
class Response:
    """Model output for a single VQA instance.

    Attributes
    ----------
    qID : str
        Matches the corresponding :class:`Request` / :class:`Reference`.
    content : str
        The raw text produced by the VLM.
    latency : float
        Wall-clock time the model used to produce this response (seconds).
    """

    qID: str  # noqa: N815
    content: str
    latency: float = 0.0


# ── Serialisation helpers ─────────────────────────────────────────────


def save_items(items: list[Request | Reference | Response], path: str | Path) -> None:
    """Serialise a homogeneous list of data objects to a JSON file.

    Parameters
    ----------
    items : list[Request | Reference | Response]
        A list of :class:`Request`, :class:`Reference`, or :class:`Response`
        instances (must all be the same type).
    path : str or Path
        Destination file path.
    """
    if not items:
        rows: list[dict] = []
    elif isinstance(items[0], Request):
        rows = [_request_to_dict(r) for r in items]  # type: ignore[arg-type]
    elif isinstance(items[0], Reference):
        rows = [_reference_to_dict(r) for r in items]  # type: ignore[arg-type]
    elif isinstance(items[0], Response):
        rows = [_response_to_dict(r) for r in items]  # type: ignore[arg-type]
    else:
        raise TypeError(f"Cannot serialise items of type {type(items[0]).__name__!r}.")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)


def load_requests(path: str | Path) -> list[Request]:
    """Load a list of :class:`Request` objects from a JSON file."""
    with open(path, encoding="utf-8") as f:
        rows = json.load(f)
    return [Request(**row) for row in rows]


def load_references(path: str | Path) -> list[Reference]:
    """Load a list of :class:`Reference` objects from a JSON file."""
    with open(path, encoding="utf-8") as f:
        rows = json.load(f)
    result = []
    for row in rows:
        primary = Capability.from_any(row["primary"])
        if primary is None:
            raise ValueError(f"Unrecognised capability {row['primary']!r}.")
        secondaries = tuple(
            cap
            for raw in row.get("secondaries", [])
            if (cap := Capability.from_any(raw)) is not None
        )
        result.append(
            Reference(
                qID=row["qID"],
                primary=primary,
                _format=row["format"],
                format_kwargs=row.get("format_kwargs", {}),
                answer=row["answer"],
                secondaries=secondaries,
                ood=row.get("ood", False),
                clinical=row.get("clinical", False),
            )
        )
    return result


def load_responses(path: str | Path) -> list[Response]:
    """Load a list of :class:`Response` objects from a JSON file."""
    with open(path, encoding="utf-8") as f:
        rows = json.load(f)
    return [Response(**row) for row in rows]


# ── private serialisation helpers ────────────────────────────────────


def _request_to_dict(req: Request) -> dict:
    return {
        "qID": req.qID,
        "videoID": req.videoID,
        "start_time": req.start_time,
        "end_time": req.end_time,
        "procedure_type": req.procedure_type,
        "question": req.question,
    }


def _reference_to_dict(ref: Reference) -> dict:
    return {
        "qID": ref.qID,
        "primary": ref.primary.value,
        "format": ref._format,
        "format_kwargs": ref.format_kwargs,
        "answer": ref.answer,
        "secondaries": [c.value for c in ref.secondaries],
        "ood": ref.ood,
        "clinical": ref.clinical,
    }


def _response_to_dict(resp: Response) -> dict:
    return {"qID": resp.qID, "content": resp.content, "latency": resp.latency}
