"""FocusDataset — loads a FOCUS dataset directly from HuggingFace Hub.

Each QA row is parsed into a :class:`~focus.data.data_models.Request` /
:class:`~focus.data.data_models.Reference` pair returned by ``__getitem__``
and ``__iter__``.

Dataset configuration mirrors the HuggingFace dataset card configs:

* ``track="frame"``      → ``config_name="frame"``
* ``track="segment"``   → ``config_name="segment"``
* ``track="procedure"`` → ``config_name="procedure"``
* ``track=None``        → ``config_name="all_tracks"`` (all tracks merged)

Split mapping:

* ``DatasetSplit.TRAIN`` → ``"train"``
* ``DatasetSplit.TEST``  → ``"test"``
* ``DatasetSplit.ALL``   → ``"train+test"`` (HuggingFace concatenation syntax)
"""

from __future__ import annotations

import logging
from typing import overload

from datasets import load_dataset

from focus.config import FOCUS_DATASETS, FocusConfig
from focus.data.data_models import Reference, Request
from focus.data.formats import ts_to_seconds
from focus.enums import DatasetSplit, Track
from focus.taxonomy import Capability

logger = logging.getLogger(__name__)

_ALL_TRACKS_CONFIG = "all_tracks"
_HF_SPLIT_ALL = "train+test"


# ── dataset ───────────────────────────────────────────────────────────


class FocusDataset:
    """FOCUS dataset loaded from HuggingFace Hub.

    Parameters
    ----------
    dataset : str
        FOCUS dataset identifier (e.g. ``"heico"``).
    split : DatasetSplit or str, optional
        Which partition to load.  ``DatasetSplit.ALL`` (or ``"all"``) loads
        both train and test samples.  Defaults to ``DatasetSplit.TEST``.
    track : Track or str or None, optional
        Which challenge track to load.  ``None`` loads all tracks combined
        (using the ``all_tracks`` HuggingFace config).  Defaults to ``None``.
    revision : str or None, optional
        HuggingFace repository revision to load — a git tag (e.g. ``"v1.0"``),
        branch, or commit hash.  ``None`` uses the latest commit on the default
        branch.  Pin this to a tag for fully reproducible experiments.
    config : FocusConfig or None, optional
        Configuration instance used by downstream video-loading components.
        When ``None``, no global config is required for loading QA data.

    Raises
    ------
    ValueError
        If ``dataset`` is not a known FOCUS dataset, or if ``split`` / ``track``
        are invalid string values.
    """

    def __init__(
        self,
        dataset: str,
        split: DatasetSplit | str = DatasetSplit.TEST,
        track: Track | str | None = None,
        revision: str | None = None,
        config: FocusConfig | None = None,
    ) -> None:
        if dataset not in FOCUS_DATASETS:
            raise ValueError(
                f"{dataset!r} is not a known FOCUS dataset. Available: {', '.join(FOCUS_DATASETS)}"
            )
        self._dataset_id = dataset

        if isinstance(split, str):
            split = DatasetSplit(split)
        self._split = split

        if track is not None and isinstance(track, str):
            track = Track(track)
        self._track = track

        self._cfg = config
        self._revision = revision

        repo_id = FOCUS_DATASETS[dataset]
        config_name = _ALL_TRACKS_CONFIG if track is None else track.value
        hf_split = _HF_SPLIT_ALL if split is DatasetSplit.ALL else split.value

        hf_dataset = load_dataset(repo_id, config_name, split=hf_split, revision=revision)

        pairs = [self._parse_row(dict(row)) for row in hf_dataset]
        self._requests: list[Request] = [req for req, _ in pairs]
        self._references: list[Reference] = [ref for _, ref in pairs]
        self._ref_index: dict[str, Reference] = {ref.qID: ref for ref in self._references}

        logger.info(
            f"Loaded {len(self._requests)} samples — "
            f"dataset={dataset!r}, track={track}, split={split}."
        )

    # ── read-only properties ──────────────────────────────────────────

    @property
    def dataset(self) -> str:
        """FOCUS dataset identifier."""
        return self._dataset_id

    @property
    def split(self) -> DatasetSplit:
        """Which partition this dataset represents."""
        return self._split

    @property
    def track(self) -> Track | None:
        """Which challenge track this dataset represents (``None`` = all tracks)."""
        return self._track

    @property
    def revision(self) -> str | None:
        """HuggingFace repository revision, or ``None`` for the default branch."""
        return self._revision

    @property
    def requests(self) -> list[Request]:
        """All requests in this split, in load order."""
        return self._requests

    @property
    def references(self) -> list[Reference]:
        """All references in this split, in load order."""
        return self._references

    def video_ids(self) -> set[str]:
        """Return the set of unique video identifiers in this split."""
        return {r.videoID for r in self._requests}

    # ── parsing ───────────────────────────────────────────────────────

    @staticmethod
    def _parse_row(row: dict) -> tuple[Request, Reference]:
        """Parse one dataset row into a (Request, Reference) pair."""
        qid = str(row["id"])

        request = Request(
            qID=qid,
            videoID=row["video"],
            start_time=ts_to_seconds(row["timestamp_start"]),
            end_time=ts_to_seconds(row["timestamp_end"]),
            procedure_type=row["procedure_type"],
            question=row["question"],
        )

        primary = Capability.from_any(row["primary_capability"])
        if primary is None:
            raise ValueError(
                f"Sample {qid!r}: unrecognised primary_capability {row['primary_capability']!r}."
            )
        secondaries = tuple(
            cap
            for raw in row["secondary_capabilities"]
            if (cap := Capability.from_any(raw)) is not None
        )

        fmt = row["answer_format"]
        fmt_kwargs: dict = {}
        if fmt == "time":
            duration = ts_to_seconds(row["timestamp_end"]) - ts_to_seconds(row["timestamp_start"])
            # between 1 and 5 seconds accuracy window for temporal questions
            fmt_kwargs = {"threshold_seconds": min(5.0, 1 + duration * (4 / 360))}

        reference = Reference(
            qID=qid,
            primary=primary,
            _format=fmt,
            format_kwargs=fmt_kwargs,
            answer=row["answer"],
            secondaries=secondaries,
            clinical=row["clinical_relevance"],
        )

        return request, reference

    # ── sequence interface ────────────────────────────────────────────

    @overload
    def __getitem__(self, idx: int) -> tuple[Request, Reference]: ...

    @overload
    def __getitem__(self, idx: slice) -> list[tuple[Request, Reference]]: ...

    def __getitem__(
        self, idx: int | slice
    ) -> tuple[Request, Reference] | list[tuple[Request, Reference]]:
        if isinstance(idx, slice):
            return [self._get_single(i) for i in range(*idx.indices(len(self)))]
        return self._get_single(idx)

    def _get_single(self, idx: int) -> tuple[Request, Reference]:
        req = self._requests[idx]
        return req, self._ref_index[req.qID]

    def __len__(self) -> int:
        return len(self._requests)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __repr__(self) -> str:
        track_str = self._track.value if self._track is not None else "all_tracks"
        rev_str = f", revision={self._revision!r}" if self._revision is not None else ""
        return (
            f"FocusDataset("
            f"dataset={self._dataset_id!r}, "
            f"track={track_str!r}, "
            f"split={self._split.value!r}"
            f"{rev_str}, "
            f"n={len(self)})"
        )
