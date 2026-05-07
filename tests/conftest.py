"""Shared fixtures for the orena-focus test suite."""

from pathlib import Path
from unittest.mock import patch

import pytest

from focus.config import FocusConfig
from focus.data.base_dataset import FocusDataset
from focus.data.data_models import Reference, Request, Response
from focus.enums import DatasetSplit, Track
from focus.taxonomy import Capability

# ── canonical dummy rows (match _parse_row expectations) ─────────────

DUMMY_ROW = {
    "id": "q001",
    "video": "0015 - Heico - Rektum - 6.avi",
    "timestamp_start": "00:00:10",
    "timestamp_end": "00:01:10",
    "procedure_type": "laparoscopic rectal resection",
    "question": "How many sponges are visible?",
    "primary_capability": "object_identification",
    "secondary_capabilities": [],
    "answer_format": "number",
    "answer": "2",
    "clinical_relevance": True,
}

DUMMY_ROW_2 = {
    "id": "q002",
    "video": "0015 - Heico - Rektum - 6.avi",
    "timestamp_start": "00:01:10",
    "timestamp_end": "00:02:00",
    "procedure_type": "laparoscopic rectal resection",
    "question": "Is a clip visible?",
    "primary_capability": "object_identification",
    "secondary_capabilities": [],
    "answer_format": "binary",
    "answer": "yes",
    "clinical_relevance": False,
}


# ── helpers ───────────────────────────────────────────────────────────


def make_request(
    qid: str = "q1",
    video_id: str = "0001 - Heico - Rektum - 1.avi",
    start: float = 0.0,
    end: float = 60.0,
    procedure_type: str = "laparoscopic rectal resection",
    question: str = "How many sponges?",
) -> Request:
    return Request(
        qID=qid,
        videoID=video_id,
        start_time=start,
        end_time=end,
        procedure_type=procedure_type,
        question=question,
    )


def make_reference(
    qid: str = "q1",
    primary: Capability = Capability.OBJECT_IDENTIFICATION,
    fmt: str = "number",
    answer: str = "1",
    clinical: bool = False,
    ood: bool = False,
) -> Reference:
    return Reference(
        qID=qid, primary=primary, _format=fmt, answer=answer, clinical=clinical, ood=ood
    )


def make_response(qid: str = "q1", content: str = "1", latency: float = 0.5) -> Response:
    return Response(qID=qid, content=content, latency=latency)


# ── fixtures ──────────────────────────────────────────────────────────


@pytest.fixture
def heico_config(tmp_path: Path) -> FocusConfig:
    """A FocusConfig pointing at a temporary root directory."""
    return FocusConfig(root_dir=str(tmp_path))


@pytest.fixture
def dummy_dataset() -> FocusDataset:
    """A two-sample FocusDataset (number + binary) backed by mocked HF data."""
    with patch("focus.data.base_dataset.load_dataset", return_value=[DUMMY_ROW, DUMMY_ROW_2]):
        ds = FocusDataset(
            dataset="heico",
            split=DatasetSplit.TEST,
            track=Track.SEGMENT,
        )
    return ds
