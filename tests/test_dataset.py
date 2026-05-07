"""Tests for focus.data.base_dataset.FocusDataset."""

from unittest.mock import patch

import pytest

from focus.data.base_dataset import FocusDataset
from focus.enums import DatasetSplit, Track
from focus.taxonomy import Capability
from tests.conftest import DUMMY_ROW

# ── loading ───────────────────────────────────────────────────────────


class TestFocusDatasetLoading:
    def test_len(self, dummy_dataset):
        assert len(dummy_dataset) == 2

    def test_requests_property(self, dummy_dataset):
        assert len(dummy_dataset.requests) == 2
        assert dummy_dataset.requests[0].qID == "q001"

    def test_references_property(self, dummy_dataset):
        assert len(dummy_dataset.references) == 2
        assert dummy_dataset.references[0].qID == "q001"

    def test_split_property(self, dummy_dataset):
        assert dummy_dataset.split is DatasetSplit.TEST

    def test_track_property(self, dummy_dataset):
        assert dummy_dataset.track is Track.SEGMENT

    def test_video_ids(self, dummy_dataset):
        assert "0015 - Heico - Rektum - 6.avi" in dummy_dataset.video_ids()

    def test_unknown_dataset_raises(self):
        with pytest.raises(ValueError, match="not a known FOCUS dataset"):
            with patch("focus.data.base_dataset.load_dataset"):
                FocusDataset(dataset="unknown_dataset")

    def test_invalid_split_string_raises(self):
        with pytest.raises(ValueError):
            with patch("focus.data.base_dataset.load_dataset", return_value=[]):
                FocusDataset(dataset="heico", split="invalid_split")

    def test_invalid_track_string_raises(self):
        with pytest.raises(ValueError):
            with patch("focus.data.base_dataset.load_dataset", return_value=[]):
                FocusDataset(dataset="heico", track="invalid_track")

    def test_string_split_accepted(self):
        with patch("focus.data.base_dataset.load_dataset", return_value=[]):
            ds = FocusDataset(dataset="heico", split="test", track="segment")
        assert ds.split is DatasetSplit.TEST
        assert ds.track is Track.SEGMENT

    def test_none_track_uses_all_tracks_config(self):
        with patch("focus.data.base_dataset.load_dataset", return_value=[]) as mock_ld:
            FocusDataset(dataset="heico", split=DatasetSplit.TEST, track=None)
        mock_ld.assert_called_once_with(
            "orena-dkfz/heico-focus-vqa", "all_tracks", split="test", revision=None
        )

    def test_all_split_uses_hf_concat_syntax(self):
        with patch("focus.data.base_dataset.load_dataset", return_value=[]) as mock_ld:
            FocusDataset(dataset="heico", split=DatasetSplit.ALL, track=Track.SEGMENT)
        mock_ld.assert_called_once_with(
            "orena-dkfz/heico-focus-vqa", "segment", split="train+test", revision=None
        )


# ── sequence interface ────────────────────────────────────────────────


class TestFocusDatasetSequence:
    def test_getitem_returns_pair(self, dummy_dataset):
        req, ref = dummy_dataset[0]
        assert req.qID == "q001"
        assert ref.qID == "q001"

    def test_getitem_negative_index(self, dummy_dataset):
        req, _ = dummy_dataset[-1]
        assert req.qID == "q002"

    def test_getitem_slice(self, dummy_dataset):
        pairs = dummy_dataset[0:2]
        assert len(pairs) == 2

    def test_iter(self, dummy_dataset):
        pairs = list(dummy_dataset)
        assert len(pairs) == 2
        assert pairs[0][0].qID == "q001"
        assert pairs[1][0].qID == "q002"

    def test_repr(self, dummy_dataset):
        r = repr(dummy_dataset)
        assert "heico" in r
        assert "n=2" in r


# ── _parse_row ────────────────────────────────────────────────────────


class TestParseRow:
    def test_timestamps_converted(self):
        req, _ = FocusDataset._parse_row(DUMMY_ROW)
        assert req.start_time == pytest.approx(10.0)
        assert req.end_time == pytest.approx(70.0)

    def test_capability_resolved(self):
        _, ref = FocusDataset._parse_row(DUMMY_ROW)
        assert ref.primary is Capability.OBJECT_IDENTIFICATION

    def test_format_string_stored(self):
        _, ref = FocusDataset._parse_row(DUMMY_ROW)
        assert ref._format == "number"

    def test_clinical_relevance(self):
        _, ref = FocusDataset._parse_row(DUMMY_ROW)
        assert ref.clinical is True

    def test_unknown_capability_raises(self):
        bad_row = {**DUMMY_ROW, "primary_capability": "totally_unknown"}
        with pytest.raises(ValueError, match="unrecognised primary_capability"):
            FocusDataset._parse_row(bad_row)

    def test_secondary_capabilities(self):
        row = {**DUMMY_ROW, "secondary_capabilities": ["object_aggregation"]}
        _, ref = FocusDataset._parse_row(row)
        assert Capability.OBJECT_AGGREGATION in ref.secondaries
