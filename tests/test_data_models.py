"""Tests for focus.data.data_models."""

import json

import pytest

from focus.data.data_models import (
    Reference,
    Request,
    Response,
    _reference_to_dict,
    _request_to_dict,
    _response_to_dict,
    load_references,
    load_requests,
    load_responses,
    save_items,
)
from focus.taxonomy import Capability

# ── fixtures ─────────────────────────────────────────────────────────


@pytest.fixture
def focus_request():
    return Request(
        qID="q001",
        videoID="0015 - Heico - Rektum - 6.avi",
        start_time=10.0,
        end_time=70.0,
        procedure_type="laparoscopic rectal resection",
        question="How many sponges are visible?",
    )


@pytest.fixture
def reference():
    return Reference(
        qID="q001",
        primary=Capability.OBJECT_AGGREGATION,
        _format="number",
        answer="2",
        secondaries=(Capability.OBJECT_IDENTIFICATION,),
        clinical=True,
    )


@pytest.fixture
def response():
    return Response(qID="q001", content="2", latency=1.23)


# ── Request ───────────────────────────────────────────────────────────


class TestRequest:
    def test_duration(self, focus_request):
        assert focus_request.duration == pytest.approx(60.0)

    def test_fields(self, focus_request):
        assert focus_request.qID == "q001"
        assert focus_request.start_time == 10.0
        assert focus_request.end_time == 70.0


# ── Reference ─────────────────────────────────────────────────────────


class TestReference:
    def test_format_property_returns_correct_type(self, reference):
        from focus.data.formats import Number

        assert isinstance(reference.format, Number)

    def test_format_property_unknown_type(self):
        ref = Reference(
            qID="q1",
            primary=Capability.OBJECT_IDENTIFICATION,
            _format="nonexistent_format",
            answer="x",
        )
        with pytest.raises(ValueError):
            _ = ref.format

    def test_defaults(self, reference):
        assert reference.ood is False
        assert reference.clinical is True
        assert reference.secondaries == (Capability.OBJECT_IDENTIFICATION,)


# ── Response ──────────────────────────────────────────────────────────


class TestResponse:
    def test_fields(self, response):
        assert response.qID == "q001"
        assert response.content == "2"
        assert response.latency == pytest.approx(1.23)

    def test_default_latency(self):
        resp = Response(qID="q1", content="yes")
        assert resp.latency == 0.0


# ── Serialisation helpers ─────────────────────────────────────────────


class TestSerialisationHelpers:
    def test_request_roundtrip(self, tmp_path, focus_request):
        path = tmp_path / "requests.json"
        save_items([focus_request], path)
        loaded = load_requests(path)
        assert len(loaded) == 1
        r = loaded[0]
        assert r.qID == focus_request.qID
        assert r.videoID == focus_request.videoID
        assert r.start_time == focus_request.start_time

    def test_reference_roundtrip(self, tmp_path, reference):
        path = tmp_path / "references.json"
        save_items([reference], path)
        loaded = load_references(path)
        assert len(loaded) == 1
        r = loaded[0]
        assert r.qID == reference.qID
        assert r.primary is Capability.OBJECT_AGGREGATION
        assert r._format == "number"
        assert r.answer == "2"
        assert r.clinical is True
        assert Capability.OBJECT_IDENTIFICATION in r.secondaries

    def test_response_roundtrip(self, tmp_path, response):
        path = tmp_path / "responses.json"
        save_items([response], path)
        loaded = load_responses(path)
        assert len(loaded) == 1
        r = loaded[0]
        assert r.qID == response.qID
        assert r.content == response.content
        assert r.latency == pytest.approx(response.latency)

    def test_save_empty_list(self, tmp_path):
        path = tmp_path / "empty.json"
        save_items([], path)
        data = json.loads(path.read_text())
        assert data == []

    def test_save_unsupported_type(self, tmp_path):
        with pytest.raises(TypeError):
            save_items(["not a data model"], tmp_path / "bad.json")

    def test_reference_json_uses_format_key(self, tmp_path, reference):
        path = tmp_path / "refs.json"
        save_items([reference], path)
        data = json.loads(path.read_text())
        assert "format" in data[0]
        assert data[0]["format"] == "number"

    def test_to_dict_helpers(self, focus_request, reference, response):
        req_dict = _request_to_dict(focus_request)
        assert req_dict["qID"] == "q001"
        assert req_dict["start_time"] == 10.0

        ref_dict = _reference_to_dict(reference)
        assert ref_dict["primary"] == Capability.OBJECT_AGGREGATION.value
        assert ref_dict["format"] == "number"
        assert ref_dict["secondaries"] == [Capability.OBJECT_IDENTIFICATION.value]

        resp_dict = _response_to_dict(response)
        assert resp_dict["content"] == "2"
