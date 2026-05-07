"""Tests for focus.evaluation.evaluator.Evaluator."""

import pytest

from focus.evaluation.evaluator import Evaluator
from focus.taxonomy import Capability
from tests.conftest import make_reference, make_request, make_response

# ── helpers ───────────────────────────────────────────────────────────


def _run(requests, references, responses, output_dir=None, **kwargs):
    return Evaluator(**kwargs).run(requests, references, responses, output_dir=output_dir)


# ── basic correctness ─────────────────────────────────────────────────


class TestEvaluatorCorrectness:
    def test_correct_number_answer(self):
        req = make_request()
        ref = make_reference(fmt="number", answer="3")
        resp = make_response(content="3")
        results, _ = _run([req], [ref], [resp])
        assert bool(results.loc[0, "correctness"]) is True

    def test_incorrect_number_answer(self):
        req = make_request()
        ref = make_reference(fmt="number", answer="3")
        resp = make_response(content="5")
        results, _ = _run([req], [ref], [resp])
        assert bool(results.loc[0, "correctness"]) is False

    def test_correct_binary_answer(self):
        req = make_request()
        ref = make_reference(fmt="binary", answer="yes")
        resp = make_response(content="YES")
        results, _ = _run([req], [ref], [resp])
        assert bool(results.loc[0, "correctness"]) is True

    def test_incorrect_binary_answer(self):
        req = make_request()
        ref = make_reference(fmt="binary", answer="yes")
        resp = make_response(content="no")
        results, _ = _run([req], [ref], [resp])
        assert bool(results.loc[0, "correctness"]) is False

    def test_unparseable_response_marked_incorrect(self):
        req = make_request()
        ref = make_reference(fmt="number", answer="2")
        resp = make_response(content="not a number")
        results, _ = _run([req], [ref], [resp])
        assert bool(results.loc[0, "correctness"]) is False

    def test_missing_response_marked_incorrect(self):
        req = make_request()
        ref = make_reference()
        results, _ = _run([req], [ref], [])
        assert bool(results.loc[0, "correctness"]) is False

    def test_missing_response_latency_zero(self):
        req = make_request()
        ref = make_reference()
        results, _ = _run([req], [ref], [])
        assert results.loc[0, "latency"] == pytest.approx(0.0)


# ── metadata passthrough ──────────────────────────────────────────────


class TestEvaluatorMetadata:
    def test_ood_flag_in_results(self):
        req = make_request()
        ref = make_reference(ood=True)
        resp = make_response()
        results, _ = _run([req], [ref], [resp])
        assert bool(results.loc[0, "ood"]) is True

    def test_clinical_flag_in_results(self):
        req = make_request()
        ref = make_reference(clinical=True)
        resp = make_response()
        results, _ = _run([req], [ref], [resp])
        assert bool(results.loc[0, "clinical"]) is True

    def test_latency_recorded(self):
        req = make_request()
        ref = make_reference()
        resp = make_response(latency=1.23)
        results, _ = _run([req], [ref], [resp])
        assert results.loc[0, "latency"] == pytest.approx(1.23)

    def test_primary_capability_in_results(self):
        req = make_request()
        ref = make_reference(primary=Capability.OBJECT_AGGREGATION, fmt="number", answer="0")
        resp = make_response(content="0")
        results, _ = _run([req], [ref], [resp])
        assert results.loc[0, "primary"] == Capability.OBJECT_AGGREGATION.value


# ── error handling ────────────────────────────────────────────────────


class TestEvaluatorErrors:
    def test_duplicate_response_raises(self):
        req = make_request()
        ref = make_reference()
        resp1 = make_response(content="1")
        resp2 = make_response(content="2")
        with pytest.raises(ValueError, match="Duplicate response"):
            _run([req], [ref], [resp1, resp2])

    def test_reference_without_request_raises(self):
        req = make_request(qid="q1")
        ref_a = make_reference(qid="q1")
        ref_b = make_reference(qid="q_orphan")
        with pytest.raises(ValueError, match="no matching request"):
            _run([req], [ref_a, ref_b], [])


# ── output files ──────────────────────────────────────────────────────


class TestEvaluatorOutput:
    def test_csv_files_written(self, tmp_path):
        req = make_request()
        ref = make_reference()
        resp = make_response()
        _run([req], [ref], [resp], output_dir=tmp_path)
        assert (tmp_path / "results.csv").exists()
        assert (tmp_path / "summary.csv").exists()

    def test_output_dir_created(self, tmp_path):
        out = tmp_path / "nested" / "output"
        req = make_request()
        ref = make_reference()
        resp = make_response()
        _run([req], [ref], [resp], output_dir=out)
        assert out.is_dir()

    def test_results_df_columns(self):
        req = make_request()
        ref = make_reference()
        resp = make_response()
        results, _ = _run([req], [ref], [resp])
        assert set(results.columns) >= {
            "qID",
            "video",
            "ood",
            "clinical",
            "primary",
            "answer_format",
            "latency",
            "correctness",
        }

    def test_summary_df_levels(self):
        req = make_request()
        ref = make_reference()
        resp = make_response()
        _, summary = _run([req], [ref], [resp])
        levels = set(summary["level"])
        assert "leaf" in levels
        assert "group" in levels
        assert "answer_format" in levels
        assert "overall" in levels


# ── hierarchical summary ──────────────────────────────────────────────


class TestHierarchicalSummary:
    def _make_multi(self):
        """Two questions from different groups: one correct, one wrong."""
        req_a = make_request(qid="a")
        ref_a = make_reference(
            qid="a", primary=Capability.OBJECT_IDENTIFICATION, fmt="number", answer="1"
        )
        resp_a = make_response(qid="a", content="1")  # correct

        req_b = make_request(qid="b")
        ref_b = make_reference(
            qid="b", primary=Capability.TEMPORAL_LOCALIZATION, fmt="binary", answer="yes"
        )
        resp_b = make_response(qid="b", content="no")  # wrong

        return [req_a, req_b], [ref_a, ref_b], [resp_a, resp_b]

    def test_overall_macro_average(self):
        reqs, refs, resps = self._make_multi()
        _, summary = _run(reqs, refs, resps)
        overall = summary.loc[summary["level"] == "overall", "accuracy"].iloc[0]
        # both questions come from the same video; 1 correct out of 2 → per-video mean = 0.5
        assert overall == pytest.approx(0.5)

    def test_leaf_accuracy(self):
        reqs, refs, resps = self._make_multi()
        _, summary = _run(reqs, refs, resps)
        leaf_row = summary[
            (summary["level"] == "leaf")
            & (summary["name"] == Capability.OBJECT_IDENTIFICATION.value)
        ]
        assert not leaf_row.empty
        assert leaf_row["accuracy"].iloc[0] == pytest.approx(1.0)

    def test_empty_summary_on_no_data(self):
        import pandas as pd

        summary = Evaluator()._hierarchical_summary(pd.DataFrame())
        assert summary.empty
