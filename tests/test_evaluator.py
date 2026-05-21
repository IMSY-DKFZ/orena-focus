"""Tests for focus.evaluation.evaluator.Evaluator."""

from unittest.mock import MagicMock, patch

import pytest

from focus.evaluation.evaluator import Evaluator
from focus.evaluation.judges import APIJudge
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


# ── API Judge Tests ───────────────────────────────────────────────────


class TestAPIJudge:
    def test_api_judge_creation(self):
        """Test that APIJudge can be instantiated with required parameters."""
        judge = APIJudge(
            api_url="https://openrouter.ai/api/v1/chat/completions",
            api_key="test-api-key",
            model_name="openai/gpt-4",
        )
        assert judge._api_url == "https://openrouter.ai/api/v1/chat/completions"
        assert judge._api_key == "test-api-key"
        assert judge._model_name == "openai/gpt-4"

    @patch("requests.post")
    def test_api_judge_correct_verdict(self, mock_post):
        """Test that APIJudge correctly identifies a correct answer."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"choices": [{"message": {"content": "CORRECT"}}]}
        mock_post.return_value = mock_response

        judge = APIJudge(
            api_url="https://api.example.com/v1/chat/completions",
            api_key="test-key",
            model_name="test-model",
        )

        req = make_request()
        ref = "The answer is yes"
        candidate = "Yes, that's correct"

        verdict = judge.judge(req, ref, candidate)
        assert verdict is True

    @patch("requests.post")
    def test_api_judge_incorrect_verdict(self, mock_post):
        """Test that APIJudge correctly identifies an incorrect answer."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"choices": [{"message": {"content": "INCORRECT"}}]}
        mock_post.return_value = mock_response

        judge = APIJudge(
            api_url="https://api.example.com/v1/chat/completions",
            api_key="test-key",
            model_name="test-model",
        )

        req = make_request()
        ref = "The answer is yes"
        candidate = "No, that's wrong"

        verdict = judge.judge(req, ref, candidate)
        assert verdict is False

    @patch("requests.post")
    def test_api_judge_with_extra_headers(self, mock_post):
        """Test that APIJudge includes extra headers in API requests."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"choices": [{"message": {"content": "CORRECT"}}]}
        mock_post.return_value = mock_response

        extra_headers = {
            "X-OpenRouter-Title": "Test Site",
            "HTTP-Referer": "https://example.com",
        }

        judge = APIJudge(
            api_url="https://openrouter.ai/api/v1/chat/completions",
            api_key="test-key",
            model_name="openai/gpt-4",
            extra_headers=extra_headers,
        )

        req = make_request()
        judge.judge(req, "ref", "candidate")

        # Verify that the headers were included in the POST request
        call_kwargs = mock_post.call_args[1]
        assert "X-OpenRouter-Title" in call_kwargs["headers"]
        assert call_kwargs["headers"]["X-OpenRouter-Title"] == "Test Site"
        assert "HTTP-Referer" in call_kwargs["headers"]

    @patch("requests.post")
    def test_api_judge_with_extra_body_params(self, mock_post):
        """Test that APIJudge includes extra body parameters in requests."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"choices": [{"message": {"content": "CORRECT"}}]}
        mock_post.return_value = mock_response

        extra_params = {"temperature": 0.5, "top_p": 0.9}

        judge = APIJudge(
            api_url="https://api.example.com/v1/chat/completions",
            api_key="test-key",
            model_name="test-model",
            extra_body_params=extra_params,
        )

        req = make_request()
        judge.judge(req, "ref", "candidate")

        # Verify that the extra params were included in the POST body
        call_kwargs = mock_post.call_args[1]
        body = call_kwargs["json"]
        assert body.get("temperature") == 0.5
        assert body.get("top_p") == 0.9

    @patch("requests.post")
    def test_api_judge_retry_on_failure(self, mock_post):
        """Test that APIJudge retries on request failure."""
        from requests.exceptions import ConnectionError

        # First two attempts fail, third succeeds
        mock_response = MagicMock()
        mock_response.json.return_value = {"choices": [{"message": {"content": "CORRECT"}}]}
        mock_post.side_effect = [
            ConnectionError("Connection failed"),
            ConnectionError("Connection failed"),
            mock_response,
        ]

        judge = APIJudge(
            api_url="https://api.example.com/v1/chat/completions",
            api_key="test-key",
            model_name="test-model",
            max_retries=3,
        )

        req = make_request()
        verdict = judge.judge(req, "ref", "candidate")

        assert verdict is True
        assert mock_post.call_count == 3

    @patch("requests.post")
    def test_api_judge_exhausts_retries(self, mock_post):
        """Test that APIJudge returns False when all retries are exhausted."""
        from requests.exceptions import ConnectionError

        mock_post.side_effect = ConnectionError("Connection failed")

        judge = APIJudge(
            api_url="https://api.example.com/v1/chat/completions",
            api_key="test-key",
            model_name="test-model",
            max_retries=2,
        )

        req = make_request()
        verdict = judge.judge(req, "ref", "candidate")

        assert verdict is False
        assert mock_post.call_count == 2

    def test_evaluator_with_api_judge(self):
        """Test that Evaluator works with APIJudge for open-ended questions."""
        with patch("requests.post") as mock_post:
            mock_response = MagicMock()
            mock_response.json.return_value = {"choices": [{"message": {"content": "CORRECT"}}]}
            mock_post.return_value = mock_response

            api_judge = APIJudge(
                api_url="https://api.example.com/v1/chat/completions",
                api_key="test-key",
                model_name="test-model",
            )

            evaluator = Evaluator(judges=[api_judge])

            # Create open-ended question
            req = make_request()
            ref = make_reference(fmt="open_ended", answer="sample answer")
            resp = make_response(content="sample answer")

            results, _ = evaluator.run([req], [ref], [resp])
            assert bool(results.loc[0, "correctness"]) is True


class TestEvaluatorThreading:
    """Test that single-thread and multi-thread evaluation produce identical results."""

    def test_api_judge_deterministic_with_threading(self):
        """Verify that num_workers=1 and num_workers>1 produce same results with APIJudge."""
        with patch("requests.post") as mock_post:
            mock_response = MagicMock()
            mock_response.json.return_value = {"choices": [{"message": {"content": "CORRECT"}}]}
            mock_post.return_value = mock_response

            api_judge = APIJudge(
                api_url="https://api.example.com/v1/chat/completions",
                api_key="test-key",
                model_name="test-model",
            )

            # Create multiple questions with different formats
            requests_list = [make_request(qid=f"q{i}") for i in range(5)]
            references_list = [
                make_reference(fmt="open_ended", answer=f"ans{i}", qid=f"q{i}") for i in range(5)
            ]
            responses_list = [make_response(content=f"ans{i}", qid=f"q{i}") for i in range(5)]

            # Run with sequential evaluation
            evaluator_seq = Evaluator(judges=[api_judge], num_workers=1)
            results_seq, summary_seq = evaluator_seq.run(
                requests_list, references_list, responses_list
            )

            # Reset mock call count
            mock_post.reset_mock()

            # Run with parallel evaluation
            evaluator_par = Evaluator(judges=[api_judge], num_workers=4)
            results_par, summary_par = evaluator_par.run(
                requests_list, references_list, responses_list
            )

            # Verify results are identical
            assert len(results_seq) == len(results_par)
            assert (results_seq["correctness"] == results_par["correctness"]).all()
            assert (results_seq["qID"] == results_par["qID"]).all()
            assert (results_seq["answer_format"] == results_par["answer_format"]).all()

            # Verify summary is identical
            assert len(summary_seq) == len(summary_par)
            assert (summary_seq["accuracy"].values == summary_par["accuracy"].values).all()
            assert (summary_seq["count"].values == summary_par["count"].values).all()

    def test_mixed_formats_deterministic_with_threading(self):
        """Test determinism with mixed answer formats (binary, number, open-ended)."""
        with patch("requests.post") as mock_post:
            mock_response = MagicMock()
            mock_response.json.return_value = {"choices": [{"message": {"content": "CORRECT"}}]}
            mock_post.return_value = mock_response

            api_judge = APIJudge(
                api_url="https://api.example.com/v1/chat/completions",
                api_key="test-key",
                model_name="test-model",
            )

            # Create requests with mixed formats
            requests_list = [
                make_request(qid="q1"),
                make_request(qid="q2"),
                make_request(qid="q3"),
                make_request(qid="q4"),
            ]

            references_list = [
                make_reference(fmt="binary", answer="yes", qid="q1"),
                make_reference(fmt="number", answer="42", qid="q2"),
                make_reference(fmt="open_ended", answer="surgical answer", qid="q3"),
                make_reference(fmt="binary", answer="no", qid="q4"),
            ]

            responses_list = [
                make_response(content="yes", qid="q1"),
                make_response(content="42", qid="q2"),
                make_response(content="surgical answer", qid="q3"),
                make_response(content="no", qid="q4"),
            ]

            # Sequential
            evaluator_seq = Evaluator(judges=[api_judge], num_workers=1)
            results_seq, _ = evaluator_seq.run(requests_list, references_list, responses_list)

            mock_post.reset_mock()

            # Parallel
            evaluator_par = Evaluator(judges=[api_judge], num_workers=4)
            results_par, _ = evaluator_par.run(requests_list, references_list, responses_list)

            # Compare results
            assert len(results_seq) == len(results_par) == 4
            assert (results_seq["correctness"] == results_par["correctness"]).all()
            assert (results_seq["answer_format"] == results_par["answer_format"]).all()
