# tests/runners/test_text_extraction.py

from pathlib import Path

import numpy as np

from choicebench.methods.library.text_extraction import TextExtractionRunner
from choicebench.scoring.types import SCORE_CORRECT, SCORE_INCORRECT, SCORE_UNSCORABLE

from tests.runners.conftest import MockBackend

REPO_ROOT = Path(__file__).resolve().parents[2]
_PROMPTS_DIR = REPO_ROOT / "prompts"


def _no_match_embed_fn(texts: list[str]) -> np.ndarray:
    """Stub embedder: every text gets an orthogonal one-hot vector, so
    cosine similarity between any two distinct texts is always 0 --
    guarantees the cosine stage stays below threshold without needing the
    real sentence-transformers model in these fast, offline unit tests."""
    n = len(texts)
    return np.eye(n, dtype=np.float64)


def _make_runner(backend, embed_fn=_no_match_embed_fn):
    return TextExtractionRunner(
        backend=backend,
        method_name="text_extraction",
        split_name="test",
        prompt_version="v1_text_extraction",
        prompts_dir=_PROMPTS_DIR,
        run_id="test_run_001",
        seed=42,
        benchmark_name="mmlu",
        embed_fn=embed_fn,
    )


class TestTextExtractionRunOne:
    def test_one_call_per_question(self, runner_question_row):
        backend = MockBackend(responses=["HTTPS"])
        _make_runner(backend).run_one(runner_question_row, sample_index=0)
        assert len(backend.requests_received) == 1

    def test_options_are_visible_in_the_prompt(self, runner_question_row):
        """Distinguishes this from two_stage's hidden-options Stage 1."""
        backend = MockBackend(responses=["HTTPS"])
        _make_runner(backend).run_one(runner_question_row, sample_index=0)
        prompt = backend.requests_received[0]
        assert "A. FTP" in prompt
        assert "B. HTTP" in prompt
        assert "C. HTTPS" in prompt
        assert "D. SMTP" in prompt

    def test_exact_text_match_scores_correct(self, runner_question_row):
        backend = MockBackend(responses=["HTTPS"])
        result = _make_runner(backend).run_one(runner_question_row, sample_index=0)
        assert result["parsed_choice"] == "C"
        assert result["is_correct"] is True
        assert result["score_status"] == SCORE_CORRECT

    def test_incorrect_text_match(self, runner_question_row):
        backend = MockBackend(responses=["FTP"])
        result = _make_runner(backend).run_one(runner_question_row, sample_index=0)
        assert result["parsed_choice"] == "A"
        assert result["is_correct"] is False
        assert result["score_status"] == SCORE_INCORRECT

    def test_unmatched_free_text_is_unscorable(self, runner_question_row):
        backend = MockBackend(responses=["completely unrelated gibberish response"])
        result = _make_runner(backend).run_one(runner_question_row, sample_index=0)
        assert result["parsed_choice"] is None
        assert result["score_status"] == SCORE_UNSCORABLE

    def test_free_text_response_is_persisted(self, runner_question_row):
        backend = MockBackend(responses=["HTTPS"])
        result = _make_runner(backend).run_one(runner_question_row, sample_index=0)
        assert result["raw_text"] == "HTTPS"

    def test_backend_failure_returns_unscorable(self, runner_question_row):
        from choicebench.clients.types import ProviderTimeoutError
        backend = MockBackend(responses=[ProviderTimeoutError("timed out")])
        result = _make_runner(backend).run_one(runner_question_row, sample_index=0)
        assert result["parsed_choice"] is None
        assert result["transport_status"] == "failure"

    def test_defaults_to_the_real_embedder_lazily_when_none_supplied(self, runner_question_row):
        """embed_fn=None (the constructor default) must defer to
        default_embed_fn -- checked without actually invoking it, so this
        stays a fast, offline assertion about the wiring, not a real
        model-load test."""
        from choicebench.scoring.text_matcher import default_embed_fn
        runner = TextExtractionRunner(
            backend=MockBackend(responses=["HTTPS"]),
            method_name="text_extraction", split_name="test",
            prompt_version="v1_text_extraction", prompts_dir=_PROMPTS_DIR,
            run_id="test_run_001", seed=42, benchmark_name="mmlu",
        )
        assert runner._effective_embed_fn() is default_embed_fn
