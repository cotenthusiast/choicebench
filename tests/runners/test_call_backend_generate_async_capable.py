# tests/runners/test_call_backend_generate_async_capable.py
#
# Bug found while implementing Batch API support: the standalone rotation-
# rerun scripts (run_two_stage_rotations.py, run_text_extraction_rotations.py,
# run_visible_llm_matcher_rotations.py) call sync, run_one-style runner
# methods (run_stage2_rotations, run_rotations, run_matching_rotations)
# directly, bypassing run_experiment.py's run_method(), which is what
# normally decides sync-vs-async dispatch. Those sync methods go through
# _call_backend_generate -> backend.generate(), and APIBackend.generate()
# unconditionally raises RuntimeError ("call generate_batch() from an async
# context instead") -- so any of those scripts pointed at a real API-backed
# model (gpt-4.1-mini, claude-haiku, the OpenRouter/Together models) would
# crash immediately, never having been exercised against anything but
# DummyBackend in their own unit tests.
#
# Fix: _call_backend_generate detects an async-capable backend and routes
# through backend.generate_single_async() via asyncio.run() instead, so
# every existing sync-style runner method (this is base-class behavior,
# shared by every run_one implementation in the framework) works correctly
# against a real API backend called directly, not just via run_many_async.

import asyncio
from pathlib import Path

from choicebench.backends.base import BaseBackend
from choicebench.clients.types import ErrorInfo, ModelResponse, FAILURE_STATUS, SUCCESS_STATUS
from choicebench.methods.direct_mcq import DirectMCQRunner

REPO_ROOT = Path(__file__).resolve().parents[2]
_PROMPTS_DIR = REPO_ROOT / "prompts"


class _AsyncCapableBackend(BaseBackend):
    """Minimal async-capable double: generate() raises like the real
    APIBackend does; generate_single_async() is the only working path."""

    def __init__(self, response: ModelResponse) -> None:
        self._response = response
        self.async_calls_received: list[str] = []

    @property
    def model_name(self) -> str:
        return "fake-api-model"

    @property
    def provider(self) -> str:
        return "fake"

    def generate(self, prompt: str, **kwargs) -> str:
        raise RuntimeError(
            "AsyncCapableBackend.generate() cannot be called inside a running "
            "event loop. Use await backend.generate_batch([prompt]) instead."
        )

    async def generate_single_async(self, prompt: str) -> ModelResponse:
        self.async_calls_received.append(prompt)
        return self._response

    def is_async_capable(self) -> bool:
        return True


def _runner_question_row() -> dict:
    from choicebench.benchmarks.base import make_normalized_row
    return make_normalized_row(
        "computer_security",
        "Which protocol is primarily used to securely browse websites?",
        ["FTP", "HTTP", "HTTPS", "SMTP"],
        correct_index=2,
    )


def _make_runner(backend):
    return DirectMCQRunner(
        backend=backend, method_name="direct_mcq", split_name="test",
        prompt_version="v1", prompts_dir=_PROMPTS_DIR, run_id="test_run_001",
    )


class TestCallBackendGenerateRoutesAsyncCapableBackends:
    def test_run_one_succeeds_against_an_async_capable_backend(self):
        """This would previously crash: DirectMCQRunner.run_one() ->
        _call_backend_generate() -> backend.generate() -> RuntimeError."""
        response = ModelResponse(
            provider="fake", model_name="fake-api-model", status=SUCCESS_STATUS,
            latency_seconds=0.0, raw_text="C",
        )
        backend = _AsyncCapableBackend(response)
        runner = _make_runner(backend)

        result = runner.run_one(_runner_question_row(), sample_index=0)

        assert result["parsed_choice"] == "C"
        assert result["is_correct"] is True

    def test_prompt_is_routed_through_generate_single_async_not_generate(self):
        response = ModelResponse(
            provider="fake", model_name="fake-api-model", status=SUCCESS_STATUS,
            latency_seconds=0.0, raw_text="C",
        )
        backend = _AsyncCapableBackend(response)
        runner = _make_runner(backend)

        runner.run_one(_runner_question_row(), sample_index=0)

        assert len(backend.async_calls_received) == 1

    def test_a_failure_response_from_generate_single_async_is_passed_through(self):
        response = ModelResponse(
            provider="fake", model_name="fake-api-model", status=FAILURE_STATUS,
            latency_seconds=0.0,
            error=ErrorInfo("ProviderTimeoutError", "timed out", True, "provider_call"),
        )
        backend = _AsyncCapableBackend(response)
        runner = _make_runner(backend)

        result = runner.run_one(_runner_question_row(), sample_index=0)

        assert result["parsed_choice"] is None
        assert result["transport_status"] == "failure"

    def test_sync_only_backend_is_unaffected(self):
        """A non-async-capable backend (Dummy/HF/MockBackend) must keep using
        the original backend.generate() path unchanged -- this is a
        regression guard for the fix itself, not new behavior."""
        from tests.runners.conftest import MockBackend

        backend = MockBackend(responses=["C"])
        runner = _make_runner(backend)

        result = runner.run_one(_runner_question_row(), sample_index=0)

        assert result["parsed_choice"] == "C"
        assert len(backend.requests_received) == 1
