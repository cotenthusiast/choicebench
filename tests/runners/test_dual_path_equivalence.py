# tests/runners/test_dual_path_equivalence.py
#
# BATCH 8 — A4: sync/async dual-path equivalence.
#
# Invariant: for a given question set and a deterministic backend, a
# runner's sync path (run_one per row) and async batch path
# (run_many_async) produce equivalent result rows — same answers, same
# scores, same prompts — and the backend receives the same total call
# volume. Guards the async-only reassembly logic (fan-out indexing,
# phase batching) that every API-backed production run executes.

import asyncio
import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from choicebench.backends.api_backend import APIBackend
from choicebench.clients.types import (
    ErrorInfo,
    ModelResponse,
    SUCCESS_STATUS,
)
from choicebench.methods.library.permutation import PermutationRunner
from choicebench.methods.library.two_stage import TwoStageRunner

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PROMPTS_DIR = _REPO_ROOT / "prompts"

# Fields that must agree between paths (runtime-only fields like latency
# are excluded by construction of this comparison).
_COMPARED_FIELDS = [
    "run_id", "question_id", "split_name", "subject", "method_name",
    "prompt_version", "sample_index",
    "provider", "model_name", "temperature", "max_tokens", "seed",
    "question_text", "prompt",
    "transport_status", "answer_status", "parse_status", "score_status",
    "parsed_choice", "is_correct",
]


def _response_for(prompt: str, raw_text: str | None) -> ModelResponse:
    return ModelResponse(
        provider="openai",
        model_name="mock-model",
        status=SUCCESS_STATUS if raw_text is not None else "error",
        latency_seconds=0.0,
        raw_text=raw_text,
        finish_reason=None,
        usage=None,
        error=None if raw_text is not None else ErrorInfo("X", "x", True, "p"),
        timestamp_utc=None,
    )


class _DeterministicAsyncBackend(APIBackend):
    """APIBackend double whose response is a pure function of the prompt.

    Because both execution paths send byte-identical prompts, a pure
    function guarantees the two paths observe identical responses; any
    row divergence therefore comes from the runner's own reassembly.
    """

    def __init__(self, responder) -> None:
        self._responder = responder
        self.prompts_received: list[str] = []
        self.batch_calls = 0

    @property
    def provider(self) -> str:
        return "openai"

    @property
    def model_name(self) -> str:
        return "mock-model"

    def generate(self, prompt: str, **kwargs) -> str:
        self.prompts_received.append(prompt)
        return self._responder(prompt)

    async def generate_batch(self, prompts: list[str]) -> list[ModelResponse]:
        self.batch_calls += 1
        self.prompts_received.extend(prompts)
        return [_response_for(p, self._responder(p)) for p in prompts]


def _letter_from_prompt_hash(prompt: str) -> str:
    digest = hashlib.md5(prompt.encode("utf-8")).hexdigest()
    return "A" if int(digest, 16) % 2 == 0 else "C"


def _question_df(n: int = 6) -> pd.DataFrame:
    rows = []
    choices = ["FTP", "HTTP", "HTTPS", "SMTP"]
    for i in range(n):
        rotated = choices[i % 4:] + choices[: i % 4]
        rows.append({
            "question_id": f"q{i:04d}",
            "subject": "computer_security",
            "question_text": f"Question {i}: which protocol secures web traffic?",
            "choices_json": json.dumps(
                [{"text": t, "source_index": j} for j, t in enumerate(rotated)]
            ),
            "correct_option": "A",
            "correct_answer_text": rotated[0],
        })
    return pd.DataFrame(rows)


def _make_permutation_runner(backend):
    return PermutationRunner(
        backend=backend, method_name="cyclic_permutation", split_name="test",
        prompt_version="v1", prompts_dir=_PROMPTS_DIR,
        run_id="dualpath", temperature=0.0, max_tokens=16, seed=42,
    )


def _make_two_stage_runner(backend):
    return TwoStageRunner(
        backend=backend, method_name="two_stage", split_name="test",
        prompt_version="v1", prompts_dir=_PROMPTS_DIR,
        run_id="dualpath", temperature=0.0, max_tokens=16, seed=42,
        fallback_on_parse_failure=True,
    )


def _assert_rows_equivalent(sync_rows, async_rows, context):
    assert len(sync_rows) == len(async_rows), context
    for s_row, a_row in zip(sync_rows, async_rows):
        for field in _COMPARED_FIELDS:
            assert s_row[field] == a_row[field], (
                f"{context}: field {field!r} diverged for "
                f"{s_row['question_id']}: {s_row[field]!r} != {a_row[field]!r}"
            )


class TestPermutationRunnerDualPath:
    def test_sync_and_async_rows_are_equivalent(self):
        df = _question_df()

        sync_backend = _DeterministicAsyncBackend(_letter_from_prompt_hash)
        runner_sync = _make_permutation_runner(sync_backend)
        sync_rows = [
            runner_sync.run_one(row, i)
            for i, (_, row) in enumerate(df.iterrows())
        ]

        async_backend = _DeterministicAsyncBackend(_letter_from_prompt_hash)
        runner_async = _make_permutation_runner(async_backend)
        async_rows = asyncio.run(runner_async.run_many_async(df))

        _assert_rows_equivalent(
            sync_rows, async_rows, "PermutationRunner dual-path"
        )
        # Same questions fan out to the same rotation prompts either way.
        assert sorted(sync_backend.prompts_received) == sorted(
            async_backend.prompts_received
        )
        # One batched call on the async path; n_questions * n_rotations
        # calls total on both paths.
        n_expected = len(df) * 4
        assert len(sync_backend.prompts_received) == n_expected
        assert len(async_backend.prompts_received) == n_expected
        assert async_backend.batch_calls == 1


class TestTwoStageRunnerDualPath:
    def test_sync_and_async_rows_are_equivalent_including_fallback(self):
        df = _question_df(4)

        def responder(prompt: str) -> str:
            if "Select the option that best matches" in prompt:
                # Stage 2 unparseable in BOTH paths -> fallback must fire.
                return "!!!unparseable!!!"
            if "Respond with a short direct answer only." in prompt:
                return "HTTPS"
            # direct_mcq fallback prompt
            return _letter_from_prompt_hash(prompt)

        sync_backend = _DeterministicAsyncBackend(responder)
        runner_sync = _make_two_stage_runner(sync_backend)
        sync_rows = [
            runner_sync.run_one(row, i)
            for i, (_, row) in enumerate(df.iterrows())
        ]
        assert all(r["fallback_used"] is True for r in sync_rows)

        async_backend = _DeterministicAsyncBackend(responder)
        runner_async = _make_two_stage_runner(async_backend)
        async_rows = asyncio.run(runner_async.run_many_async(df))

        assert all(r["fallback_used"] is True for r in async_rows)
        assert [r["fallback_used"] for r in sync_rows] == [
            r["fallback_used"] for r in async_rows
        ]
        assert [r.get("free_text_response") for r in sync_rows] == [
            r.get("free_text_response") for r in async_rows
        ]
        _assert_rows_equivalent(
            sync_rows, async_rows, "TwoStageRunner dual-path"
        )
        # Total call volume matches even though batching differs
        # (3 calls/question here: stage1 + stage2 + fallback).
        assert len(sync_backend.prompts_received) == 3 * len(df)
        assert len(async_backend.prompts_received) == 3 * len(df)
