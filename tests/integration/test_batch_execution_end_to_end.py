# tests/integration/test_batch_execution_end_to_end.py
#
# End-to-end smoke test for the Batch execution stack: a real runner
# (PermutationRunner / cyclic_permutation) driving a real BatchAPIBackend
# wrapping a real OpenAIClient, with only the OpenAI SDK's network calls
# mocked (AsyncMock on client.files.create/batches.create/batches.retrieve/
# files.content) -- never a real API key. Each layer already has its own
# unit tests; this proves they actually fit together: run_many_async() ->
# generate_batch() -> submit_batch()/poll_batch()/fetch_batch_results() ->
# reassembled ModelResponses -> parsed/scored/voted result rows.

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pandas as pd
import pytest

from choicebench.backends.api_backend import APIBackend
from choicebench.backends.batch_api_backend import BatchAPIBackend
from choicebench.clients.openai_client import OpenAIClient
from choicebench.methods.library.permutation import PermutationRunner
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
_PROMPTS_DIR = REPO_ROOT / "prompts"


def _responses_api_body(text: str) -> dict:
    return {
        "id": "resp_abc", "object": "response", "created_at": 1234567890,
        "status": "completed", "model": "gpt-4.1-mini",
        "output": [{
            "type": "message", "id": "msg_1", "status": "completed", "role": "assistant",
            "content": [{"type": "output_text", "text": text, "annotations": []}],
        }],
        "parallel_tool_calls": True, "tool_choice": "auto", "tools": [],
        "usage": {
            "input_tokens": 10, "output_tokens": 2, "total_tokens": 12,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens_details": {"reasoning_tokens": 0},
        },
    }


def _output_line(custom_id: str, text: str) -> str:
    return json.dumps({
        "custom_id": custom_id,
        "response": {"status_code": 200, "request_id": "req", "body": _responses_api_body(text)},
        "error": None,
    })


@pytest.mark.asyncio
async def test_cyclic_permutation_runs_end_to_end_through_the_real_batch_stack(tmp_path):
    client = OpenAIClient(model_name="gpt-4.1-mini", api_key="test-key")

    # Each rotation must respond with whichever DISPLAY letter currently
    # holds "HTTPS" under that rotation, not always "C" -- the display
    # position for a given canonical content shifts every rotation (that
    # is the entire point of cyclic permutation). This exercises the same
    # positional bijection PermutationRunner's own unit tests pin.
    canonical = {"A": "FTP", "B": "HTTP", "C": "HTTPS", "D": "SMTP"}
    rotations = [r.mapping for r in PermutationRunner._generate_rotations(canonical)]
    https_letters = []
    for rotation in rotations:
        for letter, text in rotation.items():
            if text == "HTTPS":
                https_letters.append(letter)
                break

    client.client.files.create = AsyncMock(return_value=SimpleNamespace(id="file_abc"))
    client.client.batches.create = AsyncMock(return_value=SimpleNamespace(id="batch_abc"))
    client.client.batches.retrieve = AsyncMock(return_value=SimpleNamespace(
        status="completed", output_file_id="out_1", error_file_id=None,
    ))
    output_jsonl = "\n".join(
        _output_line(str(i), letter) for i, letter in enumerate(https_letters)
    )
    client.client.files.content = AsyncMock(return_value=SimpleNamespace(text=output_jsonl))

    backend = BatchAPIBackend(
        provider="openai", model_name="gpt-4.1-mini", client=client,
        cache_dir=tmp_path / "cache", temperature=0.0, max_tokens=64, seed=42,
        batch_state_dir=tmp_path / "batch_state", poll_interval_seconds=0.0,
    )

    runner = PermutationRunner(
        backend=backend, method_name="cyclic_permutation", split_name="test",
        prompt_version="v1", prompts_dir=_PROMPTS_DIR, run_id="batch_smoke_test",
        seed=42, benchmark_name="mmlu",
    )

    df = pd.DataFrame([{
        "question_id": "q1", "subject": "computer_security",
        "question_text": "Which protocol is primarily used to securely browse websites?",
        "choices_json": json.dumps([
            {"text": "FTP", "source_index": 0}, {"text": "HTTP", "source_index": 1},
            {"text": "HTTPS", "source_index": 2}, {"text": "SMTP", "source_index": 3},
        ]),
        "correct_option": "C",
    }])

    results = await runner.run_many_async(df)

    assert len(results) == 1
    # All 4 rotation prompts went through ONE real batch submission, not 4
    # individual calls.
    assert client.client.batches.create.await_count == 1
    assert len(client.client.files.create.call_args.kwargs["file"][1].decode("utf-8").strip().splitlines()) == 4
    assert results[0]["parsed_choice"] == "C"
    assert results[0]["is_correct"] is True
    per_rotation = json.loads(results[0]["per_rotation_choices_json"])
    assert per_rotation == ["C", "C", "C", "C"]


@pytest.mark.asyncio
async def test_a_failed_batch_job_produces_a_scorable_but_incorrect_row_not_a_crash(tmp_path):
    client = OpenAIClient(model_name="gpt-4.1-mini", api_key="test-key")
    client.client.files.create = AsyncMock(return_value=SimpleNamespace(id="file_abc"))
    client.client.batches.create = AsyncMock(return_value=SimpleNamespace(id="batch_abc"))
    client.client.batches.retrieve = AsyncMock(return_value=SimpleNamespace(status="failed"))

    backend = BatchAPIBackend(
        provider="openai", model_name="gpt-4.1-mini", client=client,
        cache_dir=tmp_path / "cache", temperature=0.0, max_tokens=64, seed=42,
        batch_state_dir=tmp_path / "batch_state", poll_interval_seconds=0.0,
    )
    runner = PermutationRunner(
        backend=backend, method_name="cyclic_permutation", split_name="test",
        prompt_version="v1", prompts_dir=_PROMPTS_DIR, run_id="batch_smoke_test",
        seed=42, benchmark_name="mmlu",
    )
    df = pd.DataFrame([{
        "question_id": "q1", "subject": "computer_security",
        "question_text": "Which protocol is primarily used to securely browse websites?",
        "choices_json": json.dumps([
            {"text": "FTP", "source_index": 0}, {"text": "HTTP", "source_index": 1},
            {"text": "HTTPS", "source_index": 2}, {"text": "SMTP", "source_index": 3},
        ]),
        "correct_option": "C",
    }])

    results = await runner.run_many_async(df)

    assert len(results) == 1
    assert results[0]["parsed_choice"] is None
    assert results[0]["answer_status"] == "failure"


def _question_df() -> pd.DataFrame:
    return pd.DataFrame([{
        "question_id": "q1", "subject": "computer_security",
        "question_text": "Which protocol is primarily used to securely browse websites?",
        "choices_json": json.dumps([
            {"text": "FTP", "source_index": 0}, {"text": "HTTP", "source_index": 1},
            {"text": "HTTPS", "source_index": 2}, {"text": "SMTP", "source_index": 3},
        ]),
        "correct_option": "C",
    }])


def _https_letters() -> list[str]:
    canonical = {"A": "FTP", "B": "HTTP", "C": "HTTPS", "D": "SMTP"}
    rotations = [r.mapping for r in PermutationRunner._generate_rotations(canonical)]
    letters = []
    for rotation in rotations:
        for letter, text in rotation.items():
            if text == "HTTPS":
                letters.append(letter)
                break
    return letters


@pytest.mark.asyncio
async def test_sync_and_batch_backends_produce_identical_result_row_schema(tmp_path):
    """Batch vs synchronous transport must never create a different
    downstream schema -- a method's own code (and any analysis reading
    its output) must never need to branch on which one produced a row."""
    https_letters = _https_letters()

    # Batch path.
    batch_client = OpenAIClient(model_name="gpt-4.1-mini", api_key="test-key")
    batch_client.client.files.create = AsyncMock(return_value=SimpleNamespace(id="file_abc"))
    batch_client.client.batches.create = AsyncMock(return_value=SimpleNamespace(id="batch_abc"))
    batch_client.client.batches.retrieve = AsyncMock(return_value=SimpleNamespace(
        status="completed", output_file_id="out_1", error_file_id=None,
    ))
    batch_client.client.files.content = AsyncMock(return_value=SimpleNamespace(
        text="\n".join(_output_line(str(i), letter) for i, letter in enumerate(https_letters))
    ))
    batch_backend = BatchAPIBackend(
        provider="openai", model_name="gpt-4.1-mini", client=batch_client,
        cache_dir=tmp_path / "batch_cache", temperature=0.0, max_tokens=64, seed=42,
        batch_state_dir=tmp_path / "batch_state", poll_interval_seconds=0.0,
    )
    batch_runner = PermutationRunner(
        backend=batch_backend, method_name="cyclic_permutation", split_name="test",
        prompt_version="v1", prompts_dir=_PROMPTS_DIR, run_id="schema_test",
        seed=42, benchmark_name="mmlu",
    )
    batch_results = await batch_runner.run_many_async(_question_df())

    # Synchronous path -- same client class, same underlying model, but
    # via generate()/responses.create() (concurrent dispatch), never the
    # batch endpoints.
    sync_client = OpenAIClient(model_name="gpt-4.1-mini", api_key="test-key")
    call_index = {"i": 0}

    async def _fake_create(**kwargs):
        letter = https_letters[call_index["i"] % len(https_letters)]
        call_index["i"] += 1
        return SimpleNamespace(output_text=letter, usage=None)

    sync_client.client.responses.create = _fake_create
    sync_backend = APIBackend(
        "openai", "gpt-4.1-mini", sync_client,
        tmp_path / "sync_cache", 0.0, 64, 42, 10, None,
    )
    sync_runner = PermutationRunner(
        backend=sync_backend, method_name="cyclic_permutation", split_name="test",
        prompt_version="v1", prompts_dir=_PROMPTS_DIR, run_id="schema_test",
        seed=42, benchmark_name="mmlu",
    )
    sync_results = await sync_runner.run_many_async(_question_df())

    assert len(batch_results) == len(sync_results) == 1
    assert set(batch_results[0].keys()) == set(sync_results[0].keys())
    for field in ["parsed_choice", "is_correct", "answer_status", "score_status"]:
        assert batch_results[0][field] == sync_results[0][field], field
