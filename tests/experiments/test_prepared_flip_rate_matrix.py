from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest
import yaml

from choicebench.clients.types import ModelRequest, ModelResponse
from experiments.flip_rate_traces.data_source import FrozenQuestion
from experiments.flip_rate_traces.prepared_runner import (
    CsvResponseReuse,
    LegacyResponseCache,
    _expand_path,
    _historical_option_map,
    build_option_matching_prompt,
    fill_missing_responses,
    load_matrix_config,
    load_questions_from_csv,
    load_stage1_outputs,
)


def _question(*, choice_d: str = "") -> FrozenQuestion:
    return FrozenQuestion(
        question_id="qid",
        subject="subject",
        question_text="Which option?",
        choice_a="alpha",
        choice_b="beta",
        choice_c="gamma",
        choice_d=choice_d,
        correct_option="A",
    )


def test_cloud_three_option_two_prompt_preserves_blank_d_line() -> None:
    template = (
        "You are given a question, a reference answer, and four options.\n\n"
        "Question: {question}\n\n"
        "Reference answer: {free_text}\n\n"
        "Options:\n"
        "A. {option_a}\n"
        "B. {option_b}\n"
        "C. {option_c}\n"
        "D. {option_d}\n\n"
        "Respond with only the letter."
    )

    prompt = build_option_matching_prompt(
        template,
        _question(),
        "alpha",
        {"A": "alpha", "B": "beta", "C": "gamma"},
    )

    assert "four options" in prompt
    assert "\nC. gamma\nD. \n\nRespond" in prompt
    assert not prompt.endswith("\n")


def test_matrix_config_rejects_standard_run_above_500_tokens(tmp_path: Path) -> None:
    config = {
        "schema_version": "flip_rate_traces.prepared.v1",
        "model": {
            "name": "gemini-2.5-flash",
            "provider": "gemini",
            "backend": "api",
            "concurrency": 1,
            "min_delay_seconds": 1.5,
            "max_retries": 5,
            "timeout_seconds": 30,
        },
        "defaults": {
            "temperature": 0.0,
            "max_tokens": 501,
            "seed": 42,
            "prompt_version": "v1",
            "top_p": None,
            "stop": None,
        },
        "jobs": [],
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))

    with pytest.raises(ValueError, match="standard max_tokens must be 500"):
        load_matrix_config(path)


def test_matrix_config_allows_4000_only_for_cot_stage1(tmp_path: Path) -> None:
    config = {
        "schema_version": "flip_rate_traces.prepared.v1",
        "model": {
            "name": "gemini-2.5-flash",
            "provider": "gemini",
            "backend": "api",
            "concurrency": 1,
            "min_delay_seconds": 1.5,
            "max_retries": 5,
            "timeout_seconds": 30,
        },
        "defaults": {
            "temperature": 0.0,
            "max_tokens": 500,
            "seed": 42,
            "prompt_version": "v1",
            "top_p": None,
            "stop": None,
        },
        "jobs": [
            {
                "condition": "two_prompt_cot",
                "stage1": {"mode": "generate", "max_tokens": 4000},
                "stage2_max_tokens": 500,
                "subset": {
                    "path": "ids.json",
                    "sha256": "file-hash",
                    "ids_newline_sha256": "ids-hash",
                    "expected_count": 250,
                    "expected_subject_count": 50,
                    "expected_per_subject": 5,
                    "selection_seed": 42,
                },
            }
        ],
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))

    loaded = load_matrix_config(path)

    assert loaded["jobs"][0]["stage1"]["max_tokens"] == 4000
    assert loaded["jobs"][0]["stage2_max_tokens"] == 500


def test_matrix_config_rejects_4000_for_cot_stage2(tmp_path: Path) -> None:
    config = {
        "schema_version": "flip_rate_traces.prepared.v1",
        "model": {
            "name": "gemini-2.5-flash",
            "provider": "gemini",
            "backend": "api",
            "concurrency": 1,
            "min_delay_seconds": 1.5,
            "max_retries": 5,
            "timeout_seconds": 30,
        },
        "defaults": {
            "temperature": 0.0,
            "max_tokens": 500,
            "seed": 42,
            "prompt_version": "v1",
            "top_p": None,
            "stop": None,
        },
        "jobs": [
            {
                "condition": "two_prompt_cot",
                "stage1": {"mode": "generate", "max_tokens": 4000},
                "stage2_max_tokens": 4000,
                "subset": {
                    "path": "ids.json",
                    "sha256": "file-hash",
                    "ids_newline_sha256": "ids-hash",
                    "expected_count": 250,
                    "expected_subject_count": 50,
                    "expected_per_subject": 5,
                    "selection_seed": 42,
                },
            }
        ],
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))

    with pytest.raises(ValueError, match="CoT Stage 2 max_tokens must be 500"):
        load_matrix_config(path)


def test_frozen_subset_is_loaded_in_id_file_order(tmp_path: Path) -> None:
    csv_path = tmp_path / "questions.csv"
    fields = [
        "question_id",
        "subject",
        "question_text",
        "choice_a",
        "choice_b",
        "choice_c",
        "choice_d",
        "correct_option",
    ]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for qid in ("q1", "q2", "q3"):
            writer.writerow(
                {
                    "question_id": qid,
                    "subject": "s",
                    "question_text": qid,
                    "choice_a": "a",
                    "choice_b": "b",
                    "choice_c": "c",
                    "choice_d": "d",
                    "correct_option": "A",
                }
            )
    subset = tmp_path / "subset.json"
    subset.write_text(json.dumps(["q3", "q1"]))

    questions = load_questions_from_csv(
        csv_path,
        hashlib.sha256(csv_path.read_bytes()).hexdigest(),
        subset_ids_path=subset,
    )

    assert [q.question_id for q in questions] == ["q3", "q1"]


def test_legacy_cache_reuses_only_exact_request_key(tmp_path: Path) -> None:
    request = ModelRequest(
        provider="groq",
        model_name="llama-3.1-8b-instant",
        payload="prompt",
        temperature=0.0,
        max_tokens=500,
        seed=42,
        request_logprobs=False,
    )
    key_data = {
        "provider": request.provider,
        "model_name": request.model_name,
        "prompt": request.payload,
        "temperature": request.temperature,
        "max_tokens": request.max_tokens,
        "seed": request.seed,
        "request_logprobs": False,
    }
    key = hashlib.sha256(json.dumps(key_data, sort_keys=True).encode()).hexdigest()
    entry = tmp_path / key[:2] / f"{key}.json"
    entry.parent.mkdir()
    entry.write_text(
        json.dumps(
            {
                "raw_text": "A",
                "finish_reason": "stop",
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 1,
                    "total_tokens": 11,
                },
                "logprobs": None,
            }
        )
    )

    response = LegacyResponseCache(tmp_path).get(request)
    changed = ModelRequest(
        provider=request.provider,
        model_name=request.model_name,
        payload=request.payload + "\n",
        temperature=request.temperature,
        max_tokens=request.max_tokens,
        seed=request.seed,
        request_logprobs=False,
    )

    assert response is not None
    assert response.raw_text == "A"
    assert response.finish_reason == "stop"
    assert response.latency_seconds == 0.0
    assert LegacyResponseCache(tmp_path).get(changed) is None


def test_csv_reuse_requires_exact_question_and_prompt(tmp_path: Path) -> None:
    source = tmp_path / "reuse.csv"
    with source.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "question_id",
                "prompt",
                "raw_text",
                "finish_reason",
                "model_status",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "question_id": "q1",
                "prompt": "exact prompt",
                "raw_text": "B",
                "finish_reason": "stop",
                "model_status": "success",
            }
        )

    reuse = CsvResponseReuse(
        source,
        hashlib.sha256(source.read_bytes()).hexdigest(),
        provider="gemini",
        model_name="gemini-2.5-flash",
    )

    assert reuse.get("q1", "exact prompt").raw_text == "B"
    assert reuse.get("q1", "exact prompt\n") is None
    assert reuse.get("q2", "exact prompt") is None


@pytest.mark.asyncio
async def test_fill_missing_responses_generates_only_cache_misses() -> None:
    requests = [
        ModelRequest(
            provider="groq",
            model_name="llama-3.1-8b-instant",
            payload=f"prompt-{index}",
            temperature=0.0,
            max_tokens=500,
            seed=42,
        )
        for index in range(3)
    ]
    recovered = ModelResponse(
        provider="groq",
        model_name="llama-3.1-8b-instant",
        status="success",
        latency_seconds=0.0,
        raw_text="A",
    )
    sent: list[str] = []

    async def generate_batch(missing: list[ModelRequest]) -> list[ModelResponse]:
        sent.extend(request.payload for request in missing)
        return [
            ModelResponse(
                provider=request.provider,
                model_name=request.model_name,
                status="success",
                latency_seconds=1.0,
                raw_text="B",
            )
            for request in missing
        ]

    responses, reused = await fill_missing_responses(
        requests,
        [recovered, None, None],
        generate_batch,
    )

    assert sent == ["prompt-1", "prompt-2"]
    assert [response.raw_text for response in responses] == ["A", "B", "B"]
    assert reused == [True, False, False]


def test_stage1_loader_requires_every_requested_question(tmp_path: Path) -> None:
    source = tmp_path / "stage1.csv"
    source.write_text(
        "question_id,free_text_response\nq1,answer one\n",
        encoding="utf-8",
    )
    digest = hashlib.sha256(source.read_bytes()).hexdigest()

    assert load_stage1_outputs(source, digest, ["q1"]) == {"q1": "answer one"}
    with pytest.raises(ValueError, match="missing complete Stage-1 outputs"):
        load_stage1_outputs(source, digest, ["q1", "q2"])


def test_historical_option_map_preserves_frozen_whitespace() -> None:
    question = _question(choice_d="delta  ")
    question = FrozenQuestion(
        **{
            **question.__dict__,
            "choice_b": "two\u00a0\u00a0columns",
            "choice_c": "aligned   text",
        }
    )

    options = _historical_option_map(question)

    assert options == {
        "A": "alpha",
        "B": "two\u00a0\u00a0columns",
        "C": "aligned   text",
        "D": "delta  ",
    }


def test_expand_path_fails_closed_on_unset_environment_variable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("MISSING_FLIP_RATE_PATH", raising=False)

    with pytest.raises(ValueError, match="unresolved environment variable"):
        _expand_path("${MISSING_FLIP_RATE_PATH}/cache")
