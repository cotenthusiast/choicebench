"""Prepared remaining-cell runner for the flip-rate trace experiment.

This module is intentionally import-safe: loading and validating configs,
building prompts, and inspecting frozen reuse artifacts do not construct a
provider client or load a local model. Network/local inference begins only
when ``run_matrix`` is called by ``run_prepared_matrix.py``.
"""

from __future__ import annotations

import asyncio
import csv
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable

import yaml

from choicebench.clients.types import (
    ErrorInfo,
    ModelRequest,
    ModelResponse,
    UsageInfo,
)
from choicebench.infra.checkpoint import CheckpointManager

from experiments.flip_rate_traces.data_source import FrozenQuestion
from experiments.flip_rate_traces.historical_protocol import (
    build_api_prompt,
    build_local_prompt,
    generate_permutations,
)
from experiments.flip_rate_traces.runner import TraceRetainingCyclicRunner
from experiments.flip_rate_traces.trace_schema import (
    TRACE_COLUMNS,
    validate_question_traces,
)


PREPARED_SCHEMA_VERSION = "flip_rate_traces.prepared.v1"
STANDARD_MAX_TOKENS = 500
COT_STAGE1_ALLOWED_MAX_TOKENS = frozenset({1500, 4000})


def _expand_path(path: str | Path, *, base_dir: Path | None = None) -> Path:
    raw_path = str(path)
    expanded_text = os.path.expandvars(raw_path)
    if "$" in expanded_text:
        raise ValueError(f"unresolved environment variable in path: {raw_path}")
    expanded = Path(expanded_text).expanduser()
    if not expanded.is_absolute() and base_dir is not None:
        expanded = base_dir / expanded
    return expanded.resolve()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_matrix_config(path: str | Path) -> dict[str, Any]:
    """Load and validate one prepared matrix configuration."""
    config_path = Path(path).resolve()
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise ValueError("prepared config must contain a YAML mapping")
    if cfg.get("schema_version") != PREPARED_SCHEMA_VERSION:
        raise ValueError(
            f"schema_version must be {PREPARED_SCHEMA_VERSION!r}"
        )

    defaults = cfg.get("defaults")
    if not isinstance(defaults, dict):
        raise ValueError("prepared config requires a defaults mapping")
    if defaults.get("max_tokens") != STANDARD_MAX_TOKENS:
        raise ValueError("standard max_tokens must be 500")
    if defaults.get("temperature") != 0.0:
        raise ValueError("temperature must be 0.0")
    if defaults.get("seed") != 42:
        raise ValueError("seed must be 42")
    if defaults.get("top_p") is not None:
        raise ValueError("top_p must be unset/null")
    if defaults.get("stop") is not None:
        raise ValueError("stop must be unset/null")

    model = cfg.get("model")
    if not isinstance(model, dict):
        raise ValueError("prepared config requires a model mapping")
    for key in ("name", "provider", "backend"):
        if not model.get(key):
            raise ValueError(f"model.{key} is required")
    if model["backend"] not in {"api", "local"}:
        raise ValueError("model.backend must be 'api' or 'local'")

    jobs = cfg.get("jobs")
    if not isinstance(jobs, list):
        raise ValueError("prepared config requires a jobs list")
    for job in jobs:
        if not isinstance(job, dict):
            raise ValueError("each job must be a mapping")
        condition = job.get("condition")
        if condition not in {"baseline", "two_prompt", "two_prompt_cot"}:
            raise ValueError(f"unsupported condition: {condition!r}")
        if job.get("stage2_max_tokens", STANDARD_MAX_TOKENS) != STANDARD_MAX_TOKENS:
            if condition == "two_prompt_cot":
                raise ValueError("CoT Stage 2 max_tokens must be 500")
            raise ValueError("standard max_tokens must be 500")
        stage1 = job.get("stage1")
        if condition == "two_prompt_cot":
            if not isinstance(stage1, dict) or stage1.get("mode") != "generate":
                raise ValueError("CoT requires generated Stage 1")
            if stage1.get("max_tokens") not in COT_STAGE1_ALLOWED_MAX_TOKENS:
                raise ValueError("CoT Stage 1 max_tokens must be 1500 or 4000")
            subset = job.get("subset")
            if not isinstance(subset, dict):
                raise ValueError("CoT requires a pinned subset mapping")
            required_subset_fields = {
                "path",
                "sha256",
                "ids_newline_sha256",
                "expected_count",
                "expected_subject_count",
                "expected_per_subject",
                "selection_seed",
            }
            missing_subset_fields = required_subset_fields - set(subset)
            if missing_subset_fields:
                raise ValueError(
                    f"CoT subset is missing fields: {sorted(missing_subset_fields)}"
                )
        elif isinstance(stage1, dict) and stage1.get("max_tokens", STANDARD_MAX_TOKENS) != STANDARD_MAX_TOKENS:
            raise ValueError("4000 max_tokens is allowed only for CoT Stage 1")

    cfg["_config_path"] = str(config_path)
    cfg["_config_dir"] = str(config_path.parent)
    return cfg


def build_option_matching_prompt(
    template: str,
    question: FrozenQuestion,
    free_text: str,
    displayed_options: dict[str, str],
) -> str:
    """Render the exact configured Stage-2 template.

    Positional slots use ``dict.get(..., "")`` deliberately. With the
    historical cloud template, a genuine three-option ARC question therefore
    retains the literal blank ``D. `` line and the words ``four options``.
    """
    return template.format(
        question=question.question_text,
        free_text=free_text,
        option_a=displayed_options.get("A", ""),
        option_b=displayed_options.get("B", ""),
        option_c=displayed_options.get("C", ""),
        option_d=displayed_options.get("D", ""),
        options="\n".join(
            f"{label}. {text}" for label, text in displayed_options.items()
        ),
    )


def load_questions_from_csv(
    path: str | Path,
    expected_sha256: str,
    *,
    subset_ids_path: str | Path | None = None,
) -> list[FrozenQuestion]:
    """Load frozen question identity rows with SHA and subset validation."""
    source_path = Path(path)
    actual_sha256 = _sha256_file(source_path)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"question source sha256 mismatch: expected={expected_sha256} "
            f"actual={actual_sha256} path={source_path}"
        )

    questions_by_id: dict[str, FrozenQuestion] = {}
    source_order: list[str] = []
    with source_path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            question_id = row["question_id"]
            if question_id in questions_by_id:
                raise ValueError(
                    f"duplicate question_id {question_id!r} in {source_path}"
                )
            questions_by_id[question_id] = FrozenQuestion(
                question_id=question_id,
                subject=row["subject"],
                question_text=row["question_text"],
                choice_a=row["choice_a"],
                choice_b=row["choice_b"],
                choice_c=row["choice_c"],
                choice_d=row.get("choice_d", ""),
                correct_option=row["correct_option"],
            )
            source_order.append(question_id)

    if subset_ids_path is None:
        ordered_ids = source_order
    else:
        subset_path = Path(subset_ids_path)
        ordered_ids = json.loads(subset_path.read_text(encoding="utf-8"))
        if not isinstance(ordered_ids, list) or not all(
            isinstance(item, str) for item in ordered_ids
        ):
            raise ValueError("subset ID file must contain a JSON string list")
        if len(ordered_ids) != len(set(ordered_ids)):
            raise ValueError("subset ID file contains duplicate IDs")
        unknown = [item for item in ordered_ids if item not in questions_by_id]
        if unknown:
            raise ValueError(f"subset contains unknown IDs: {unknown[:5]}")

    return [questions_by_id[question_id] for question_id in ordered_ids]


def load_stage1_outputs(
    path: str | Path,
    expected_sha256: str,
    question_ids: list[str],
    *,
    response_column: str = "free_text_response",
) -> dict[str, str]:
    """Load and prove complete the fixed Stage-1 answers for one job."""
    source_path = Path(path)
    actual_sha256 = _sha256_file(source_path)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"Stage-1 source sha256 mismatch: expected={expected_sha256} "
            f"actual={actual_sha256} path={source_path}"
        )

    answers: dict[str, str] = {}
    with source_path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            question_id = row["question_id"]
            answer = row.get(response_column)
            if isinstance(answer, str) and answer.strip():
                answers[question_id] = answer

    missing = [question_id for question_id in question_ids if question_id not in answers]
    if missing:
        raise ValueError(
            "missing complete Stage-1 outputs for "
            f"{len(missing)} question(s): {missing[:5]}"
        )
    return {question_id: answers[question_id] for question_id in question_ids}


class LegacyResponseCache:
    """Read successful responses from the historical twoprompt cache format."""

    def __init__(self, cache_dir: str | Path) -> None:
        self._cache_dir = Path(cache_dir)
        if not self._cache_dir.is_dir():
            raise ValueError(
                f"legacy response cache directory does not exist: {self._cache_dir}"
            )

    @staticmethod
    def key(request: ModelRequest) -> str:
        key_data = {
            "provider": request.provider,
            "model_name": request.model_name,
            "prompt": request.payload,
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "seed": request.seed,
            "request_logprobs": request.request_logprobs,
        }
        return hashlib.sha256(
            json.dumps(key_data, sort_keys=True).encode()
        ).hexdigest()

    def get(self, request: ModelRequest) -> ModelResponse | None:
        key = self.key(request)
        path = self._cache_dir / key[:2] / f"{key}.json"
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        raw_text = payload.get("raw_text")
        if not isinstance(raw_text, str) or not raw_text.strip():
            return None

        usage = None
        usage_payload = payload.get("usage")
        if isinstance(usage_payload, dict):
            usage = UsageInfo(
                prompt_tokens=int(usage_payload.get("prompt_tokens", 0)),
                completion_tokens=int(usage_payload.get("completion_tokens", 0)),
                total_tokens=int(usage_payload.get("total_tokens", 0)),
            )
        return ModelResponse(
            provider=request.provider,
            model_name=request.model_name,
            status="success",
            latency_seconds=0.0,
            raw_text=raw_text,
            finish_reason=payload.get("finish_reason"),
            usage=usage,
            logprobs=payload.get("logprobs"),
        )


class CsvResponseReuse:
    """Read successful responses keyed by exact question ID and prompt."""

    def __init__(
        self,
        path: str | Path,
        expected_sha256: str,
        *,
        provider: str,
        model_name: str,
    ) -> None:
        source_path = Path(path)
        actual_sha256 = _sha256_file(source_path)
        if actual_sha256 != expected_sha256:
            raise ValueError(
                f"reuse source sha256 mismatch: expected={expected_sha256} "
                f"actual={actual_sha256} path={source_path}"
            )
        self._provider = provider
        self._model_name = model_name
        self._responses: dict[tuple[str, str], ModelResponse] = {}
        with source_path.open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                if row.get("model_status", "success") != "success":
                    continue
                raw_text = row.get("raw_text", "")
                if not raw_text.strip():
                    continue
                key = (row["question_id"], row["prompt"])
                if key in self._responses:
                    raise ValueError(
                        f"duplicate reusable response for {key!r} in {source_path}"
                    )
                self._responses[key] = ModelResponse(
                    provider=provider,
                    model_name=model_name,
                    status="success",
                    latency_seconds=0.0,
                    raw_text=raw_text,
                    finish_reason=row.get("finish_reason") or None,
                )

    def get(self, question_id: str, prompt: str) -> ModelResponse | None:
        return self._responses.get((question_id, prompt))


async def fill_missing_responses(
    requests: list[ModelRequest],
    recovered: list[ModelResponse | None],
    generate_batch: Callable[
        [list[ModelRequest]], Awaitable[list[ModelResponse]]
    ],
) -> tuple[list[ModelResponse], list[bool]]:
    """Generate only unresolved requests while retaining original order."""
    if len(requests) != len(recovered):
        raise ValueError("requests and recovered responses must have equal length")

    missing_indices = [
        index for index, response in enumerate(recovered) if response is None
    ]
    generated = await generate_batch([requests[index] for index in missing_indices])
    if len(generated) != len(missing_indices):
        raise ValueError("generate_batch returned an unexpected response count")

    complete = list(recovered)
    for index, response in zip(missing_indices, generated, strict=True):
        complete[index] = response
    return (
        [response for response in complete if response is not None],
        [response is not None for response in recovered],
    )


def _historical_option_map(question: FrozenQuestion) -> dict[str, str]:
    """Keep frozen option bytes while dropping only genuinely absent choices."""
    options: dict[str, str] = {}
    for label, value in (
        ("A", question.choice_a),
        ("B", question.choice_b),
        ("C", question.choice_c),
        ("D", question.choice_d),
    ):
        if value is None:
            continue
        text = str(value)
        if not text.strip() or text.strip().lower() == "nan":
            continue
        options[label] = text
    if question.correct_option not in options:
        raise ValueError(
            f"{question.question_id}: correct option {question.correct_option!r} "
            f"is absent from {list(options)}"
        )
    return options


def _validate_subset(
    spec: dict[str, Any],
    cfg: dict[str, Any],
    questions: list[FrozenQuestion],
) -> None:
    subset_path = _job_path(spec["path"], cfg)
    actual_file_sha = _sha256_file(subset_path)
    if actual_file_sha != spec["sha256"]:
        raise ValueError(
            f"subset file sha256 mismatch: expected={spec['sha256']} "
            f"actual={actual_file_sha}"
        )
    ordered_ids = json.loads(subset_path.read_text(encoding="utf-8"))
    ids_digest = hashlib.sha256(
        ("\n".join(ordered_ids) + "\n").encode("utf-8")
    ).hexdigest()
    if ids_digest != spec["ids_newline_sha256"]:
        raise ValueError(
            "subset canonical ID digest mismatch: "
            f"expected={spec['ids_newline_sha256']} actual={ids_digest}"
        )
    if len(questions) != spec["expected_count"]:
        raise ValueError(
            f"subset expected {spec['expected_count']} rows, found {len(questions)}"
        )
    subject_counts: dict[str, int] = {}
    for question in questions:
        subject_counts[question.subject] = subject_counts.get(question.subject, 0) + 1
    if len(subject_counts) != spec["expected_subject_count"]:
        raise ValueError(
            "subset subject count mismatch: "
            f"expected={spec['expected_subject_count']} actual={len(subject_counts)}"
        )
    wrong_counts = {
        subject: count
        for subject, count in subject_counts.items()
        if count != spec["expected_per_subject"]
    }
    if wrong_counts:
        raise ValueError(f"subset is not evenly stratified: {wrong_counts}")


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=TRACE_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _make_request(
    cfg: dict[str, Any],
    prompt: str,
    *,
    max_tokens: int,
) -> ModelRequest:
    defaults = cfg["defaults"]
    model = cfg["model"]
    return ModelRequest(
        provider=model["provider"],
        model_name=model["name"],
        payload=prompt,
        temperature=defaults["temperature"],
        max_tokens=max_tokens,
        seed=defaults["seed"],
        request_logprobs=False,
    )


def _build_api_client(model: dict[str, Any]) -> Any:
    """Construct the selected provider client only when execution begins."""
    common = {
        "model_name": model["name"],
        "timeout": model["timeout_seconds"],
        "concurrency_limit": model["concurrency"],
        "max_retries": model["max_retries"],
        "min_delay_seconds": model["min_delay_seconds"],
    }
    if model["provider"] == "gemini":
        from choicebench.clients.gemini_client import GeminiClient

        return GeminiClient(api_key=os.environ.get("GEMINI_API_KEY"), **common)
    if model["provider"] == "groq":
        from choicebench.clients.groq_client import GroqClient

        return GroqClient(api_key=os.environ.get("GROQ_API_KEY"), **common)
    if model["provider"] == "together":
        from choicebench.clients.together_client import TogetherAIClient

        return TogetherAIClient(
            api_key=os.environ.get("TOGETHER_API_KEY"),
            **common,
        )
    raise ValueError(f"unsupported API provider: {model['provider']!r}")


def _load_local_backend(model: dict[str, Any]) -> Any:
    """Load the historical raw-prompt Hugging Face backend on demand."""
    from choicebench.backends.hf_backend import HuggingFaceBackend

    backend = HuggingFaceBackend(
        model["name"],
        device=model.get("device", "cuda"),
        add_bos_token=model.get("add_bos_token", True),
        revision=model.get("revision"),
    )
    backend.load()
    return backend


def _job_path(
    value: str | Path,
    cfg: dict[str, Any],
) -> Path:
    return _expand_path(value, base_dir=Path(cfg["_config_dir"]))


def _csv_reuse(
    spec: dict[str, Any] | None,
    cfg: dict[str, Any],
) -> CsvResponseReuse | None:
    if spec is None:
        return None
    return CsvResponseReuse(
        _job_path(spec["path"], cfg),
        spec["sha256"],
        provider=cfg["model"]["provider"],
        model_name=cfg["model"]["name"],
    )


def _recover_response(
    question_id: str,
    request: ModelRequest,
    *,
    csv_reuse: CsvResponseReuse | None,
    legacy_cache: LegacyResponseCache | None,
) -> ModelResponse | None:
    response = (
        csv_reuse.get(question_id, request.payload)
        if csv_reuse is not None
        else None
    )
    if response is None and legacy_cache is not None:
        response = legacy_cache.get(request)
    return response


async def _generate_local_batch(
    backend: Any,
    requests: list[ModelRequest],
) -> list[ModelResponse]:
    responses: list[ModelResponse] = []
    for request in requests:
        started = time.monotonic()
        try:
            raw_text = backend.generate(
                request.payload,
                max_new_tokens=request.max_tokens,
                temperature=request.temperature,
                do_sample=False,
                seed=request.seed,
            )
            responses.append(
                ModelResponse(
                    provider=request.provider,
                    model_name=request.model_name,
                    status="success",
                    latency_seconds=time.monotonic() - started,
                    raw_text=raw_text,
                )
            )
        except Exception as exc:  # noqa: BLE001
            responses.append(
                ModelResponse(
                    provider=request.provider,
                    model_name=request.model_name,
                    status="failure",
                    latency_seconds=time.monotonic() - started,
                    error=ErrorInfo(
                        type(exc).__name__,
                        str(exc) or "Unexpected exception with no message.",
                        False,
                        "backend_generate",
                    ),
                )
            )
    return responses


def _client_for_job(
    raw_client: Any,
    cfg: dict[str, Any],
    job: dict[str, Any],
) -> Any:
    cache = job.get("cache", {})
    if not cache.get("enabled", False):
        return raw_client
    from choicebench.infra.cache import CachingClientWrapper, ResponseCache

    cache_root = _job_path(cfg["paths"]["cache_root"], cfg)
    return CachingClientWrapper(
        raw_client,
        ResponseCache(
            cache_root / job["run_id"],
            namespace=cache["namespace"],
        ),
    )


async def _run_job(
    cfg: dict[str, Any],
    job: dict[str, Any],
    *,
    raw_api_client: Any | None,
    local_backend: Any | None,
) -> Path:
    model = cfg["model"]
    question_spec = job["question_source"]
    subset = job.get("subset")
    subset_path = _job_path(subset["path"], cfg) if subset else None
    questions = load_questions_from_csv(
        _job_path(question_spec["path"], cfg),
        question_spec["sha256"],
        subset_ids_path=subset_path,
    )
    if subset is not None:
        _validate_subset(subset, cfg, questions)
    question_ids = [question.question_id for question in questions]

    stage1_answers: dict[str, str] | None = None
    if job["condition"] == "two_prompt":
        stage1 = job["stage1"]
        if stage1["mode"] != "reuse":
            raise ValueError("standard two_prompt Stage 1 must use fixed outputs")
        stage1_answers = load_stage1_outputs(
            _job_path(stage1["path"], cfg),
            stage1["sha256"],
            question_ids,
            response_column=stage1.get("response_column", "free_text_response"),
        )

    csv_reuse = _csv_reuse(job.get("reuse", {}).get("csv_permutation0"), cfg)
    legacy_cache = None
    if job.get("reuse", {}).get("legacy_cache", False):
        legacy_cache = LegacyResponseCache(
            _job_path(cfg["paths"]["legacy_cache_root"], cfg)
        )

    if model["backend"] == "api":
        generate_batch = _client_for_job(raw_api_client, cfg, job).generate_batch
    else:
        async def generate_batch(requests: list[ModelRequest]) -> list[ModelResponse]:
            return await _generate_local_batch(local_backend, requests)

    output_root = _job_path(cfg["paths"]["output_root"], cfg)
    output_path = (
        output_root
        / job["run_id"]
        / f"{job['run_id']}_{job['cell_id']}_traced.csv"
    )
    checkpoint_every = int(job["checkpoint_every_questions"])
    checkpoint = None
    state = None
    if checkpoint_every > 0:
        checkpoint = CheckpointManager(
            checkpoint_dir=_job_path(cfg["paths"]["checkpoint_root"], cfg),
            run_id=job["run_id"],
            condition=job["condition"],
            model=model["name"],
            benchmark=job["benchmark"],
        )
        state = checkpoint.load()

    started_at = datetime.now(timezone.utc).isoformat()
    completed_ids: list[str] = []
    all_rows: list[dict[str, Any]] = []
    if state is not None:
        completed_ids = list(state["completed_ids"])
        all_rows = list(state["results"])
        started_at = state["started_at"]
    completed = set(completed_ids)

    assembler = TraceRetainingCyclicRunner(
        backend=None,
        model_name=model["name"],
        provider=model["provider"],
        benchmark=job["benchmark"],
        split_name=job.get("split_name", "robustness"),
        cell_id=job["cell_id"],
        run_id=job["run_id"],
    )
    option_template = cfg["prompts"]["option_matching"]
    cot_prompt_template = cfg["prompts"].get("cot_stage1")

    remaining = [question for question in questions if question.question_id not in completed]
    print(
        f"[prepared] {job['run_id']}: {len(completed)}/{len(questions)} "
        "questions already complete",
        flush=True,
    )
    for index, question in enumerate(remaining):
        canonical_options = _historical_option_map(question)
        permutations = generate_permutations(canonical_options)

        if job["condition"] == "baseline":
            prompts = [
                (
                    build_local_prompt(question.question_text, permutation)
                    if model["provider"] == "huggingface"
                    else build_api_prompt(question.question_text, permutation)
                )
                for permutation in permutations
            ]
        else:
            if job["condition"] == "two_prompt":
                free_text = stage1_answers[question.question_id]
            else:
                if not cot_prompt_template:
                    raise ValueError("CoT Stage-1 prompt is missing")
                stage1_prompt = cot_prompt_template.format(
                    question=question.question_text
                )
                stage1_request = _make_request(
                    cfg,
                    stage1_prompt,
                    max_tokens=job["stage1"]["max_tokens"],
                )
                stage1_response = (await generate_batch([stage1_request]))[0]
                if not stage1_response.is_success():
                    error = stage1_response.error
                    raise RuntimeError(
                        "CoT Stage 1 failed for "
                        f"{question.question_id}: "
                        f"{error.error_type if error else 'unknown'} "
                        f"{error.message if error else ''}"
                    )
                free_text = stage1_response.raw_text
            prompts = [
                build_option_matching_prompt(
                    option_template,
                    question,
                    free_text,
                    permutation,
                )
                for permutation in permutations
            ]

        requests = [
            _make_request(
                cfg,
                prompt,
                max_tokens=job.get(
                    "stage2_max_tokens",
                    cfg["defaults"]["max_tokens"],
                ),
            )
            for prompt in prompts
        ]
        recovered = [
            _recover_response(
                question.question_id,
                request,
                csv_reuse=csv_reuse,
                legacy_cache=legacy_cache,
            )
            for request in requests
        ]
        responses, reused = await fill_missing_responses(
            requests,
            recovered,
            generate_batch,
        )
        latencies = [response.latency_seconds for response in responses]
        cache_hits = [
            was_reused or response.latency_seconds == 0.0
            for was_reused, response in zip(reused, responses, strict=True)
        ]
        rows = assembler._assemble_rows(
            question,
            index,
            permutations,
            prompts,
            responses,
            latencies,
            cache_hits,
            False,
            canonical_options,
        )
        validate_question_traces(question.question_id, rows)
        all_rows.extend(rows)
        completed_ids.append(question.question_id)
        completed.add(question.question_id)

        done = len(completed)
        if checkpoint is not None and (
            done % checkpoint_every == 0 or done == len(questions)
        ):
            checkpoint.save(completed_ids, all_rows, started_at)
            _write_csv(all_rows, output_path)
            print(
                f"[prepared] {job['run_id']}: {done}/{len(questions)} complete",
                flush=True,
            )

    _write_csv(all_rows, output_path)
    if checkpoint is not None and len(completed) == len(questions):
        checkpoint.delete()
    print(f"[prepared] DONE: {output_path}", flush=True)
    return output_path


async def run_matrix(cfg: dict[str, Any]) -> list[Path]:
    """Run every job in one validated prepared matrix configuration."""
    raw_api_client = None
    local_backend = None
    if cfg["model"]["backend"] == "api":
        raw_api_client = _build_api_client(cfg["model"])
    else:
        local_backend = _load_local_backend(cfg["model"])

    outputs = []
    for job in cfg["jobs"]:
        outputs.append(
            await _run_job(
                cfg,
                job,
                raw_api_client=raw_api_client,
                local_backend=local_backend,
            )
        )
    return outputs


def run_config(path: str | Path) -> list[Path]:
    """Synchronous CLI entry point; this is the inference boundary."""
    return asyncio.run(run_matrix(load_matrix_config(path)))
