# experiments/flip_rate_traces/runner.py
#
# Trace-retaining cyclic-permutation runner. Reuses choicebench's prompt
# building, option mapping, parsing, and scoring infrastructure verbatim;
# changes exactly one thing versus the historical protocol (see
# historical_protocol.py docstring for the full diff): every permutation's
# full trace is captured and returned, not just permutation 0's.
#
# Dispatch mirrors choicebench's own ExperimentRunner convention (see
# ChoiceBench v0.2.0 Audit.md Finding 1): run_many_async()/generate_batch()
# for APIBackend, run_one()/generate() synchronously for HuggingFaceBackend.
# APIBackend.generate() must never be called directly (it raises by design).

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any

from choicebench.clients.types import ErrorInfo, ModelResponse
from choicebench.identity import redact_text
from choicebench.parsing.parser import parse_model_answer
from choicebench.parsing.types import ParseResult
from choicebench.pipeline.options import build_option_map
from choicebench.pipeline.prompt_builder import (
    build_direct_mcq_prompt,
    load_prompt_templates,
)
from choicebench.scoring.scorer import score_prediction

from experiments.flip_rate_traces.data_source import FrozenQuestion
from experiments.flip_rate_traces.historical_protocol import (
    HISTORICAL_MAX_TOKENS,
    HISTORICAL_PROMPT_VERSION,
    HISTORICAL_SEED,
    HISTORICAL_TEMPERATURE,
    generate_permutations,
    historical_majority_vote,
    unpermute_choice,
)
from experiments.flip_rate_traces.trace_schema import build_trace_row


def _question_row(q: FrozenQuestion) -> dict[str, Any]:
    return {
        "question_id": q.question_id,
        "subject": q.subject,
        "question_text": q.question_text,
        "choice_a": q.choice_a,
        "choice_b": q.choice_b,
        "choice_c": q.choice_c,
        "choice_d": q.choice_d,
        "correct_option": q.correct_option,
    }


class TraceRetainingCyclicRunner:
    """Executes the cyclic-generation-majority protocol for one question,
    returning one trace row per permutation instead of discarding N-1 of them.
    """

    def __init__(
        self,
        *,
        backend: Any,
        model_name: str,
        provider: str,
        benchmark: str,
        split_name: str,
        cell_id: str,
        run_id: str,
    ) -> None:
        self._backend = backend
        self._model_name = model_name
        self._provider = provider
        self._benchmark = benchmark
        self._split_name = split_name
        self._cell_id = cell_id
        self._run_id = run_id
        self._prompts = load_prompt_templates(HISTORICAL_PROMPT_VERSION)

    def _build_prompt(self, question_text: str, options: dict[str, str]) -> str:
        return build_direct_mcq_prompt(
            template=self._prompts["direct_mcq"],
            question=question_text,
            options=options,
        )

    def _assemble_rows(
        self,
        q: FrozenQuestion,
        sample_index: int,
        permutations: list[dict[str, str]],
        prompts: list[str],
        per_perm_response: list[ModelResponse],
        per_perm_latency: list[float],
        per_perm_cache_hit: list[bool | None],
        is_diagnostic_canary: bool,
    ) -> list[dict]:
        canonical_options = build_option_map(_question_row(q))
        n = len(permutations)

        displayed_parsed: list[str | None] = []
        semantic_parsed: list[str | None] = []
        parse_results = []
        for perm, response in zip(permutations, per_perm_response):
            if response.is_success():
                parsed = parse_model_answer(response.raw_text, perm)
                sem = unpermute_choice(parsed.final_choice, perm, canonical_options) \
                    if parsed.final_choice is not None else None
            else:
                parsed = None
                sem = None
            parse_results.append(parsed)
            displayed_parsed.append(parsed.final_choice if parsed else None)
            semantic_parsed.append(sem)

        voted_letter = historical_majority_vote(semantic_parsed)
        cleaned = [x for x in semantic_parsed if x is not None]
        is_tie = False
        if cleaned:
            import collections
            counts = collections.Counter(cleaned)
            top = counts.most_common(2)
            is_tie = len(top) > 1 and top[0][1] == top[1][1]
        majority_is_correct = (voted_letter == q.correct_option) if voted_letter else False

        rows = []
        for idx, (perm, response, parsed, sem, latency, cache_hit) in enumerate(zip(
            permutations, per_perm_response, parse_results, semantic_parsed,
            per_perm_latency, per_perm_cache_hit,
        )):
            if parsed is not None:
                # score_prediction must be scored in SEMANTIC (canonical) space,
                # not displayed space -- parsed.final_choice is the letter as
                # shown for this permutation, which only equals the canonical
                # letter for permutation 0. Build a semantic-space ParseResult
                # (same status/reason, final_choice replaced by sem) so
                # score_status and is_correct always agree.
                semantic_parse_result = ParseResult(
                    final_choice=sem,
                    status=parsed.status,
                    raw_text=parsed.raw_text,
                    normalized_text=parsed.normalized_text,
                    reason=parsed.reason,
                )
                score = score_prediction(semantic_parse_result, q.correct_option)
                is_correct = score.is_correct if score.is_correct is not None else False
                parse_status = parsed.status
                parse_reason = parsed.reason
                normalized_text = parsed.normalized_text
            else:
                score = None
                is_correct = False
                parse_status = None
                parse_reason = None
                normalized_text = None

            err = response.error if not response.is_success() else None
            row = build_trace_row(
                schema_version="flip_rate_traces.trace.v1",
                run_id=self._run_id,
                cell_id=self._cell_id,
                question_id=q.question_id,
                benchmark=self._benchmark,
                split_name=self._split_name,
                subject=q.subject,
                model_name=self._model_name,
                provider=self._provider,
                n_options=n,
                permutation_index=idx,
                n_permutations=n,
                canonical_options_json=_json(canonical_options),
                displayed_options_json=_json(perm),
                displayed_to_semantic_json=_json({
                    letter: unpermute_choice(letter, perm, canonical_options)
                    for letter in perm
                }),
                correct_option=q.correct_option,
                prompt=prompts[idx],
                prompt_sha256=_sha256_text(prompts[idx]),
                temperature=HISTORICAL_TEMPERATURE,
                max_tokens=HISTORICAL_MAX_TOKENS,
                seed=HISTORICAL_SEED,
                prompt_version=HISTORICAL_PROMPT_VERSION,
                raw_text=response.raw_text,
                finish_reason=response.finish_reason,
                displayed_parsed_choice=parsed.final_choice if parsed else None,
                semantic_parsed_choice=sem,
                parse_status=parse_status,
                parse_reason=parse_reason,
                normalized_text=normalized_text,
                is_correct=is_correct,
                score_status=(score.status if parsed is not None and score else None),
                transport_status=response.status,
                error_type=(err.error_type if err else None),
                error_message=(err.message if err else None),
                error_stage=(err.stage if err else None),
                error_retryable=(err.retryable if err else None),
                latency_seconds=latency,
                timestamp_utc=datetime.now(timezone.utc).isoformat(),
                cache_hit=cache_hit,
                majority_semantic_choice=voted_letter,
                majority_is_tie=is_tie,
                majority_tie_break_used=(is_tie and voted_letter is not None),
                majority_is_correct=majority_is_correct,
                is_diagnostic_canary=is_diagnostic_canary,
            )
            rows.append(row)
        return rows

    def run_one_sync(
        self, q: FrozenQuestion, sample_index: int, is_diagnostic_canary: bool = False,
    ) -> list[dict]:
        """Synchronous path for local (HuggingFace) backends.

        Never converts a raised exception into a success row: on failure,
        raw_text stays None, transport_status becomes "failure", and
        error_type/message/stage are populated from the real exception.
        """
        canonical_options = build_option_map(_question_row(q))
        permutations = generate_permutations(canonical_options)
        prompts = [self._build_prompt(q.question_text, perm) for perm in permutations]

        responses: list[ModelResponse] = []
        latencies: list[float] = []
        cache_hits: list[bool | None] = []
        for prompt in prompts:
            t0 = time.monotonic()
            try:
                raw_text = self._backend.generate(
                    prompt,
                    max_new_tokens=HISTORICAL_MAX_TOKENS,
                    temperature=HISTORICAL_TEMPERATURE,
                    seed=HISTORICAL_SEED,
                )
                latency = time.monotonic() - t0
                responses.append(ModelResponse(
                    provider=self._provider,
                    model_name=self._model_name,
                    status="success",
                    latency_seconds=latency,
                    raw_text=raw_text,
                ))
            except Exception as exc:
                latency = time.monotonic() - t0
                responses.append(ModelResponse(
                    provider=self._provider,
                    model_name=self._model_name,
                    status="failure",
                    latency_seconds=latency,
                    error=ErrorInfo(type(exc).__name__, redact_text(exc), False, "backend_generate"),
                ))
            latencies.append(latency)
            cache_hits.append(None)  # local backend has no cache layer

        return self._assemble_rows(
            q, sample_index, permutations, prompts, responses, latencies, cache_hits,
            is_diagnostic_canary,
        )

    async def run_one_async(
        self, q: FrozenQuestion, sample_index: int, is_diagnostic_canary: bool = False,
    ) -> list[dict]:
        """Async path for API backends. Never calls backend.generate() —
        only backend.generate_batch(), per APIBackend's own contract."""
        canonical_options = build_option_map(_question_row(q))
        permutations = generate_permutations(canonical_options)
        prompts = [self._build_prompt(q.question_text, perm) for perm in permutations]

        t0 = time.monotonic()
        responses = await self._backend.generate_batch(prompts)
        elapsed = time.monotonic() - t0
        latencies = [
            r.latency_seconds if r.latency_seconds else elapsed / len(prompts)
            for r in responses
        ]
        cache_hits = [r.latency_seconds == 0.0 for r in responses]

        return self._assemble_rows(
            q, sample_index, permutations, prompts, responses, latencies, cache_hits,
            is_diagnostic_canary,
        )


def _json(d: dict) -> str:
    import json
    return json.dumps(d, sort_keys=True, ensure_ascii=False)


def _sha256_text(text: str) -> str:
    import hashlib
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
