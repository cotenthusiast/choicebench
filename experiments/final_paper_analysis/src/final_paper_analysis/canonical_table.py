"""Canonical row-level analysis table (spec section 3).

One row per question x model x benchmark x method/condition. Deliberately
decoupled from choicebench.importing's own narrow published-result schema:
Component 2's verified import runs only publish a 6-field row (question_id,
question_text, correct_option, choices_json, prediction_origin,
predicted_option) by design (engine.py's reduced scope), which loses the
richer per-method columns (raw free text, parse status, fallback flags,
text-match score) that live only in the original source CSVs. So this module
takes a ``CanonicalRowSource`` per row -- assembled by the final-boss session
from BOTH a verified import run's trusted identity metadata (model/method/
benchmark, prediction_origin, derivation_origin, evidence_status,
scope_disposition -- all hash-verified) AND the same hash-verified source
CSV's richer raw columns -- and derives every standardized canonical field
deterministically from that combination. Never collapses local/API model
variants or two-stage versions into one row: every CanonicalRowSource keeps
its own model_key/backend/method_key/realization_id.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import pandas as pd

_PARSE_FAILURE_STATUSES = frozenset({"parse_missing", "parse_failed", "score_unscorable"})

CANONICAL_COLUMNS = (
    "question_id", "model_key", "backend", "benchmark_name", "method_key",
    "condition_id", "realization_id", "option_count", "gold_answer",
    "displayed_gold_position", "predicted_option", "is_correct",
    "raw_prediction", "parse_status", "parse_failed", "fallback_used",
    "has_prediction", "prediction_origin", "derivation_origin",
    "evidence_status", "scope_disposition", "status",
)

_STATUS_BY_EVIDENCE = {
    "complete": "included",
    "qualified": "qualified",
    "partial": "incomplete",
    "malformed": "incomplete",
    "recoverable": "incomplete",
    "failed": "incomplete",
}


class CanonicalTableError(ValueError):
    """Raised when row-level canonical-table construction is given
    inconsistent or duplicate input."""


@dataclass(frozen=True)
class CanonicalRowSource:
    """One row's full raw content plus its verified identity metadata.
    ``choices_json`` may be a JSON string or an already-parsed mapping of
    option letter to option text (including empty-string entries for a
    structurally absent option, e.g. the 3 genuine three-option ARC
    questions' choice_d)."""

    question_id: str
    model_key: str
    backend: str  # "api" | "local"
    benchmark_name: str
    method_key: str
    condition_id: str
    realization_id: str
    prediction_origin: str
    derivation_origin: str
    evidence_status: str
    scope_disposition: str
    correct_option: str
    predicted_option: str | None
    choices_json: str | Mapping[str, str]
    raw_prediction: str | None = None
    parse_status: str | None = None
    fallback_used: bool | None = None


def _parsed_choices(choices_json: str | Mapping[str, str]) -> Mapping[str, str]:
    if isinstance(choices_json, Mapping):
        return choices_json
    if isinstance(choices_json, str) and choices_json.strip():
        return json.loads(choices_json)
    raise CanonicalTableError(f"Cannot parse choices_json: {choices_json!r}.")


def _option_count(choices_json: str | Mapping[str, str]) -> int:
    parsed = _parsed_choices(choices_json)
    return sum(1 for text in parsed.values() if isinstance(text, str) and text.strip())


def _row_status(*, evidence_status: str, scope_disposition: str) -> str:
    if scope_disposition == "excluded_from_paper_matrix":
        return "excluded"
    if scope_disposition == "held":
        return "incomplete"
    if evidence_status not in _STATUS_BY_EVIDENCE:
        raise CanonicalTableError(f"Unknown evidence_status {evidence_status!r}.")
    return _STATUS_BY_EVIDENCE[evidence_status]


def build_canonical_table(sources: Sequence[CanonicalRowSource]) -> pd.DataFrame:
    """Build the canonical row-level table. Fails closed on a duplicate
    (question_id, model_key, backend, benchmark_name, method_key,
    condition_id) key -- exactly the identity that must produce one row,
    never silently collapsed or silently duplicated."""
    seen_keys: set[tuple[str, ...]] = set()
    rows: list[dict[str, Any]] = []
    for source in sources:
        key = (
            source.question_id, source.model_key, source.backend,
            source.benchmark_name, source.method_key, source.condition_id,
        )
        if key in seen_keys:
            raise CanonicalTableError(f"Duplicate canonical row key: {key}.")
        seen_keys.add(key)

        has_prediction = source.predicted_option is not None
        parse_failed = (
            not has_prediction
            or (source.parse_status is not None and source.parse_status in _PARSE_FAILURE_STATUSES)
        )
        is_correct: bool | None = None
        if has_prediction and not parse_failed:
            is_correct = source.predicted_option == source.correct_option

        rows.append({
            "question_id": source.question_id,
            "model_key": source.model_key,
            "backend": source.backend,
            "benchmark_name": source.benchmark_name,
            "method_key": source.method_key,
            "condition_id": source.condition_id,
            "realization_id": source.realization_id,
            "option_count": _option_count(source.choices_json),
            "gold_answer": source.correct_option,
            "displayed_gold_position": source.correct_option,
            "predicted_option": source.predicted_option,
            "is_correct": is_correct,
            "raw_prediction": source.raw_prediction,
            "parse_status": source.parse_status,
            "parse_failed": parse_failed,
            "fallback_used": source.fallback_used,
            "has_prediction": has_prediction,
            "prediction_origin": source.prediction_origin,
            "derivation_origin": source.derivation_origin,
            "evidence_status": source.evidence_status,
            "scope_disposition": source.scope_disposition,
            "status": _row_status(
                evidence_status=source.evidence_status, scope_disposition=source.scope_disposition
            ),
        })

    frame = pd.DataFrame(rows, columns=CANONICAL_COLUMNS)
    return frame
