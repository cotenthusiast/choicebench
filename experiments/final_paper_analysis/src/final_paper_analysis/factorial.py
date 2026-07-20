"""Full 2x2 factorial decomposition (spec section 6).

The four conditions are named by role, not guessed from method names:
``hidden_embedding``, ``visible_embedding``, ``hidden_llm``, ``visible_llm``
(options-hidden/visible x embedding/LLM matcher). Binding each role to the
exact historical cell/realization with correct provenance -- never averaging
or substituting across two_stage_v1/v2/v3, and verifying the visible+LLM
condition reuses the exact visible Stage-1 response used by visible+embedding
-- is the final-boss import session's job, using Component 2's orchestration
and the handoff's own provenance records. This module only consumes four
already-bound, already-paired correctness columns; it does not look up or
guess which historical cell fills which role.

Not every model/benchmark cell will have all four roles populated in the
real inventory (per the task's instruction to derive availability from the
verified canonical inventory, not assume a complete matrix) -- functions
here fail closed with a clear "missing role" error rather than silently
computing a partial factorial with an absent condition treated as zero.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

_ROLES = ("hidden_embedding", "visible_embedding", "hidden_llm", "visible_llm")


class FactorialError(ValueError):
    """Raised when factorial decomposition input is missing a required role
    or the four conditions are not paired on an identical question set."""


def join_factorial_conditions(cells: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Join the four role frames on question_id into one wide table with
    columns {role}__correct, {role}__coverage_flag, etc. Fails closed if any
    of the four roles is missing, or if the four question_id sets are not
    identical (a real factorial contrast requires the exact same frozen
    questions in all four cells)."""
    missing = [role for role in _ROLES if role not in cells]
    if missing:
        raise FactorialError(f"Missing required factorial role(s): {missing}.")

    id_sets = {role: set(cells[role]["question_id"]) for role in _ROLES}
    reference = id_sets[_ROLES[0]]
    mismatched = {role: ids for role, ids in id_sets.items() if ids != reference}
    if mismatched:
        raise FactorialError(
            "Factorial roles do not share an identical question_id set: "
            f"{sorted({role for role in mismatched})}."
        )

    joined = None
    for role in _ROLES:
        frame = cells[role][["question_id", "is_correct", "has_prediction", "parse_failed", "fallback_used"]].copy()
        frame["is_correct"] = frame["is_correct"].fillna(False).astype(bool)
        frame = frame.rename(columns={
            "is_correct": f"{role}__correct",
            "has_prediction": f"{role}__has_prediction",
            "parse_failed": f"{role}__parse_failed",
            "fallback_used": f"{role}__fallback_used",
        })
        joined = frame if joined is None else joined.merge(frame, on="question_id", how="inner")
    return joined


def _main_effect_visible(correct: Mapping[str, np.ndarray]) -> np.ndarray:
    """Main effect of showing options during Stage 1, averaged across both
    matcher types."""
    return 0.5 * (
        (correct["visible_llm"] - correct["hidden_llm"])
        + (correct["visible_embedding"] - correct["hidden_embedding"])
    )


def _main_effect_llm(correct: Mapping[str, np.ndarray]) -> np.ndarray:
    """Main effect of replacing embedding matching with LLM matching,
    averaged across both options-visibility settings."""
    return 0.5 * (
        (correct["hidden_llm"] - correct["hidden_embedding"])
        + (correct["visible_llm"] - correct["visible_embedding"])
    )


def _interaction_effect(correct: Mapping[str, np.ndarray]) -> np.ndarray:
    """Standard 2x2 interaction: does the effect of showing options depend
    on which matcher is used (equivalently, does the effect of the matcher
    depend on options visibility)."""
    return (
        (correct["visible_llm"] - correct["hidden_llm"])
        - (correct["visible_embedding"] - correct["hidden_embedding"])
    )


@dataclass(frozen=True)
class ContrastEstimate:
    point_estimate: float
    lower: float
    upper: float
    confidence_level: float


@dataclass(frozen=True)
class FactorialEffects:
    n_questions: int
    accuracy_by_role: Mapping[str, float]
    main_effect_visible_options: ContrastEstimate
    main_effect_llm_matcher: ContrastEstimate
    interaction_effect: ContrastEstimate
    seed: int
    n_resamples: int


def compute_factorial_effects(
    joined: pd.DataFrame, *, seed: int, n_resamples: int = 10_000, confidence_level: float = 0.95
) -> FactorialEffects:
    """Compute main effects, interaction, and paired bootstrap CIs for each,
    resampling question indices jointly across all four roles per replicate
    (preserving the paired/correlated structure -- never resampling each
    role independently)."""
    if joined.empty:
        raise FactorialError("Cannot compute factorial effects over zero questions.")
    n = len(joined)
    correct = {role: joined[f"{role}__correct"].to_numpy().astype(float) for role in _ROLES}
    accuracy_by_role = {role: float(values.mean()) for role, values in correct.items()}

    point_visible = float(_main_effect_visible(correct).mean())
    point_llm = float(_main_effect_llm(correct).mean())
    point_interaction = float(_interaction_effect(correct).mean())

    rng = np.random.default_rng(seed)
    replicate_visible = np.empty(n_resamples)
    replicate_llm = np.empty(n_resamples)
    replicate_interaction = np.empty(n_resamples)
    for i in range(n_resamples):
        indices = rng.integers(0, n, size=n)
        resampled = {role: values[indices] for role, values in correct.items()}
        replicate_visible[i] = _main_effect_visible(resampled).mean()
        replicate_llm[i] = _main_effect_llm(resampled).mean()
        replicate_interaction[i] = _interaction_effect(resampled).mean()

    alpha = 1.0 - confidence_level

    def _contrast(point: float, replicates: np.ndarray) -> ContrastEstimate:
        return ContrastEstimate(
            point_estimate=point,
            lower=float(np.percentile(replicates, 100 * (alpha / 2))),
            upper=float(np.percentile(replicates, 100 * (1 - alpha / 2))),
            confidence_level=confidence_level,
        )

    return FactorialEffects(
        n_questions=n,
        accuracy_by_role=accuracy_by_role,
        main_effect_visible_options=_contrast(point_visible, replicate_visible),
        main_effect_llm_matcher=_contrast(point_llm, replicate_llm),
        interaction_effect=_contrast(point_interaction, replicate_interaction),
        seed=seed, n_resamples=n_resamples,
    )


@dataclass(frozen=True)
class FactorialDiagnosticDiffs:
    coverage_by_role: Mapping[str, float]
    parse_failure_rate_by_role: Mapping[str, float]
    fallback_rate_by_role: Mapping[str, float]


def compute_factorial_diagnostic_diffs(joined: pd.DataFrame) -> FactorialDiagnosticDiffs:
    """Accompanying coverage/parse/failure/fallback rates per role (not
    differenced automatically -- callers difference whichever pair they
    need; reported per-role so no information is discarded)."""
    coverage_by_role = {}
    parse_failure_by_role = {}
    fallback_by_role = {}
    for role in _ROLES:
        has_prediction = joined[f"{role}__has_prediction"]
        parse_failed = joined[f"{role}__parse_failed"]
        fallback_used = joined[f"{role}__fallback_used"]
        coverage_by_role[role] = float(has_prediction.mean())
        parse_failure_by_role[role] = float(parse_failed.mean())
        tracked = fallback_used[fallback_used.notna()]
        fallback_by_role[role] = float((tracked == True).mean()) if len(tracked) else float("nan")  # noqa: E712
    return FactorialDiagnosticDiffs(
        coverage_by_role=coverage_by_role,
        parse_failure_rate_by_role=parse_failure_by_role,
        fallback_rate_by_role=fallback_by_role,
    )


def verify_shared_stage1_response(
    visible_llm_responses: Mapping[str, str], visible_embedding_responses: Mapping[str, str]
) -> None:
    """Verify the visible+LLM condition reuses the EXACT visible Stage-1
    response used by the visible+embedding condition (task's explicit
    verification requirement for this factorial cell). Fails closed on any
    mismatch or missing question, listing up to 5 offending IDs."""
    missing = sorted(set(visible_llm_responses) ^ set(visible_embedding_responses))
    if missing:
        raise FactorialError(
            f"visible_llm and visible_embedding Stage-1 response sets differ: {missing[:5]}."
        )
    mismatched = sorted(
        qid for qid in visible_llm_responses
        if visible_llm_responses[qid] != visible_embedding_responses[qid]
    )
    if mismatched:
        raise FactorialError(
            f"visible_llm and visible_embedding do not share the exact same Stage-1 "
            f"response for question(s): {mismatched[:5]}."
        )
