# src/choicebench/pride_gate.py

"""PriDe modal-k compatibility gate.

PriDe estimates a single global positional-bias prior and applies it to every
evaluation question (Eq. 8). That calibration only makes sense when the
evaluation questions share one option count. This gate enforces that: given a
benchmark's precomputed modal choice count (k) and a configured threshold, it
either lets the run proceed on the modal-k subset, or refuses when too few
questions match.

This check is PriDe-specific by design. direct_mcq and cyclic_permutation are
per-question methods with no global calibration step, so a mix of option counts
does not break them and they are never subject to this gate.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd

from choicebench.stats import n_choices_for_row


class ModalKGateError(Exception):
    """Raised when a benchmark's modal-k coverage is below the PriDe threshold.

    Carries the full accounting in `report` so a caller that wants to record
    *why* PriDe was skipped for this benchmark (a by-design exclusion, not a
    bug) doesn't have to re-derive it from the message string.
    """

    def __init__(self, message: str, report: "ModalKGateReport | None" = None) -> None:
        super().__init__(message)
        self.report = report


@dataclass
class ModalKGateReport:
    benchmark: str
    modal_k: int
    threshold: float
    n_total: int
    n_evaluated: int
    n_excluded: int
    proportion: float
    reason: str

    def as_dict(self) -> dict:
        return asdict(self)


def apply_modal_k_gate(
    questions: pd.DataFrame,
    modal_k: int,
    threshold: float,
    benchmark: str,
) -> tuple[pd.DataFrame, ModalKGateReport]:
    """Gate a PriDe evaluation set on modal-k coverage.

    Args:
        questions: The evaluation questions (already loaded/filtered/sampled).
        modal_k: The benchmark's precomputed modal choice count.
        threshold: Minimum proportion of questions that must have modal_k
            options for the run to proceed. Must be in (0, 1].
        benchmark: Benchmark label, for messages/reporting.

    Returns:
        (filtered_questions, report) where filtered_questions contains only the
        modal-k questions and report records the exclusion accounting.

    Raises:
        ModalKGateError: if the modal-k proportion is below threshold. The
            message names the benchmark, modal k, the actual proportion, and
            the configured threshold.
    """
    n_total = int(len(questions))
    if n_total == 0:
        raise ModalKGateError(
            f"PriDe modal-k gate: benchmark {benchmark!r} has no questions to evaluate.",
            report=ModalKGateReport(
                benchmark=benchmark,
                modal_k=int(modal_k),
                threshold=float(threshold),
                n_total=0,
                n_evaluated=0,
                n_excluded=0,
                proportion=0.0,
                reason="benchmark has no questions to evaluate",
            ),
        )

    counts = questions.apply(lambda r: n_choices_for_row(r.to_dict()), axis=1)
    match_mask = counts == modal_k
    n_evaluated = int(match_mask.sum())
    n_excluded = n_total - n_evaluated
    proportion = n_evaluated / n_total

    if proportion < threshold:
        raise ModalKGateError(
            f"PriDe modal-k gate failed for benchmark {benchmark!r}: modal k={modal_k}, "
            f"but only {n_evaluated}/{n_total} questions ({proportion:.3f}) have {modal_k} "
            f"options — below the configured pride.modal_k_threshold={threshold}. "
            f"PriDe's global calibration assumes a single option count. Run PriDe on a "
            f"benchmark whose modal-k coverage is at least {threshold:.0%}, or lower "
            f"pride.modal_k_threshold.",
            report=ModalKGateReport(
                benchmark=benchmark,
                modal_k=int(modal_k),
                threshold=float(threshold),
                n_total=n_total,
                n_evaluated=n_evaluated,
                n_excluded=n_excluded,
                proportion=float(proportion),
                reason=(
                    f"skipped by design: only {n_evaluated}/{n_total} question(s) have "
                    f"modal k={modal_k}, below the {threshold:.0%} coverage threshold"
                ),
            ),
        )

    filtered = questions[match_mask].copy()
    reason = (
        f"excluded {n_excluded} question(s) whose option count != modal k={modal_k}"
        if n_excluded
        else f"all {n_total} question(s) have modal k={modal_k}; none excluded"
    )
    report = ModalKGateReport(
        benchmark=benchmark,
        modal_k=int(modal_k),
        threshold=float(threshold),
        n_total=n_total,
        n_evaluated=n_evaluated,
        n_excluded=n_excluded,
        proportion=float(proportion),
        reason=reason,
    )
    return filtered, report
