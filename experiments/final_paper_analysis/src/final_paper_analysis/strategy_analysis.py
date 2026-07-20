"""Mentor-driven strategy analysis (spec section 7).

Answers: do any tested interventions reliably improve accuracy and reduce
positional bias across models and benchmarks? Operates on one ``CellEffect``
per (model, benchmark, strategy) triple -- callers derive availability of a
given strategy for a given model/benchmark from the verified canonical
inventory (Component 2/3), never assuming every strategy exists for every
model x benchmark. Macro-averages are unweighted means over cells (one vote
per cell), so no single model or benchmark can dominate through having more
rows/questions than another.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Callable, Sequence

_USABLE_STATUSES = frozenset({"included", "qualified"})


class StrategyAnalysisError(ValueError):
    """Raised when strategy-analysis input is structurally invalid."""


@dataclass(frozen=True)
class CellEffect:
    model_key: str
    benchmark_name: str
    strategy: str
    baseline_metric: float
    strategy_metric: float
    status: str  # included | qualified | incomplete | excluded

    @property
    def effect(self) -> float:
        return self.strategy_metric - self.baseline_metric

    @property
    def usable(self) -> bool:
        return self.status in _USABLE_STATUSES


def _usable(effects: Sequence[CellEffect]) -> list[CellEffect]:
    return [effect for effect in effects if effect.usable]


@dataclass(frozen=True)
class MacroAverageResult:
    macro_average_effect: float
    n_usable_cells: int
    n_incomplete_cells: int
    n_excluded_cells: int


def macro_average_effect(effects: Sequence[CellEffect]) -> MacroAverageResult:
    """Unweighted mean of .effect across usable (included/qualified) cells
    only -- incomplete/excluded cells are counted and reported, never
    silently folded into the average as if they were a real (bad) effect."""
    usable = _usable(effects)
    incomplete = sum(1 for e in effects if e.status == "incomplete")
    excluded = sum(1 for e in effects if e.status == "excluded")
    if not usable:
        return MacroAverageResult(
            macro_average_effect=float("nan"), n_usable_cells=0,
            n_incomplete_cells=incomplete, n_excluded_cells=excluded,
        )
    mean_effect = sum(e.effect for e in usable) / len(usable)
    return MacroAverageResult(
        macro_average_effect=mean_effect, n_usable_cells=len(usable),
        n_incomplete_cells=incomplete, n_excluded_cells=excluded,
    )


@dataclass(frozen=True)
class DirectionConsistency:
    n_improved: int
    n_worsened: int
    n_unchanged: int
    fraction_improved: float


def direction_of_effect_consistency(
    effects: Sequence[CellEffect], *, tie_epsilon: float = 1e-9
) -> DirectionConsistency:
    usable = _usable(effects)
    if not usable:
        return DirectionConsistency(0, 0, 0, float("nan"))
    improved = sum(1 for e in usable if e.effect > tie_epsilon)
    worsened = sum(1 for e in usable if e.effect < -tie_epsilon)
    unchanged = len(usable) - improved - worsened
    return DirectionConsistency(
        n_improved=improved, n_worsened=worsened, n_unchanged=unchanged,
        fraction_improved=improved / len(usable),
    )


@dataclass(frozen=True)
class WinTieLoss:
    wins: int
    ties: int
    losses: int
    n_cells: int


def win_tie_loss_counts(effects: Sequence[CellEffect], *, tie_epsilon: float = 1e-9) -> WinTieLoss:
    """Cell-level win/tie/loss (one vote per model-benchmark cell), distinct
    from Component 5's question-level paired win/loss/tie."""
    consistency = direction_of_effect_consistency(effects, tie_epsilon=tie_epsilon)
    return WinTieLoss(
        wins=consistency.n_improved, losses=consistency.n_worsened,
        ties=consistency.n_unchanged, n_cells=consistency.n_improved + consistency.n_worsened + consistency.n_unchanged,
    )


def heterogeneity_by_group(
    effects: Sequence[CellEffect], *, group_key: Callable[[CellEffect], str]
) -> dict[str, MacroAverageResult]:
    """Macro-average effect broken down by an arbitrary grouping (e.g. by
    model_key, or by benchmark_name) -- reveals whether an aggregate effect
    is driven by one subgroup rather than holding broadly."""
    groups: dict[str, list[CellEffect]] = {}
    for effect in effects:
        groups.setdefault(group_key(effect), []).append(effect)
    return {group: macro_average_effect(group_effects) for group, group_effects in groups.items()}


@dataclass(frozen=True)
class OutlierReport:
    outliers: tuple[CellEffect, ...]
    mean_effect: float
    stdev_effect: float
    z_threshold: float


def detect_outliers(effects: Sequence[CellEffect], *, z_threshold: float = 2.0) -> OutlierReport:
    """Flag usable cells whose effect deviates by more than ``z_threshold``
    standard deviations from the macro-average -- explicit outlier
    reporting, never silently smoothed into the aggregate. Requires >=2
    usable cells to define a standard deviation; fewer returns an empty
    report with stdev_effect=NaN."""
    usable = _usable(effects)
    if len(usable) < 2:
        mean_effect = usable[0].effect if usable else float("nan")
        return OutlierReport(outliers=(), mean_effect=mean_effect, stdev_effect=float("nan"), z_threshold=z_threshold)
    values = [e.effect for e in usable]
    mean_effect = statistics.fmean(values)
    stdev_effect = statistics.stdev(values)
    if stdev_effect == 0:
        return OutlierReport(outliers=(), mean_effect=mean_effect, stdev_effect=0.0, z_threshold=z_threshold)
    outliers = tuple(
        e for e in usable if abs((e.effect - mean_effect) / stdev_effect) > z_threshold
    )
    return OutlierReport(outliers=outliers, mean_effect=mean_effect, stdev_effect=stdev_effect, z_threshold=z_threshold)


def derive_available_strategies(
    inventory_status: dict[tuple[str, str, str], str]
) -> dict[str, set[tuple[str, str]]]:
    """Given {(model_key, benchmark_name, strategy): status}, return
    {strategy: {(model_key, benchmark_name) pairs where that strategy has a
    usable (included/qualified) cell}}. This is the mechanism for deriving
    strategy availability from the verified canonical inventory rather than
    assuming a complete matrix -- a strategy present for only 3 of 12
    model-benchmark pairs must only be macro-averaged over those 3."""
    result: dict[str, set[tuple[str, str]]] = {}
    for (model_key, benchmark_name, strategy), status in inventory_status.items():
        if status in _USABLE_STATUSES:
            result.setdefault(strategy, set()).add((model_key, benchmark_name))
    return result
