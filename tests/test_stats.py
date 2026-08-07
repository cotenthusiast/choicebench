# tests/test_stats.py

import pandas as pd

from choicebench.benchmarks.base import make_normalized_row
from choicebench.stats import (
    choice_count_distribution,
    compute_benchmark_stats,
    modal_k,
    n_choices_for_row,
    read_stats,
    stats_path_for,
    write_stats,
)


def _mp_row(n: int, correct_index: int = 0) -> dict:
    return make_normalized_row("cat", "q", [f"o{i}" for i in range(n)], correct_index)


# ---------------------------------------------------------------------------
# modal_k + tiebreak
# ---------------------------------------------------------------------------

def test_modal_k_picks_most_common():
    assert modal_k({4: 2, 10: 9, 6: 1}) == 10


def test_modal_k_tiebreak_prefers_lowest_k():
    # 4 and 6 both appear 5 times — the lowest k wins deterministically.
    assert modal_k({6: 5, 4: 5}) == 4


def test_modal_k_three_way_tie_prefers_lowest():
    assert modal_k({10: 3, 4: 3, 6: 3}) == 4


# ---------------------------------------------------------------------------
# distribution + stats structure
# ---------------------------------------------------------------------------

def test_choice_count_distribution_new_schema():
    df = pd.DataFrame([_mp_row(4), _mp_row(4), _mp_row(10)])
    assert choice_count_distribution(df) == {4: 2, 10: 1}


def test_n_choices_for_row_counts_built_choices_when_n_choices_absent():
    row = _mp_row(6)
    del row["n_choices"]
    assert n_choices_for_row(row) == 6


def test_compute_benchmark_stats_reports_modal_k_and_proportion():
    df = pd.DataFrame([_mp_row(4)] * 3 + [_mp_row(10)] * 2)
    stats = compute_benchmark_stats(df, benchmark="demo")
    assert stats["benchmark"] == "demo"
    assert stats["n_questions"] == 5
    assert stats["modal_k"] == 4
    assert stats["modal_k_count"] == 3
    assert stats["modal_k_proportion"] == 3 / 5
    assert stats["choice_count_distribution"] == {"4": 3, "10": 2}


def test_compute_benchmark_stats_tiebreak_lowest_k():
    # Artificial tie: two 4-option and two 6-option questions.
    df = pd.DataFrame([_mp_row(4), _mp_row(4), _mp_row(6), _mp_row(6)])
    stats = compute_benchmark_stats(df, benchmark="tie")
    assert stats["modal_k"] == 4
    assert stats["modal_k_proportion"] == 0.5


def test_compute_benchmark_stats_empty():
    df = pd.DataFrame(columns=["choices_json", "correct_index"])
    stats = compute_benchmark_stats(df, benchmark="empty")
    assert stats["n_questions"] == 0
    assert stats["modal_k"] is None


# ---------------------------------------------------------------------------
# sidecar round-trip (readable independently of any method)
# ---------------------------------------------------------------------------

def test_stats_sidecar_round_trip(tmp_path):
    csv_path = tmp_path / "demo_normalized.csv"
    df = pd.DataFrame([_mp_row(4), _mp_row(4)])
    stats = compute_benchmark_stats(df, benchmark="demo")
    path = write_stats(stats, stats_path_for(csv_path))
    assert path == tmp_path / "demo_stats.json"
    assert read_stats(path) == stats
