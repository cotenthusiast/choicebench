# tests/test_pride_gate.py

import pandas as pd
import pytest

from choicebench.benchmarks.base import make_normalized_row
from choicebench.pride_gate import ModalKGateError, apply_modal_k_gate


def _rows(counts: dict[int, int]) -> pd.DataFrame:
    rows = []
    for k, n in counts.items():
        for _ in range(n):
            rows.append(make_normalized_row("cat", f"q{k}_{_}", [f"o{i}" for i in range(k)], 0))
    return pd.DataFrame(rows)


def test_gate_passes_when_coverage_meets_threshold():
    # 96/100 have modal k=4 → passes at threshold 0.95.
    df = _rows({4: 96, 6: 4})
    filtered, report = apply_modal_k_gate(df, modal_k=4, threshold=0.95, benchmark="demo")
    assert len(filtered) == 96
    assert report.n_total == 100
    assert report.n_evaluated == 96
    assert report.n_excluded == 4
    assert report.proportion == 0.96
    assert "excluded 4" in report.reason


def test_gate_raises_below_threshold_with_informative_message():
    # 83/100 modal k=10 → below 0.95.
    df = _rows({10: 83, 4: 17})
    with pytest.raises(ModalKGateError) as exc:
        apply_modal_k_gate(df, modal_k=10, threshold=0.95, benchmark="mmlu_pro")
    msg = str(exc.value)
    assert "mmlu_pro" in msg          # names the benchmark
    assert "modal k=10" in msg        # names the modal k
    assert "0.830" in msg             # names the actual proportion
    assert "0.95" in msg              # names the configured threshold


def test_gate_excludes_non_modal_and_reports_counts():
    df = _rows({4: 8, 10: 2})
    filtered, report = apply_modal_k_gate(df, modal_k=4, threshold=0.5, benchmark="demo")
    # Only modal-k rows survive; none of the excluded rows are forced through.
    from choicebench.stats import n_choices_for_row
    assert all(n_choices_for_row(r) == 4 for r in filtered.to_dict("records"))
    assert report.n_evaluated == 8
    assert report.n_excluded == 2


def test_gate_all_modal_reports_zero_excluded():
    df = _rows({4: 10})
    filtered, report = apply_modal_k_gate(df, modal_k=4, threshold=0.95, benchmark="demo")
    assert report.n_excluded == 0
    assert report.n_evaluated == 10
    assert "none excluded" in report.reason


def test_gate_raises_on_empty_question_set():
    with pytest.raises(ModalKGateError, match="no questions"):
        apply_modal_k_gate(pd.DataFrame([]), modal_k=4, threshold=0.95, benchmark="demo")
