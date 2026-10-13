import importlib.util
import json
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("te_diag", REPO_ROOT / "scripts/paper/te_corrected_diagnostics.py")
diag = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(diag)


def _frame(choices, gold=("A", "B", "C", "A")):
    return pd.DataFrame({"question_id": [f"q{i}" for i in range(len(gold))], "correct_option": list(gold),
                         "parsed_choice": list(choices),
                         "is_correct": [c == g if c is not None else None for c, g in zip(choices, gold)]})


def _cell(visible=True):
    return {"text_extraction_original": _frame([None, "B", None, "A"]), "text_extraction_corrected": _frame(["A", "B", "C", "A"]),
            "baseline": _frame(["A", "B", "A", "A"]), "two_stage": _frame(["A", "A", "C", "A"]),
            "semantic_matching": _frame(["A", "B", None, None]), "visible_llm_matcher": _frame(["A", "B", "C", "A"]) if visible else None}


def test_tables_have_one_original_and_one_corrected_row_per_cell_and_corrected_is_higher():
    metrics, grid, paired = diag.build_tables({("gpt", "mmlu"): _cell()})
    assert sorted(metrics["variant"]) == ["text_extraction_corrected", "text_extraction_original"]
    orig, corr = (metrics.set_index("variant").loc[v, "accuracy"] for v in ("text_extraction_original", "text_extraction_corrected"))
    assert (orig, corr) == (0.5, 1.0)
    assert grid.loc[0, "text_extraction_corrected_accuracy"] == 1.0 and grid.loc[0, "semantic_matching_coverage"] == 0.5


def test_paired_tests_compare_the_corrected_cell_with_every_other_cell_and_with_the_original():
    _, _, paired = diag.build_tables({("gpt", "mmlu"): _cell()})
    assert set(paired["b"]) == {"text_extraction_original", "baseline", "two_stage", "semantic_matching", "visible_llm_matcher"}
    row = paired.set_index("b").loc["text_extraction_original"]
    assert row["a_correct_b_wrong"] == 2 and row["a_wrong_b_correct"] == 0 and row["n_paired"] == 4


def test_a_cell_without_the_visible_matcher_is_reported_as_nan_not_dropped():
    _, grid, paired = diag.build_tables({("local_llama", "arc"): _cell(visible=False)})
    assert pd.isna(grid.loc[0, "visible_llm_matcher_accuracy"])
    assert "visible_llm_matcher" not in set(paired["b"])
