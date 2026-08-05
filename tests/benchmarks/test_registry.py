# tests/benchmarks/test_registry.py
#
# Covers the benchmark auto-discovery registry introduced in the
# mcq-framework registry refactor.

from __future__ import annotations

import ast
import importlib.util
import pathlib
from unittest.mock import patch

import pandas as pd
import pytest

import choicebench.benchmarks  # ensures all modules register themselves

from choicebench.benchmarks.registry import (
    BENCHMARK_REGISTRY,
    BenchmarkEntry,
    benchmark,
    get_by_hf_path,
)
from choicebench.config.schema import get_valid_benchmarks

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]

_EXPECTED_BENCHMARKS = [
    ("mmlu",         "cais/mmlu",            "all",              "test"),
    ("arc_challenge","allenai/ai2_arc",       "ARC-Challenge",    "test"),
    ("mmlu_pro",     "TIGER-Lab/MMLU-Pro",    None,               "test"),
    ("hellaswag",    "Rowan/hellaswag",        None,               "validation"),
    ("truthful_qa",  "truthfulqa/truthful_qa","multiple_choice",  "validation"),
]


# ---------------------------------------------------------------------------
# Registry population
# ---------------------------------------------------------------------------

def test_registry_populated_after_import():
    assert len(BENCHMARK_REGISTRY) >= 5


def test_all_five_registered_benchmarks_present():
    for name, *_ in _EXPECTED_BENCHMARKS:
        assert name in BENCHMARK_REGISTRY, f"{name!r} missing from BENCHMARK_REGISTRY"


@pytest.mark.parametrize("name,hf_path,hf_subset,default_split", _EXPECTED_BENCHMARKS)
def test_registry_entry_fields(name, hf_path, hf_subset, default_split):
    entry = BENCHMARK_REGISTRY[name]
    assert isinstance(entry, BenchmarkEntry)
    assert entry.name == name
    assert entry.hf_path == hf_path
    assert entry.hf_subset == hf_subset
    assert entry.default_split == default_split
    assert callable(entry.normalizer)


# ---------------------------------------------------------------------------
# get_by_hf_path
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name,hf_path,hf_subset,_split", _EXPECTED_BENCHMARKS)
def test_get_by_hf_path_returns_correct_entry(name, hf_path, hf_subset, _split):
    entry = get_by_hf_path(hf_path, hf_subset)
    assert entry is not None
    assert entry.name == name


def test_get_by_hf_path_returns_none_for_unknown():
    assert get_by_hf_path("no/such/dataset") is None


def test_get_by_hf_path_subset_mismatch_returns_none():
    assert get_by_hf_path("cais/mmlu", "wrong_subset") is None


def test_get_by_hf_path_old_truthful_qa_path_not_aliased():
    """The pre-move HF path (truthful_qa) is intentionally not registered as
    an alias for the canonical truthfulqa/truthful_qa path: it fails at
    datasets.load_dataset() before ever reaching the registry, so aliasing
    it here would not make old configs work."""
    assert get_by_hf_path("truthful_qa", "multiple_choice") is None


# ---------------------------------------------------------------------------
# get_valid_benchmarks
# ---------------------------------------------------------------------------

def test_get_valid_benchmarks_includes_all_registered():
    valid = get_valid_benchmarks()
    for name, *_ in _EXPECTED_BENCHMARKS:
        assert name in valid


def test_get_valid_benchmarks_includes_special_cases():
    valid = get_valid_benchmarks()
    assert "toy" in valid
    assert "huggingface" in valid


# ---------------------------------------------------------------------------
# Normalizers produce the variable-choice schema
# ---------------------------------------------------------------------------

_NORMALIZED_COLS = {
    "question_id", "subject", "question_text",
    "choices_json", "correct_index", "correct_option",
    "correct_answer_text", "n_choices",
}


def _make_mmlu_df() -> pd.DataFrame:
    # choices as string (legacy CSV format)
    return pd.DataFrame([{
        "subject": "math",
        "question": "What is 2+2?",
        "choices": '["1", "2", "3", "4"]',
        "answer": 3,
    }])


def _make_mmlu_df_hf_format() -> pd.DataFrame:
    # choices as Python list (HuggingFace native format)
    return pd.DataFrame([{
        "subject": "math",
        "question": "What is 2+2?",
        "choices": ["1", "2", "3", "4"],
        "answer": 3,
    }])


def _make_arc_df() -> pd.DataFrame:
    return pd.DataFrame([{
        "id": "Mercury_7038048",
        "question": "Which planet is closest to the Sun?",
        "choices": {"text": ["Venus", "Mercury", "Earth", "Mars"], "label": ["A", "B", "C", "D"]},
        "answerKey": "B",
    }])


def _make_mmlu_pro_df() -> pd.DataFrame:
    return pd.DataFrame([{
        "question": "What is the integral of x?",
        "options": ["x", "x^2/2", "2x", "x^2"],
        "answer": "B",
        "answer_index": 1,
        "category": "math",
        "src": "orig",
    }])


def _make_hellaswag_df() -> pd.DataFrame:
    return pd.DataFrame([{
        "ind": "1",
        "activity_label": "Archery",
        "ctx": "They drew back their bow and",
        "endings": ["released the arrow.", "fell asleep.", "ate lunch.", "drank water."],
        "label": 0,
    }])


def _make_truthful_qa_df() -> pd.DataFrame:
    return pd.DataFrame([{
        "question": "What happens if you eat a watermelon seed?",
        "mc1_targets": {
            "choices": ["Nothing special.", "A vine grows.", "You get sick.", "It helps digestion."],
            "labels": [1, 0, 0, 0],
        },
        "mc2_targets": {"choices": [], "labels": []},
    }])


@pytest.mark.parametrize("name,make_df", [
    ("mmlu",          _make_mmlu_df),
    ("arc_challenge", _make_arc_df),
    ("mmlu_pro",      _make_mmlu_pro_df),
    ("hellaswag",     _make_hellaswag_df),
    ("truthful_qa",   _make_truthful_qa_df),
])
def test_normalizer_produces_normalized_schema(name, make_df):
    entry = BENCHMARK_REGISTRY[name]
    result = entry.normalizer(make_df())
    assert not result.empty
    assert set(result.columns) == _NORMALIZED_COLS


# ---------------------------------------------------------------------------
# load_benchmark() error command generation (run_experiment.py)
# ---------------------------------------------------------------------------

def _load_run_experiment():
    import choicebench.cli.run_experiment as module
    return importlib.reload(module)


@pytest.mark.parametrize("name,hf_path,hf_subset,default_split", _EXPECTED_BENCHMARKS)
def test_load_benchmark_missing_csv_generates_correct_command(
    tmp_path, monkeypatch, name, hf_path, hf_subset, default_split
):
    run_exp = _load_run_experiment()

    # Hermetic: resolve the normalized-CSV path into an empty tmp dir so the
    # "missing CSV" branch is exercised regardless of whatever the developer
    # has prepared under data/processed locally.
    monkeypatch.setattr(run_exp, "PROCESSED_DIR", tmp_path)

    from choicebench.config.schema import BenchmarkConfig

    with pytest.raises(FileNotFoundError) as exc_info:
        run_exp.load_benchmark_selection(BenchmarkConfig(name=name, split=default_split), run_seed=42)

    msg = str(exc_info.value)
    assert f"--hf-path {hf_path}" in msg
    if hf_subset:
        assert f"--hf-subset {hf_subset}" in msg
    if default_split != "test":
        assert f"--split {default_split}" in msg


def test_load_benchmark_unregistered_huggingface_gives_non_circular_error():
    # A `name: huggingface` dataset with no registered normalizer must NOT loop
    # the user back to a prepare_data.py command that itself raises
    # NotImplementedError. The error must point to the Add-a-benchmark workflow.
    run_exp = _load_run_experiment()

    from choicebench.config.schema import BenchmarkConfig

    bench = BenchmarkConfig(
        name="huggingface",
        hf_path="someorg/my-oneoff-mcq",
        hf_subset=None,
    )
    with pytest.raises(FileNotFoundError) as exc_info:
        run_exp.load_benchmark_selection(bench, run_seed=42)

    msg = str(exc_info.value)
    # Names the offending dataset and points to the real next step.
    assert "someorg/my-oneoff-mcq" in msg
    assert "Add a benchmark" in msg
    assert "@benchmark" in msg
    # Does NOT send the user back to the failing prepare_data.py command.
    assert "python scripts/prepare_data.py --hf-path" not in msg


# ---------------------------------------------------------------------------
# normalize_to_schema() routing in prepare_data.py
# ---------------------------------------------------------------------------

def _load_prepare_data():
    import choicebench.cli.prepare_data as module
    return importlib.reload(module)


@pytest.mark.parametrize("name,make_df,hf_path,hf_subset", [
    ("mmlu",          _make_mmlu_df,          "cais/mmlu",         "all"),
    ("arc_challenge", _make_arc_df,           "allenai/ai2_arc",   "ARC-Challenge"),
    ("mmlu_pro",      _make_mmlu_pro_df,      "TIGER-Lab/MMLU-Pro", None),
    ("hellaswag",     _make_hellaswag_df,     "Rowan/hellaswag",    None),
    ("truthful_qa",   _make_truthful_qa_df,   "truthfulqa/truthful_qa", "multiple_choice"),
])
def test_normalize_to_schema_routes_correctly(name, make_df, hf_path, hf_subset):
    prepare = _load_prepare_data()
    result = prepare.normalize_to_schema(make_df(), hf_path, hf_subset)
    assert not result.empty
    assert set(result.columns) == _NORMALIZED_COLS


def test_normalize_to_schema_raises_not_implemented_for_unknown():
    prepare = _load_prepare_data()
    df = pd.DataFrame([{"col": "val"}])
    with pytest.raises(NotImplementedError) as exc_info:
        prepare.normalize_to_schema(df, "unknown/dataset", "subset")
    msg = str(exc_info.value)
    assert "unknown/dataset" in msg
    assert "subset" in msg
    assert "src/choicebench/benchmarks/__init__.py" in msg


# ---------------------------------------------------------------------------
# New @benchmark registrations are picked up dynamically
# ---------------------------------------------------------------------------

def test_mmlu_normalizer_accepts_hf_list_format():
    """MMLU normalizer must handle choices as a Python list (HuggingFace native)."""
    import json

    entry = BENCHMARK_REGISTRY["mmlu"]
    result = entry.normalizer(_make_mmlu_df_hf_format())
    assert not result.empty
    assert set(result.columns) == _NORMALIZED_COLS
    choices = json.loads(result.iloc[0]["choices_json"])
    assert [c["text"] for c in choices] == ["1", "2", "3", "4"]
    assert result.iloc[0]["correct_option"] == "D"
    assert result.iloc[0]["correct_answer_text"] == "4"


def test_registering_new_benchmark_adds_to_registry_and_valid_benchmarks():
    original_keys = set(BENCHMARK_REGISTRY.keys())

    @benchmark(name="_test_dummy_bench", hf_path="test/dummy", hf_subset=None)
    def _dummy_normalizer(row: dict) -> dict:
        return dict(row)

    try:
        assert "_test_dummy_bench" in BENCHMARK_REGISTRY
        assert "_test_dummy_bench" in get_valid_benchmarks()
        entry = BENCHMARK_REGISTRY["_test_dummy_bench"]
        assert entry.hf_path == "test/dummy"
        df = pd.DataFrame([{"a": 1}])
        pd.testing.assert_frame_equal(entry.normalizer(df), df)
    finally:
        BENCHMARK_REGISTRY.pop("_test_dummy_bench", None)
        assert "_test_dummy_bench" not in BENCHMARK_REGISTRY


def test_get_by_hf_path_finds_newly_registered_benchmark():
    @benchmark(name="_test_find_bench", hf_path="find/me", hf_subset="sub")
    def _find_normalizer(row: dict) -> dict:
        return dict(row)

    try:
        found = get_by_hf_path("find/me", "sub")
        assert found is not None
        assert found.name == "_test_find_bench"
    finally:
        BENCHMARK_REGISTRY.pop("_test_find_bench", None)
