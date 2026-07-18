# tests/scripts/test_prepare_data.py
#
# Regression for MF-D: preparing a registered hf_path with no --output-name must
# default the CSV stem to the registry entry's name (e.g. arc_challenge), which
# is exactly the file load_benchmark() looks for. Otherwise the stem is derived
# from the hf_path ("ai2_arc") and the run hits a self-contradicting dead end.

import importlib
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_prepare_data():
    import choicebench.cli.prepare_data as module
    return importlib.reload(module)


def test_registered_hf_path_defaults_stem_to_registry_name():
    mod = _load_prepare_data()
    # allenai/ai2_arc + ARC-Challenge is the registered arc_challenge benchmark.
    stem = mod._resolve_output_stem(None, "allenai/ai2_arc", "ARC-Challenge")
    assert stem == "arc_challenge"


def test_explicit_output_name_wins_over_registry():
    mod = _load_prepare_data()
    stem = mod._resolve_output_stem("custom", "allenai/ai2_arc", "ARC-Challenge")
    assert stem == "custom"


def test_unregistered_hf_path_derives_from_tail():
    mod = _load_prepare_data()
    stem = mod._resolve_output_stem(None, "some-org/My-Cool-Bench", None)
    assert stem == "my_cool_bench"


import json

import pandas as pd
import pytest


def test_filter_flag_appends_filtered_suffix_when_no_output_name(monkeypatch, tmp_path):
    mod = _load_prepare_data()
    monkeypatch.setattr(mod, "PROCESSED_DIR", tmp_path)

    raw_df = pd.DataFrame(
        [
            {"subject": "x", "question": "q1", "choices": "['4','5','6','7']", "answer": 0},
            {"subject": "x", "question": "q2", "choices": "['A and B','5','6','7']", "answer": 1},
        ]
    )
    monkeypatch.setattr(mod, "download_from_huggingface", lambda *a, **k: raw_df)
    monkeypatch.setattr(
        mod,
        "parse_args",
        lambda: __import__("argparse").Namespace(
            hf_path="cais/mmlu",
            hf_subset="all",
            split="test",
            output_name=None,
            filter_permutation_unsafe=True,
        ),
    )

    mod.main()

    out_csv = next(tmp_path.rglob("normalized.csv"))
    assert out_csv.exists()
    df = pd.read_csv(out_csv)
    assert len(df) == 1  # q2 excluded

    sidecar = out_csv.with_name("normalized_permutation_filter.json")
    report = json.loads(sidecar.read_text())
    assert report["n_total"] == 2
    assert report["n_excluded"] == 1
    assert report["exclusions"][0]["matched_pattern"] == "letter_and_letter"


def test_filter_flag_respects_explicit_output_name(monkeypatch, tmp_path):
    mod = _load_prepare_data()
    monkeypatch.setattr(mod, "PROCESSED_DIR", tmp_path)

    raw_df = pd.DataFrame(
        [{"subject": "x", "question": "q1", "choices": "['4','5','6','7']", "answer": 0}]
    )
    monkeypatch.setattr(mod, "download_from_huggingface", lambda *a, **k: raw_df)
    monkeypatch.setattr(
        mod,
        "parse_args",
        lambda: __import__("argparse").Namespace(
            hf_path="cais/mmlu",
            hf_subset="all",
            split="test",
            output_name="custom_stem",
            filter_permutation_unsafe=True,
        ),
    )

    mod.main()

    out = next(tmp_path.rglob("normalized.csv"))
    assert "custom_stem" in out.parts
    assert out.with_name("artifact.json").exists()


def test_no_filter_flag_leaves_output_unfiltered(monkeypatch, tmp_path):
    mod = _load_prepare_data()
    monkeypatch.setattr(mod, "PROCESSED_DIR", tmp_path)

    raw_df = pd.DataFrame(
        [{"subject": "x", "question": "q1", "choices": "['A and B','5','6','7']", "answer": 0}]
    )
    monkeypatch.setattr(mod, "download_from_huggingface", lambda *a, **k: raw_df)
    monkeypatch.setattr(
        mod,
        "parse_args",
        lambda: __import__("argparse").Namespace(
            hf_path="cais/mmlu",
            hf_subset="all",
            split="test",
            output_name=None,
            filter_permutation_unsafe=False,
        ),
    )

    mod.main()

    out = next(tmp_path.rglob("normalized.csv"))
    df = pd.read_csv(out)
    assert len(df) == 1  # kept — filter never ran
    assert not out.with_name("normalized_permutation_filter.json").exists()
