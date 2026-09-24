# tests/scripts/test_paper_derive_semantic_matching.py

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT_PATH = REPO_ROOT / "scripts" / "paper" / "derive_semantic_matching.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("derive_semantic_matching", _SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


mod = _load_module()


def _source_csv(tmp_path) -> Path:
    choices_json = json.dumps([
        {"text": "FTP", "source_index": 0}, {"text": "HTTP", "source_index": 1},
        {"text": "HTTPS", "source_index": 2}, {"text": "SMTP", "source_index": 3},
    ])
    df = pd.DataFrame([
        dict(run_id="src_run", benchmark_name="mmlu", question_id="q1",
             method_name="two_stage_v1", seed=42, choices_json=choices_json,
             correct_option="C", free_text_response="HTTPS"),
        dict(run_id="src_run", benchmark_name="mmlu", question_id="q2",
             method_name="two_stage_v1", seed=42, choices_json=choices_json,
             correct_option="A", free_text_response="FTP"),
    ])
    path = tmp_path / "two_stage_v1_result.csv"
    df.to_csv(path, index=False)
    return path


def test_derive_reads_csv_writes_csv_with_zero_new_calls(tmp_path, monkeypatch):
    # Stub the real embedder so this test stays fast/offline -- patched on
    # the analysis module derive_from_free_text imports from.
    import choicebench.analysis.derive_from_free_text as derive_mod
    monkeypatch.setattr(derive_mod, "default_embed_fn", lambda texts: np.eye(len(texts)))

    source = _source_csv(tmp_path)
    output = tmp_path / "out" / "semantic_matching_v1_result.csv"

    derived_df = mod.derive(source, output)

    assert output.exists()
    assert list(derived_df["method_name"]) == ["semantic_matching_v1", "semantic_matching_v1"]
    assert list(derived_df["parsed_choice"]) == ["C", "A"]
    assert list(derived_df["derived_from_method_name"]) == ["two_stage_v1", "two_stage_v1"]

    reloaded = pd.read_csv(output)
    assert len(reloaded) == 2
    assert list(reloaded["parsed_choice"]) == ["C", "A"]
