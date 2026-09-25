# tests/scripts/test_paper_derive_baseline_from_cyclic.py

import importlib.util
import json
from pathlib import Path

import pandas as pd

from choicebench.clients.types import SUCCESS_STATUS

REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT_PATH = REPO_ROOT / "scripts" / "paper" / "derive_baseline_from_cyclic.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("derive_baseline_from_cyclic_script", _SCRIPT_PATH)
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
             method_name="cyclic_permutation", seed=42, choices_json=choices_json,
             correct_option="C", raw_text="C", transport_status=SUCCESS_STATUS,
             # majority vote deliberately differs from rotation 0's own "C",
             # to prove the derivation uses rotation 0, not this column.
             parsed_choice="A", is_correct=False, answer_status=SUCCESS_STATUS,
             per_rotation_choices_json=json.dumps(["C", "A", "A", "A"])),
        dict(run_id="src_run", benchmark_name="mmlu", question_id="q2",
             method_name="cyclic_permutation", seed=42, choices_json=choices_json,
             correct_option="A", raw_text="A", transport_status=SUCCESS_STATUS,
             parsed_choice="A", is_correct=True, answer_status=SUCCESS_STATUS,
             per_rotation_choices_json=json.dumps(["A", "A", "A", "A"])),
    ])
    path = tmp_path / "cyclic_permutation_result.csv"
    df.to_csv(path, index=False)
    return path


def test_derive_reads_csv_writes_csv_with_zero_new_calls():
    import inspect
    assert not any("backend" in p or "client" in p for p in inspect.signature(mod.derive).parameters)


def test_derive_uses_rotation0_not_the_majority_vote(tmp_path):
    source = _source_csv(tmp_path)
    output = tmp_path / "out" / "direct_mcq_result.csv"

    derived_df = mod.derive(source, output)

    assert output.exists()
    assert list(derived_df["method_name"]) == ["direct_mcq", "direct_mcq"]
    # q1: rotation 0 raw_text="C" (correct), even though the source row's
    # own majority-vote parsed_choice was "A" (incorrect).
    assert list(derived_df["parsed_choice"]) == ["C", "A"]
    assert list(derived_df["is_correct"]) == [True, True]
    assert list(derived_df["derived_from_method_name"]) == ["cyclic_permutation", "cyclic_permutation"]

    reloaded = pd.read_csv(output)
    assert len(reloaded) == 2
    assert list(reloaded["parsed_choice"]) == ["C", "A"]
    assert "per_rotation_choices_json" not in reloaded.columns


def test_method_name_is_overridable_for_reasoning_mcq(tmp_path):
    source = _source_csv(tmp_path)
    output = tmp_path / "reasoning_mcq_result.csv"

    derived_df = mod.derive(source, output, method_name="reasoning_mcq")

    assert list(derived_df["method_name"]) == ["reasoning_mcq", "reasoning_mcq"]
