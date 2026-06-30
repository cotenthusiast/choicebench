# tests/scripts/test_prepare_data.py
#
# Regression for MF-D: preparing a registered hf_path with no --output-name must
# default the CSV stem to the registry entry's name (e.g. arc_challenge), which
# is exactly the file load_benchmark() looks for. Otherwise the stem is derived
# from the hf_path ("ai2_arc") and the run hits a self-contradicting dead end.

import importlib.util
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_prepare_data():
    spec = importlib.util.spec_from_file_location(
        "prepare_data_under_test",
        _REPO_ROOT / "scripts" / "prepare_data.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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
