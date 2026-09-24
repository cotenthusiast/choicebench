# tests/scripts/test_paper_freeze_stochasticity_manifests.py
#
# Pins the deterministic hash-based stochasticity subset selection
# (stochasticity-subset-v1 protocol) used by
# scripts/paper/freeze_stochasticity_manifests.py.

import importlib.util
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT_PATH = REPO_ROOT / "scripts" / "paper" / "freeze_stochasticity_manifests.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("freeze_stochasticity_manifests", _SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


mod = _load_module()


def test_select_returns_requested_count():
    ids = [f"q{i}" for i in range(500)]
    subset = mod.select_stochasticity_subset(ids, "mmlu")
    assert len(subset) == mod.N_STOCHASTICITY


def test_select_is_deterministic():
    ids = [f"q{i}" for i in range(300)]
    a = mod.select_stochasticity_subset(ids, "arc_challenge")
    b = mod.select_stochasticity_subset(ids, "arc_challenge")
    assert a == b


def test_select_is_invariant_to_input_order():
    ids = [f"q{i}" for i in range(300)]
    forward = mod.select_stochasticity_subset(ids, "mmlu")
    shuffled = mod.select_stochasticity_subset(list(reversed(ids)), "mmlu")
    assert set(forward) == set(shuffled)


def test_select_differs_across_benchmarks_for_the_same_ids():
    """The benchmark_id is part of the hash key, so the same underlying
    question_id space (a contrived overlap) must not select the same
    subset for two different benchmark_ids."""
    ids = [f"q{i}" for i in range(300)]
    mmlu_subset = set(mod.select_stochasticity_subset(ids, "mmlu"))
    arc_subset = set(mod.select_stochasticity_subset(ids, "arc_challenge"))
    assert mmlu_subset != arc_subset


def test_all_selected_ids_come_from_the_input_population():
    ids = [f"q{i}" for i in range(150)]
    subset = mod.select_stochasticity_subset(ids, "mmlu")
    assert set(subset).issubset(set(ids))


def test_raises_naturally_if_population_smaller_than_requested_count():
    ids = [f"q{i}" for i in range(10)]
    subset = mod.select_stochasticity_subset(ids, "mmlu")
    # No explicit guard in the selector -- slicing a shorter list just
    # returns everything available. Document that behavior explicitly
    # rather than leaving it as an untested edge case.
    assert len(subset) == 10
