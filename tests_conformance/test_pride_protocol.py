"""Section J: PriDe calibration.

Frozen invariant: PriDe's calibration pool is built eligibility-BEFORE-
sampling (never sample-then-filter), the runtime PriDeRunner's own
_calibration_n attribute (not merely the YAML's preflight.n field) must
equal the frozen calibration_n (77 for MMLU, 15 for ARC), calibration rows
are disjoint from evaluation rows, selection is deterministic under seed
42, an insufficient eligible pool is a hard failure under
require_full_calibration, and PriDe uses full-vocabulary local scoring
(HuggingFace/Dummy backend.score_options()) rather than vLLM/top-k --
structurally enforced by config validation itself, no API provider ever
satisfies requires_logprobs.

J10 (full-logit local path) and J11 (no alpha parameter) are covered as
config-level structural checks in test_production_configs.py
(test_model_supports_logprobs_structural_gate, test_pride_has_no_alpha_parameter)
since they're properties of config/schema.py's validation gate, not of a
particular PriDeRunner instance.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from choicebench.backends.dummy_backend import DummyBackend
from choicebench.cli.run_experiment import instantiate_runner, load_benchmark_selection
from choicebench.config.schema import load_config
from choicebench.methods.library.pride import PriDeRunner
from choicebench.preflight import load_preflight

REPO_ROOT = Path(__file__).resolve().parents[1]
PAPER_CONFIG_DIR = REPO_ROOT / "config" / "paper"
PROMPTS_DIR = REPO_ROOT / "prompts"


def _real_calibration_n(config_name: str, run_id: str, calibration_runs_dir: Path) -> tuple[int, int]:
    """Build a PriDeRunner through the exact production construction path
    (load_config -> load_benchmark_selection -> load_preflight ->
    instantiate_runner) against the REAL frozen manifest + real prepared
    validation-split data, with only the backend faked. Returns
    (runtime_calibration_n, n_preflight_records_loaded).

    instantiate_runner() auto-injects calibration_runs_dir=RUNS_DIR (the
    real repo runs/ directory) whenever a runner accepts that parameter and
    the caller didn't set it -- explicitly override it via
    extra_runtime_kwargs so the calibration sidecar JSON lands under
    tmp_path instead (spec R5: never touch real historical outputs)."""
    config = load_config(str(PAPER_CONFIG_DIR / config_name))
    benchmark_cfg = config.benchmarks[0]
    method_cfg = config.methods[0]
    assert method_cfg.name == "pride"

    selection = load_benchmark_selection(benchmark_cfg, config.run.seed)
    preflight_records = load_preflight(
        method_cfg, benchmark_cfg, config.run.seed,
        eval_question_ids=set(selection.questions["question_id"]),
        eval_artifact_id=selection.artifact.artifact_id,
        eval_sample_identities=set(selection.sample_identities),
    )

    backend = DummyBackend()
    runner = instantiate_runner(
        config, config.models[0], method_cfg, backend, run_id, benchmark_cfg,
        preflight_questions=preflight_records,
        extra_runtime_kwargs={"calibration_runs_dir": calibration_runs_dir},
    )
    assert isinstance(runner, PriDeRunner)
    runner._ensure_calibration()
    return runner._calibration_n, len(preflight_records)


@pytest.mark.skipif(
    not (REPO_ROOT / "data" / "processed" / "mmlu" / "validation").exists(),
    reason="requires the prepared MMLU validation-split source under data/processed/mmlu/validation/",
)
def test_j1_mmlu_pride_runtime_calibration_n_is_77(tmp_path):
    calibration_n, n_loaded = _real_calibration_n("mmlu_pride.yaml", "conformance-j1", tmp_path)
    assert calibration_n == 77, (
        f"runtime PriDeRunner._calibration_n={calibration_n}, expected 77. "
        "(YAML.preflight.n and YAML.methods[].params.calibration_n must both be "
        "set to 77, or the runner silently calibrates on the params.calibration_n "
        "default of 50 instead.)"
    )
    assert n_loaded == 77


@pytest.mark.skipif(
    not (REPO_ROOT / "data" / "processed" / "arc_challenge" / "validation").exists(),
    reason="requires the prepared ARC-Challenge validation-split source under data/processed/arc_challenge/validation/",
)
def test_j2_arc_pride_runtime_calibration_n_is_15(tmp_path):
    calibration_n, n_loaded = _real_calibration_n("arc_pride.yaml", "conformance-j2", tmp_path)
    assert calibration_n == 15
    assert n_loaded == 15


# --- Synthetic calibration-pool construction tests (J3-J9) -----------------

def _synthetic_row(qid: str, n_options: int, correct_pos: int = 0) -> dict:
    texts = [f"option_{qid}_{i}" for i in range(n_options)]
    return {
        "question_id": qid,
        "subject": "synthetic",
        "question_text": f"Synthetic question {qid}?",
        "choices_json": [{"text": t, "source_index": i} for i, t in enumerate(texts)],
        "correct_option": "A",
    }


def _build_pride_runner(
    *, calibration_questions: list[dict], calibration_n: int, calibration_runs_dir,
    calibration_seed: int = 42, require_full_calibration: bool = True, modal_k: int = 4,
    run_id: str = "conformance-pride-synthetic",
) -> PriDeRunner:
    return PriDeRunner(
        backend=DummyBackend(),
        method_name="pride",
        split_name="test",
        prompt_version="v1",
        prompts_dir=PROMPTS_DIR,
        run_id=run_id,
        seed=42,
        benchmark_name="diag_bench",
        calibration_n=calibration_n,
        calibration_seed=calibration_seed,
        require_full_calibration=require_full_calibration,
        calibration_questions=calibration_questions,
        modal_k=modal_k,
        # PriDeRunner._sidecar_path() defaults calibration_runs_dir to
        # Path(".") -- always pass tmp_path explicitly so calibration
        # sidecar JSON files never land in the repo working tree (spec R5).
        calibration_runs_dir=calibration_runs_dir,
    )


def test_j3_mmlu_shaped_pool_filters_then_selects_exactly_77(tmp_path):
    pool = [_synthetic_row(f"mmlu-cal-{i:03d}", 4) for i in range(200)]
    runner = _build_pride_runner(calibration_questions=pool, calibration_n=77, calibration_runs_dir=tmp_path)
    runner._ensure_calibration()
    assert len(runner._calibration_state.estimation_question_ids) == 77
    assert set(runner._calibration_state.estimation_question_ids) <= {r["question_id"] for r in pool}


def test_j4_j5_arc_shaped_adversarial_pool_filters_before_sampling(tmp_path):
    """56-row pool: 16 eligible (4-option) + 40 ineligible (3-/5-option).
    Sampling 15 FIRST from the raw 56 would, in expectation, yield only
    ~15 * 16/56 ~= 4.3 eligible rows -- far short of 15. Filtering to the
    16 eligible rows FIRST, then sampling 15 from those, always succeeds.
    This is the exact discriminating fixture spec J5 demands."""
    eligible = [_synthetic_row(f"arc-cal-eligible-{i:02d}", 4) for i in range(16)]
    ineligible = (
        [_synthetic_row(f"arc-cal-3opt-{i:02d}", 3) for i in range(20)]
        + [_synthetic_row(f"arc-cal-5opt-{i:02d}", 5) for i in range(20)]
    )
    pool = eligible + ineligible
    runner = _build_pride_runner(calibration_questions=pool, calibration_n=15, require_full_calibration=True, calibration_runs_dir=tmp_path)
    runner._ensure_calibration()  # must NOT raise -- filter-then-sample always finds 15
    selected_ids = set(runner._calibration_state.estimation_question_ids)
    assert len(selected_ids) == 15
    eligible_ids = {r["question_id"] for r in eligible}
    assert selected_ids <= eligible_ids, (
        f"selected calibration ids include ineligible rows: {selected_ids - eligible_ids}"
    )


def test_j6_all_selected_arc_calibration_rows_are_4_choice(tmp_path):
    eligible = [_synthetic_row(f"arc-j6-eligible-{i:02d}", 4) for i in range(16)]
    ineligible = [_synthetic_row(f"arc-j6-3opt-{i:02d}", 3) for i in range(40)]
    pool = eligible + ineligible
    runner = _build_pride_runner(calibration_questions=pool, calibration_n=15, calibration_runs_dir=tmp_path)
    runner._ensure_calibration()
    selected_ids = set(runner._calibration_state.estimation_question_ids)
    by_id = {r["question_id"]: r for r in pool}
    for qid in selected_ids:
        assert len(by_id[qid]["choices_json"]) == 4, qid


def test_j7_calibration_selection_deterministic_under_seed_42(tmp_path):
    pool = [_synthetic_row(f"det-{i:03d}", 4) for i in range(50)]
    runner_a = _build_pride_runner(calibration_questions=pool, calibration_n=15, calibration_seed=42, calibration_runs_dir=tmp_path, run_id="run-a")
    runner_a._ensure_calibration()
    runner_b = _build_pride_runner(calibration_questions=pool, calibration_n=15, calibration_seed=42, calibration_runs_dir=tmp_path, run_id="run-b")
    runner_b._ensure_calibration()
    assert (
        runner_a._calibration_state.estimation_question_ids
        == runner_b._calibration_state.estimation_question_ids
    )

    runner_c = _build_pride_runner(calibration_questions=pool, calibration_n=15, calibration_seed=43, calibration_runs_dir=tmp_path, run_id="run-c")
    runner_c._ensure_calibration()
    assert (
        runner_c._calibration_state.estimation_question_ids
        != runner_a._calibration_state.estimation_question_ids
    )


def test_j8_insufficient_eligible_pool_hard_failure(tmp_path):
    pool = [_synthetic_row(f"scarce-{i}", 4) for i in range(5)]
    runner = _build_pride_runner(
        calibration_questions=pool, calibration_n=15, require_full_calibration=True, calibration_runs_dir=tmp_path,
    )
    with pytest.raises(RuntimeError, match="requires 15 eligible"):
        runner._ensure_calibration()


@pytest.mark.skipif(
    not (REPO_ROOT / "data" / "processed" / "mmlu" / "validation").exists(),
    reason="requires the prepared MMLU validation-split source under data/processed/mmlu/validation/",
)
def test_j9_calibration_ids_disjoint_from_evaluation_ids_real_data():
    config = load_config(str(PAPER_CONFIG_DIR / "mmlu_pride.yaml"))
    benchmark_cfg = config.benchmarks[0]
    method_cfg = config.methods[0]
    selection = load_benchmark_selection(benchmark_cfg, config.run.seed)
    eval_ids = set(selection.questions["question_id"].astype(str))

    preflight_records = load_preflight(
        method_cfg, benchmark_cfg, config.run.seed,
        eval_question_ids=eval_ids,
        eval_artifact_id=selection.artifact.artifact_id,
        eval_sample_identities=set(selection.sample_identities),
    )
    calib_ids = {str(r["question_id"]) for r in preflight_records}
    assert calib_ids & eval_ids == set()
    assert len(calib_ids) == 77
