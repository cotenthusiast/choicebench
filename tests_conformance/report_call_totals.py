"""Section S2: derive production call totals from the FINAL manifests/configs,
from first principles (spec §S formulas), NOT by copying any historical
documentation total.

This is a REPORT, not a pytest-collected test -- it prints a table and exits
0. Run it directly:

    python tests_conformance/report_call_totals.py

Uncertainty is flagged explicitly wherever the "production matrix" (which
benchmark x model x method combinations actually run in the final paper) is
not fully determinable from config/paper/*.yaml alone.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from choicebench.config.schema import load_config  # noqa: E402

PAPER_CONFIG_DIR = REPO_ROOT / "config" / "paper"
MMLU_MANIFEST = REPO_ROOT / "data" / "manifests" / "mmlu_eval_v2.csv"
ARC_MANIFEST = REPO_ROOT / "data" / "manifests" / "arc_challenge_eval_v2.csv"
ARC_NORMALIZED = REPO_ROOT / "data" / "processed" / "arc_challenge_normalized.csv"

_EXPORT_ONLY_CONFIGS = {"mmlu_stochasticity_export.yaml", "arc_stochasticity_export.yaml"}

# §S formulas, restated here for the report (identical to test_call_arithmetic.py).
_SINGLE_CALL_METHODS = {
    "cyclic_permutation": lambda n: n,
    "reasoning_cyclic": lambda n: n,
    "text_extraction": lambda n: n,          # accuracy row (run_one) is 1/question;
                                              # flip-rate rotation rerun is n/question --
                                              # see PER-CONDITION NOTE below.
    "independent_hypothesis": lambda n: n,
}
_TWO_STAGE_METHODS = {"two_stage", "reasoning_two_stage"}  # 1 stage1 + 1 stage2 for accuracy (2/question, n-independent)


def _mmlu_n_choices() -> dict[str, int]:
    df = pd.read_csv(MMLU_MANIFEST, dtype={"question_id": "string"})
    return {qid: 4 for qid in df["question_id"]}  # MMLU is always 4-option


def _arc_n_choices() -> dict[str, int]:
    manifest = pd.read_csv(ARC_MANIFEST, dtype={"question_id": "string"})
    normalized = pd.read_csv(ARC_NORMALIZED, dtype={"question_id": "string"})
    joined = manifest.merge(normalized[["question_id", "n_choices"]], on="question_id", how="left")
    return dict(zip(joined["question_id"], joined["n_choices"].astype(int)))


def main() -> None:
    n_choices_by_benchmark = {"mmlu": _mmlu_n_choices(), "arc_challenge": _arc_n_choices()}

    rows = []
    uncertainty_notes: list[str] = []

    for path in sorted(PAPER_CONFIG_DIR.glob("*.yaml")):
        if path.name in _EXPORT_ONLY_CONFIGS:
            continue
        config = load_config(str(path))
        benchmark_name = config.benchmarks[0].name
        n_models = len(config.models)
        n_choices_map = n_choices_by_benchmark.get(benchmark_name)
        if n_choices_map is None:
            uncertainty_notes.append(f"{path.name}: unknown benchmark {benchmark_name!r}, skipped")
            continue
        n_questions = len(n_choices_map)
        total_n_choices = sum(n_choices_map.values())

        for method in config.methods:
            if method.name == "pride":
                # PriDe scores via score_options(), not generate() -- not a
                # "target-model generation call" in the §S sense; excluded
                # from this generation-call total (calibration rollout calls
                # are a separate, already-covered accounting in Task 9/§J).
                rows.append({"config": path.name, "benchmark": benchmark_name, "method": method.name,
                             "n_models": n_models, "n_questions": n_questions, "calls": 0,
                             "note": "logprob scoring, not generation -- excluded"})
                continue
            if method.name in _SINGLE_CALL_METHODS:
                calls = total_n_choices * n_models
                rows.append({"config": path.name, "benchmark": benchmark_name, "method": method.name,
                             "n_models": n_models, "n_questions": n_questions, "calls": calls, "note": ""})
            elif method.name in _TWO_STAGE_METHODS:
                calls = 2 * n_questions * n_models  # accuracy run: 1 stage1 + 1 stage2/question, n-independent
                rows.append({"config": path.name, "benchmark": benchmark_name, "method": method.name,
                             "n_models": n_models, "n_questions": n_questions, "calls": calls,
                             "note": "accuracy run only (1 stage1 + 1 stage2/question); flip-rate rotation rerun is separate, not counted here"})
            else:
                uncertainty_notes.append(f"{path.name}: method {method.name!r} has no known §S formula, skipped")

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    print()
    print(f"GRAND TOTAL (accuracy-run generation calls across all config/paper/*.yaml methods): {df['calls'].sum()}")
    print()
    print("NOT INCLUDED in the total above (flagged, not silently omitted):")
    print("  - direct_mcq / reasoning_mcq baselines: 0 additional calls (derived from cyclic_permutation / reasoning_cyclic)")
    print("  - semantic_matching_v1: 0 additional calls (derived from two_stage's free_text_response)")
    print("  - visible_llm_matcher: no dedicated config/paper/*.yaml found for it -- its own")
    print("    1-new-call-per-question total depends on external orchestration (joining a prior")
    print("    text_extraction result) not fully visible in config/paper/*.yaml alone. UNCERTAIN.")
    print("  - flip-rate rotation reruns (cyclic is already all-rotations; two_stage/text_extraction/")
    print("    visible_llm_matcher's *_rotations.py scripts) are a SEPARATE additional call volume,")
    print("    not part of the accuracy-run total above -- see spec §S's own per-method formulas.")
    print("  - PriDe calibration rollout calls (score_options, not generate): 77 or 15 calibration")
    print("    questions x n_choices cyclic rollouts each, per model -- see test_pride_protocol.py.")
    print("  - stochasticity: a SEPARATE frozen 100-question-per-benchmark subset (not the full")
    print("    eval manifest), 4 API models x (3 additional for baseline/reasoning_mcq, 6 additional")
    print("    for two_stage/reasoning_two_stage) -- not part of the full-manifest total above.")
    if uncertainty_notes:
        print()
        print("UNCERTAINTY NOTES:")
        for note in uncertainty_notes:
            print(f"  - {note}")


if __name__ == "__main__":
    main()
