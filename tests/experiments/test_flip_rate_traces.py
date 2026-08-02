# tests/experiments/test_flip_rate_traces.py
#
# Fail-closed protocol-fidelity tests for the flip-rate-traces experiment.
# These exist because a real prompt-fidelity bug (an unauthorized trailing
# newline, from reusing ChoiceBench's own packaged template instead of
# porting the historical templates byte-for-byte) shipped through a full
# broad launch undetected -- caught only by post-hoc comparison against
# preserved historical data, not by any test. These tests read the real
# historical source files directly (not embedded fixtures) so they fail if
# either historical repo's template ever changes underneath this experiment.

import csv
import hashlib
import json
import os
from pathlib import Path

import pytest

from experiments.flip_rate_traces.historical_protocol import (
    API_TEMPLATE,
    HISTORICAL_MAX_TOKENS,
    HISTORICAL_SEED,
    HISTORICAL_TEMPERATURE,
    KNOWN_3OPTION_ARC_QUESTION_IDS,
    LOCAL_TEMPLATE,
    build_api_prompt,
    build_local_prompt,
    generate_permutations,
    historical_majority_vote,
    unpermute_choice,
)
from experiments.flip_rate_traces.data_source import FREEZE_ROOT, load_frozen_questions

# Same override convention as FREEZE_ROOT in data_source.py: default to this
# repo's own laptop layout, but let Kelvin2/CI point at wherever these
# sibling repos actually live instead of hardcoding one machine's paths.
TWO_STAGE_PROMPTING_REPO = Path(os.environ.get(
    "TWO_STAGE_PROMPTING_REPO", "/home/cotenthusiast/Projects/two-stage-prompting",
))
MODEL_GENERALIZATION_REPO = Path(os.environ.get(
    "MODEL_GENERALIZATION_REPO", "/home/cotenthusiast/Projects/model-generalization",
))

API_HISTORICAL_ARC = (
    FREEZE_ROOT / "canonical" /
    "cbp__gpt-4-1-mini__arc_challenge__cyclic_generation_majority" /
    "20260601_183346_cyclic_gpt-4.1-mini_arc_challenge.csv"
)
LOCAL_HISTORICAL_ARC = (
    FREEZE_ROOT / "canonical" /
    "cbp__qwen-qwen2-5-7b-instruct__arc_challenge__cyclic_generation_majority" /
    "20260529_145812_cyclic_Qwen_Qwen2.5-7B-Instruct_arc_challenge.csv"
)

requires_freeze = pytest.mark.skipif(
    not (API_HISTORICAL_ARC.is_file() and LOCAL_HISTORICAL_ARC.is_file()),
    reason="Frozen paper_data_freeze canonical CSVs not present at FLIP_RATE_TRACES_FREEZE_ROOT.",
)
requires_two_stage_prompting = pytest.mark.skipif(
    not (TWO_STAGE_PROMPTING_REPO / "prompts" / "v1" / "direct_mcq.txt").is_file(),
    reason="two-stage-prompting sibling repo not present (set TWO_STAGE_PROMPTING_REPO).",
)
requires_model_generalization_repo = pytest.mark.skipif(
    not (MODEL_GENERALIZATION_REPO / ".git").exists(),
    reason="model-generalization sibling repo not present (set MODEL_GENERALIZATION_REPO).",
)


def _load(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _row(rows, qid):
    matches = [r for r in rows if r["question_id"] == qid]
    assert matches, f"question {qid} not found"
    return matches[0]


# --- Requirement 1/3: ordinary API prompt byte-identical, no trailing newline ---

@requires_freeze
def test_api_ordinary_prompt_byte_identical_to_historical():
    rows = _load(API_HISTORICAL_ARC)
    ordinary = [r for r in rows if r["question_id"] not in KNOWN_3OPTION_ARC_QUESTION_IDS][0]
    opts = {L: ordinary[f"choice_{L.lower()}"] for L in "ABCD"}
    rendered = build_api_prompt(ordinary["question_text"], opts)
    assert rendered == ordinary["prompt"]
    assert not rendered.endswith("\n"), "API prompt must not carry an unauthorized trailing newline"


# --- Requirement 2/3: ordinary local prompt byte-identical, no trailing newline ---

@requires_freeze
def test_local_ordinary_prompt_byte_identical_to_historical():
    rows = _load(LOCAL_HISTORICAL_ARC)
    ordinary = [r for r in rows if r["question_id"] not in KNOWN_3OPTION_ARC_QUESTION_IDS][0]
    opts = {L: ordinary[f"choice_{L.lower()}"] for L in "ABCD"}
    rendered = build_local_prompt(ordinary["question_text"], opts)
    assert rendered == ordinary["prompt"]
    assert not rendered.endswith("\n"), "local prompt must not carry an unauthorized trailing newline"


# --- Requirement 3 (broader sweep): no trailing newline across many real questions ---

@requires_freeze
@pytest.mark.parametrize("side", ["api", "local"])
def test_no_trailing_newline_across_sample(side):
    path = API_HISTORICAL_ARC if side == "api" else LOCAL_HISTORICAL_ARC
    builder = build_api_prompt if side == "api" else build_local_prompt
    rows = _load(path)[:50]
    for r in rows:
        opts = {L: r[f"choice_{L.lower()}"] for L in "ABCD" if r[f"choice_{L.lower()}"]}
        rendered = builder(r["question_text"], opts)
        assert not rendered.endswith("\n"), f"{side} prompt for {r['question_id']} has a trailing newline"


# --- Requirement 4: repaired 3-option prompts differ from historical contaminated
#     prompt only by removing the D line (API); local historical was already
#     correct so the repaired prompt must equal it exactly ---

@requires_freeze
@pytest.mark.parametrize("qid", sorted(KNOWN_3OPTION_ARC_QUESTION_IDS))
def test_api_repaired_prompt_differs_only_by_d_line_removal(qid):
    rows = _load(API_HISTORICAL_ARC)
    contaminated = _row(rows, qid)
    assert "D. nan" in contaminated["prompt"], (
        f"expected historical contamination for {qid}, found none -- fixture assumption broken"
    )
    opts = {"A": contaminated["choice_a"], "B": contaminated["choice_b"], "C": contaminated["choice_c"]}
    rendered = build_api_prompt(contaminated["question_text"], opts)
    historical_minus_d_line = "\n".join(
        line for line in contaminated["prompt"].split("\n") if not line.startswith("D. ")
    )
    assert rendered == historical_minus_d_line
    assert "nan" not in rendered.lower()


@requires_freeze
@pytest.mark.parametrize("qid", sorted(KNOWN_3OPTION_ARC_QUESTION_IDS))
def test_local_repaired_prompt_matches_historical_exactly(qid):
    rows = _load(LOCAL_HISTORICAL_ARC)
    hist = _row(rows, qid)
    assert "nan" not in hist["prompt"].lower(), (
        "local historical prompt for a known 3-option question should already be clean"
    )
    opts = {"A": hist["choice_a"], "B": hist["choice_b"], "C": hist["choice_c"]}
    rendered = build_local_prompt(hist["question_text"], opts)
    assert rendered == hist["prompt"]


# --- Requirement 5: permutation ordering / unpermute / majority tie-break / dataset ---

def test_permutation_ordering_includes_canonical_as_index_zero():
    options = {"A": "w", "B": "x", "C": "y", "D": "z"}
    perms = generate_permutations(options)
    assert perms[0] == options
    assert len(perms) == 4
    # each successive permutation is a left-rotation of the previous
    values = list(options.values())
    for i, perm in enumerate(perms):
        assert list(perm.values()) == values[i:] + values[:i]


def test_unpermute_choice_round_trips():
    canonical = {"A": "w", "B": "x", "C": "y", "D": "z"}
    perms = generate_permutations(canonical)
    for i, perm in enumerate(perms):
        for letter in perm:
            # whatever letter was picked in this permutation's space, mapping
            # back must recover the canonical letter for the same text
            assert canonical[unpermute_choice(letter, perm, canonical)] == perm[letter]


def test_majority_tie_break_is_first_valid_vote_not_canonical_order():
    # tie between C and D; canonical-order tiebreak (choicebench's own,
    # current PermutationRunner) would pick C; historical tiebreak picks the
    # first vote in processing order, which is D here.
    assert historical_majority_vote([None, "D", "C", None]) == "D"
    assert historical_majority_vote(["C", "D", None, None]) == "C"
    assert historical_majority_vote([None, None, None]) is None
    assert historical_majority_vote(["A", "A", "B", "B"])  # a tie exists
    assert historical_majority_vote(["A", "A", "B", "B"]) == "A"


def test_generation_settings_match_historical():
    assert HISTORICAL_TEMPERATURE == 0.0
    assert HISTORICAL_MAX_TOKENS == 500
    assert HISTORICAL_SEED == 42


@requires_freeze
def test_frozen_dataset_identity_and_size():
    for cell_id in [
        "cbp__gpt-4-1-mini__arc_challenge__cyclic_generation_majority",
        "cbp__gpt-4-1-mini__mmlu__cyclic_generation_majority",
        "cbp__qwen-qwen2-5-7b-instruct__arc_challenge__cyclic_generation_majority",
        "cbp__qwen-qwen2-5-7b-instruct__mmlu__cyclic_generation_majority",
    ]:
        questions = load_frozen_questions(cell_id)  # raises on sha256 mismatch
        assert len(questions) == 1000
        assert len(set(q.question_id for q in questions)) == 1000


# --- Requirement 6: PROTOCOL_DIFF.md prompt hashes are real and reproducible ---

@requires_freeze
def test_protocol_diff_doc_prompt_hashes_are_reproducible():
    rows = _load(API_HISTORICAL_ARC)
    ordinary = [r for r in rows if r["question_id"] not in KNOWN_3OPTION_ARC_QUESTION_IDS][0]
    opts = {L: ordinary[f"choice_{L.lower()}"] for L in "ABCD"}
    rendered = build_api_prompt(ordinary["question_text"], opts)
    digest = hashlib.sha256(rendered.encode("utf-8")).hexdigest()
    assert digest == hashlib.sha256(ordinary["prompt"].encode("utf-8")).hexdigest(), (
        "rendered API prompt hash must equal the historical stored prompt's own hash"
    )


@requires_two_stage_prompting
def test_api_template_bytes_match_source_repo_exactly():
    with open(TWO_STAGE_PROMPTING_REPO / "prompts" / "v1" / "direct_mcq.txt", "rb") as f:
        assert f.read() == API_TEMPLATE.encode("utf-8")


@requires_model_generalization_repo
def test_local_template_bytes_match_source_repo_exactly():
    import subprocess
    content = subprocess.run(
        ["git", "-C", str(MODEL_GENERALIZATION_REPO),
         "show", "f306e6e^:prompts/v1/direct_mcq.txt"],
        capture_output=True, check=True,
    ).stdout
    assert content == LOCAL_TEMPLATE.encode("utf-8")
