"""Section D: cyclic + baseline call graph.

Frozen invariant: baseline (direct_mcq) IS cyclic_permutation's own rotation
0 observation, re-scored from that rotation's own raw_text -- never a second
independent target-model call. See README.md's "Architecture corrections"
#1: the production target for this is
choicebench.analysis.derive_baseline_from_cyclic.derive_baseline_from_cyclic,
not methods/library/shuffled_baseline.py.
"""

from __future__ import annotations

import inspect

import pandas as pd
import pytest

from choicebench.analysis.derive_baseline_from_cyclic import derive_baseline_from_cyclic
from choicebench.methods.library.permutation import PermutationRunner

from fakes.fixtures import q_n3_primary, q_n4_paris, q_n5_wide
from fakes.spy_backend import SpyBackend


def _run_cyclic(question_row: dict, base_runner_kwargs: dict) -> tuple[SpyBackend, dict]:
    spy = SpyBackend()
    runner = PermutationRunner(
        backend=spy, method_name="cyclic_permutation", **base_runner_kwargs,
    )
    row = runner.run_one(question_row, sample_index=0)
    return spy, row


def test_derive_baseline_from_cyclic_has_no_backend_parameter():
    """Structural proof that deriving baseline cannot make a model call at
    all: the function has no backend/client parameter to take one (D2/D3)."""
    params = inspect.signature(derive_baseline_from_cyclic).parameters
    assert not any(
        "backend" in name or "client" in name for name in params
    ), f"derive_baseline_from_cyclic accepts a backend/client-shaped parameter: {list(params)}"


@pytest.mark.parametrize(
    "question_row,expected_n_calls",
    [(q_n4_paris, 4), (q_n3_primary, 3), (q_n5_wide, 5)],
    ids=["n4", "n3", "n5"],
)
def test_cyclic_call_count_and_baseline_zero_new_calls(
    question_row, expected_n_calls, base_runner_kwargs,
):
    """D1/D2/D4/D5: cyclic makes exactly N calls; deriving baseline from it
    adds zero more, and the derived row's own trace fields (prompt/raw_text/
    transport_status) are rotation 0's own recorded values, not a second,
    independently-generated observation that merely happens to agree."""
    spy, cyclic_row = _run_cyclic(question_row, base_runner_kwargs)
    assert spy.call_count == expected_n_calls

    source_df = pd.DataFrame([cyclic_row])
    baseline_df = derive_baseline_from_cyclic(source_df, method_name="direct_mcq")

    # Zero new backend interaction: no new SpyBackend calls occurred, and
    # the spy object itself was never touched again after run_one() returned.
    assert spy.call_count == expected_n_calls

    baseline_row = baseline_df.iloc[0]
    # D3: lineage by field equality against the SAME source row (rotation 0's
    # own top-level fields), not by independently recomputing and comparing.
    assert baseline_row["prompt"] == cyclic_row["prompt"]
    assert baseline_row["raw_text"] == cyclic_row["raw_text"]
    assert baseline_row["transport_status"] == cyclic_row["transport_status"]
    assert baseline_row["question_id"] == cyclic_row["question_id"]
    assert baseline_row["derived_from_run_id"] == cyclic_row["run_id"]
    assert baseline_row["derived_from_method_name"] == "cyclic_permutation"
    assert baseline_row["method_name"] == "direct_mcq"

    # Rotation 0 is the identity mapping, so the baseline's re-parse of
    # rotation 0's own raw_text against canonical options must reproduce
    # exactly what DirectMCQRunner would have parsed for the SAME raw_text --
    # here, SpyBackend's deterministic "CALL_001" marker text is unparseable
    # by the real answer parser either way, so both cyclic's own vote and the
    # derived baseline legitimately end up PARSE_MISSING; the meaningful
    # assertion is that they were derived from identical raw_text/prompt,
    # already checked above.
    assert baseline_row["raw_text"] == "CALL_001"


def test_derive_baseline_from_cyclic_rejects_non_cyclic_source():
    """D6: an artifact that isn't cyclic-permutation-shaped (no
    per_rotation_choices_json column) must fail loudly, not silently
    degrade to some default baseline."""
    bogus_df = pd.DataFrame([{"question_id": "x", "raw_text": "CALL_001"}])
    with pytest.raises(ValueError, match="per_rotation_choices_json"):
        derive_baseline_from_cyclic(bogus_df, method_name="direct_mcq")
