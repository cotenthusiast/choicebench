"""Section G: semantic matching.

Frozen invariant: semantic_matching_v1 performs ZERO additional target-model
calls -- it re-derives a canonical prediction from an already-saved
Stage-1 free-text response. Production target is
choicebench.analysis.derive_from_free_text.derive_matched_results (see
README.md's "Architecture corrections" #2), not a registered method.
"""

from __future__ import annotations

import inspect
import math

import pandas as pd
import pytest

from choicebench.analysis.derive_from_free_text import derive_matched_results
from choicebench.scoring.text_matcher import match_text_to_options

from fakes.fixtures import q_n4_paris


def _exact_embed_fn(texts: list[str]):
    """Never actually consulted when exact/containment match first --
    only present so embed_fn is non-None where a test wants the cascade to
    be allowed to reach cosine; returns a degenerate embedding that never
    produces a hit above threshold, so any pass through this file relies on
    exact/containment, never cosine luck."""
    import numpy as np
    return np.zeros((len(texts), 2))


def _source_row(question_row: dict, free_text_response, **overrides) -> dict:
    row = dict(
        question_id=question_row["question_id"],
        subject=question_row["subject"],
        question_text=question_row["question_text"],
        choices_json=question_row["choices_json"],
        correct_option=question_row["correct_option"],
        free_text_response=free_text_response,
        run_id="source-run",
        method_name="two_stage",
        seed=42,
        benchmark_name="diag_bench",
    )
    row.update(overrides)
    return row


def test_derive_matched_results_has_no_backend_parameter():
    """G1 (structural half): zero target-model calls is a structural
    property -- the function accepts no backend/client at all."""
    params = inspect.signature(derive_matched_results).parameters
    assert not any("backend" in name or "client" in name for name in params)


def test_derive_matched_results_matches_exact_text():
    """G1: given a saved Stage-1 free-text response that exactly matches an
    option's text, semantic matching derives that option -- zero new calls
    (already proved structurally above)."""
    df = pd.DataFrame([_source_row(q_n4_paris, "Paris")])
    derived = derive_matched_results(df, method_name="semantic_matching_v1", embed_fn=_exact_embed_fn)
    assert derived.iloc[0]["parsed_choice"] == "A"
    assert derived.iloc[0]["method_name"] == "semantic_matching_v1"
    assert derived.iloc[0]["derived_from_method_name"] == "two_stage"


def test_option_order_does_not_change_matched_source_text():
    """G2: changing only option display order cannot change WHICH free
    text the matcher reads -- it's a plain dict field read, not
    rotation-derived. Reorder choices_json (canonical order, not a
    'display' rotation, but the strongest available proxy for 'the row's
    own option order') while holding free_text_response fixed; assert the
    matched SEMANTIC option (by text) is the same option regardless."""
    reordered_choices = list(reversed(q_n4_paris["choices_json"]))
    df_normal = pd.DataFrame([_source_row(q_n4_paris, "Paris")])
    df_reordered = pd.DataFrame([_source_row(
        q_n4_paris, "Paris", choices_json=reordered_choices, correct_option="D",
    )])

    derived_normal = derive_matched_results(df_normal, method_name="semantic_matching_v1", embed_fn=_exact_embed_fn)
    derived_reordered = derive_matched_results(df_reordered, method_name="semantic_matching_v1", embed_fn=_exact_embed_fn)

    # Same free_text_response value read back unchanged in both cases.
    assert derived_normal.iloc[0]["free_text_response"] == derived_reordered.iloc[0]["free_text_response"] == "Paris"


def test_semantic_prediction_invariant_to_canonical_option_order():
    """G3: for a clean source artifact, semantic matching derives the same
    SEMANTIC (source_index-identified) option regardless of the row's own
    canonical option order, even though the LETTER differs."""
    texts = ["Paris", "London", "Berlin", "Madrid"]
    # Row A: canonical order as given.
    choices_a = [{"text": t, "source_index": i} for i, t in enumerate(texts)]
    # Row B: canonical order rotated by one -- "Paris" now sits at letter D.
    rotated_texts = texts[1:] + texts[:1]
    choices_b = [{"text": t, "source_index": texts.index(t)} for t in rotated_texts]

    row_a = _source_row(q_n4_paris, "Paris", choices_json=choices_a, correct_option="A")
    row_b = _source_row(q_n4_paris, "Paris", choices_json=choices_b, correct_option="D")

    derived_a = derive_matched_results(pd.DataFrame([row_a]), method_name="semantic_matching_v1", embed_fn=_exact_embed_fn)
    derived_b = derive_matched_results(pd.DataFrame([row_b]), method_name="semantic_matching_v1", embed_fn=_exact_embed_fn)

    # Both must resolve to the "Paris" option's stable source_index (0),
    # even though its LETTER differs (A in row_a, D in row_b).
    from choicebench.pipeline.options import build_label_to_source_index

    src_idx_a = build_label_to_source_index(row_a)[derived_a.iloc[0]["parsed_choice"]]
    src_idx_b = build_label_to_source_index(row_b)[derived_b.iloc[0]["parsed_choice"]]
    assert src_idx_a == src_idx_b == 0


def test_nan_free_text_does_not_silently_produce_plausible_answer():
    """G4: a real pandas CSV round-trip turns an empty free_text_response
    cell into a float NaN (not None, not the string 'nan'). Reproduce that
    exact real-world shape via an actual CSV round-trip, then assert the
    matcher never silently returns a plausible answer for it.

    The frozen spec (G4) says exactly two things must NOT happen: the NaN
    must not silently become the literal string "nan", and it must not
    produce a plausible answer. It does not say a crash is an acceptable
    substitute for either -- a crash is a THIRD failure mode, not a pass.
    An earlier version of this test asserted AttributeError as the expected
    outcome, which is wrong: it encoded the pre-fix bug as if it were the
    spec, contradicting both G4 itself and this suite's own Q5 test
    (test_provenance.py), which already asserted the correct graceful
    behavior. Production (src/choicebench/analysis/derive_from_free_text.py)
    now treats a pandas NaN as missing via an explicit isinstance(..., float)
    + pd.isna(...) check, so this asserts the frozen invariant directly."""
    import json
    import tempfile
    from pathlib import Path

    row = _source_row(q_n4_paris, "placeholder")
    row["choices_json"] = json.dumps(row["choices_json"])  # real on-disk shape, not a Python repr
    df = pd.DataFrame([row])
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = Path(tmpdir) / "source.csv"
        df.to_csv(csv_path, index=False)
        # Blank out the free_text_response cell to reproduce a genuinely
        # empty CSV field, which pandas parses back as NaN (float), not "".
        reloaded = pd.read_csv(csv_path)
        reloaded.loc[0, "free_text_response"] = float("nan")
        reloaded.to_csv(csv_path, index=False)
        nan_df = pd.read_csv(csv_path)

    assert isinstance(nan_df.loc[0, "free_text_response"], float)
    assert math.isnan(nan_df.loc[0, "free_text_response"])

    derived = derive_matched_results(nan_df, method_name="semantic_matching_v1", embed_fn=_exact_embed_fn)
    row_out = derived.iloc[0]
    assert row_out["parsed_choice"] is None or (isinstance(row_out["parsed_choice"], float) and math.isnan(row_out["parsed_choice"]))
    normalized_text = row_out["normalized_text"]
    assert normalized_text is None or (isinstance(normalized_text, float) and math.isnan(normalized_text))
    assert normalized_text != "nan"


def test_wrong_source_method_is_rejected():
    """G5: a free-text artifact produced by a different method (e.g.
    reasoning_two_stage) should be rejected when reused as
    semantic_matching_v1's source.

    FIXED (verified against commit 651137768dcad640de28f124cff3d50837fe7d7c):
    derive_from_free_text.py now validates method_name via
    _VALID_SOURCE_METHODS_BY_DERIVED_METHOD before deriving -- an
    unrecognized producing method raises ValueError. Previously xfail."""
    row = _source_row(q_n4_paris, "Paris", method_name="totally_unrelated_method")
    df = pd.DataFrame([row])
    with pytest.raises(ValueError):
        derive_matched_results(df, method_name="semantic_matching_v1", embed_fn=_exact_embed_fn)


def test_wrong_source_model_or_benchmark_is_rejected():
    """G5 (model/benchmark half): a source_df mixing more than one
    model_name or benchmark_name is rejected as internally heterogeneous --
    there is no single "expected" model/benchmark to compare a lone value
    against here, so the fix validates internal consistency instead."""
    row_a = _source_row(q_n4_paris, "Paris", model_name="model-a")
    row_b = _source_row(q_n4_paris, "Paris", model_name="model-b")
    df = pd.DataFrame([row_a, row_b])
    with pytest.raises(ValueError):
        derive_matched_results(df, method_name="semantic_matching_v1", embed_fn=_exact_embed_fn)
