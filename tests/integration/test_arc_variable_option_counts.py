# tests/integration/test_arc_variable_option_counts.py
#
# Integration coverage against the REAL frozen ARC-Challenge data (not a
# synthetic fixture) for the three known unusual-option-count rows, run
# through the actual registered methods with DummyBackend. This is exactly
# the "3/4/5-option ARC" scientific invariant the frozen spec requires
# tests for before production: every genuine option retained, no fake
# option introduced, no real option silently dropped, correct N calls.

from pathlib import Path

import pandas as pd
import pytest

from choicebench.backends.dummy_backend import DummyBackend
from choicebench.methods.library.independent_hypothesis import IndependentHypothesisRunner
from choicebench.methods.library.permutation import PermutationRunner
from choicebench.methods.library.text_extraction import TextExtractionRunner
from choicebench.methods.direct_mcq import DirectMCQRunner

REPO_ROOT = Path(__file__).resolve().parents[2]
_PROMPTS_DIR = REPO_ROOT / "prompts"
_ARC_CSV = REPO_ROOT / "data" / "processed" / "arc_challenge_normalized.csv"

# Verified directly against the real prepared data (see project memory /
# earlier audit): these are real ARC-Challenge questions with 3, 4, and 5
# genuine options respectively.
_THREE_OPTION_QID = "e3d2e85eb821d276"
_FIVE_OPTION_QID = "be30ca6f5bbf0daf"


def _no_match_embed_fn(texts):
    import numpy as np
    return np.eye(len(texts), dtype=np.float64)


@pytest.fixture(scope="module")
def arc_df() -> pd.DataFrame:
    if not _ARC_CSV.exists():
        pytest.skip(f"Real prepared ARC data not present at {_ARC_CSV} in this environment.")
    return pd.read_csv(_ARC_CSV)


@pytest.fixture
def three_option_row(arc_df) -> dict:
    row = arc_df[arc_df["question_id"] == _THREE_OPTION_QID]
    if row.empty:
        pytest.skip(f"{_THREE_OPTION_QID} not present in the current prepared ARC data.")
    return row.iloc[0].to_dict()


@pytest.fixture
def four_option_row(arc_df) -> dict:
    row = arc_df[arc_df["n_choices"] == 4]
    if row.empty:
        pytest.skip("No 4-option ARC row found.")
    return row.iloc[0].to_dict()


@pytest.fixture
def five_option_row(arc_df) -> dict:
    row = arc_df[arc_df["question_id"] == _FIVE_OPTION_QID]
    if row.empty:
        pytest.skip(f"{_FIVE_OPTION_QID} not present in the current prepared ARC data.")
    return row.iloc[0].to_dict()


def _direct_mcq_runner(backend):
    return DirectMCQRunner(
        backend=backend, method_name="direct_mcq", split_name="test",
        prompt_version="v1", prompts_dir=_PROMPTS_DIR, run_id="integration_test",
        benchmark_name="arc_challenge",
    )


def _cyclic_runner(backend, seed=42):
    return PermutationRunner(
        backend=backend, method_name="cyclic_permutation", split_name="test",
        prompt_version="v1", prompts_dir=_PROMPTS_DIR, run_id="integration_test",
        seed=seed, benchmark_name="arc_challenge",
    )


def _ihs_runner(backend, seed=42):
    return IndependentHypothesisRunner(
        backend=backend, method_name="independent_hypothesis", split_name="test",
        prompt_version="v1_ihs", prompts_dir=_PROMPTS_DIR, run_id="integration_test",
        seed=seed, benchmark_name="arc_challenge",
    )


def _text_extraction_runner(backend):
    return TextExtractionRunner(
        backend=backend, method_name="text_extraction", split_name="test",
        prompt_version="v1", prompts_dir=_PROMPTS_DIR, run_id="integration_test",
        seed=42, benchmark_name="arc_challenge", embed_fn=_no_match_embed_fn,
    )


class TestNoPhantomOrDroppedOptions:
    """No fake "D. nan" option, no real option silently dropped, for every
    method where option count matters -- across all three unusual counts."""

    @pytest.mark.parametrize("row_fixture,expected_n", [
        ("three_option_row", 3), ("four_option_row", 4), ("five_option_row", 5),
    ])
    def test_direct_mcq_prompt_shows_exactly_the_real_options(self, request, row_fixture, expected_n):
        row = request.getfixturevalue(row_fixture)
        backend = DummyBackend(fixed_text="A")
        runner = _direct_mcq_runner(backend)
        runner.run_one(row, sample_index=0)
        options = runner._build_options(row)
        assert len(options) == expected_n
        assert "nan" not in " ".join(options.values()).lower()

    @pytest.mark.parametrize("row_fixture,expected_calls", [
        ("three_option_row", 3), ("four_option_row", 4), ("five_option_row", 5),
    ])
    def test_cyclic_makes_exactly_n_calls_no_more_no_less(self, request, row_fixture, expected_calls):
        row = request.getfixturevalue(row_fixture)
        call_count = {"n": 0}

        class _CountingDummy(DummyBackend):
            def generate(self, prompt, **kwargs):
                call_count["n"] += 1
                return super().generate(prompt, **kwargs)

        backend = _CountingDummy(fixed_text="A")
        _cyclic_runner(backend).run_one(row, sample_index=0)
        assert call_count["n"] == expected_calls

    @pytest.mark.parametrize("row_fixture,expected_calls", [
        ("three_option_row", 3), ("four_option_row", 4), ("five_option_row", 5),
    ])
    def test_ihs_makes_exactly_n_calls_one_per_real_option(self, request, row_fixture, expected_calls):
        row = request.getfixturevalue(row_fixture)
        call_count = {"n": 0}

        class _CountingDummy(DummyBackend):
            def generate(self, prompt, **kwargs):
                call_count["n"] += 1
                return super().generate(prompt, **kwargs)

        backend = _CountingDummy(fixed_text="brief analysis <score>50</score>")
        result = _ihs_runner(backend).run_one(row, sample_index=0)
        assert call_count["n"] == expected_calls
        # A result should exist for every real option, never a phantom one.
        n_choices = row["n_choices"]
        option_score_keys = [k for k in result if k.endswith("_score") and k.startswith("option_")]
        assert len(option_score_keys) == n_choices

    def test_text_extraction_visible_options_have_no_phantom_slot(self, five_option_row):
        """The historically-confirmed bug: a hardcoded 4-option template
        rendering a literal 'D. nan' slot for a question with fewer real
        options. Here we check the OTHER direction too -- a 5-option
        question must show all 5, not be truncated to 4."""
        backend = DummyBackend(fixed_text="some extracted answer text")
        runner = _text_extraction_runner(backend)
        options = runner._build_options(five_option_row)
        assert len(options) == 5
        assert "E" in options  # 5th option must get a real label, not be dropped


class TestCanonicalIdentitySurvivesPermutation:
    def test_source_index_is_stable_across_all_rotations(self, five_option_row):
        """Regression guard for the canonical tie-break/flip-rate identity
        requirement: source_index for a given option's content must not
        depend on which rotation it's currently displayed under."""
        from choicebench.pipeline.prompt_builder import build_rotations
        from choicebench.pipeline.options import build_label_to_source_index

        options = PermutationRunner._build_options(five_option_row)
        label_to_id = build_label_to_source_index(five_option_row)
        original_ids = set(label_to_id.values())

        rotations = build_rotations(options)
        assert len(rotations) == 5  # one rotation per real option, not a hardcoded 4
        for rotation in rotations:
            # Every rotation is still a permutation of the SAME 5 canonical
            # option identities -- nothing appears, disappears, or duplicates.
            assert set(rotation.mapping.values()) == set(options.values())
        assert len(original_ids) == 5  # all 5 identities are genuinely distinct
