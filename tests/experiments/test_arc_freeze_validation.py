# tests/experiments/test_arc_freeze_validation.py

import os
from pathlib import Path

import pandas as pd
import pytest

from experiments.visible_llm_matcher.arc_freeze_validation import (
    EXPECTED_ROBUSTNESS_ROW_COUNT,
    FreezeValidationError,
    KNOWN_UPSTREAM_MISSING_OPTION_E_IDS,
    load_frozen_arc_dataset,
    load_frozen_robustness_ids,
    validate_against_freeze,
)

# Same override convention as FLIP_RATE_TRACES_FREEZE_ROOT in
# experiments/flip_rate_traces/data_source.py: default to this repo's own
# laptop layout, but let Kelvin2/CI point at wherever these sibling repos
# actually live instead of hardcoding one machine's paths.
_MODEL_GENERALIZATION_REPO = Path(os.environ.get(
    "MODEL_GENERALIZATION_REPO", "/home/cotenthusiast/Projects/model-generalization",
))
_TWO_STAGE_PROMPTING_REPO = Path(os.environ.get(
    "TWO_STAGE_PROMPTING_REPO", "/home/cotenthusiast/Projects/two-stage-prompting",
))

_FREEZE_ROOT = _MODEL_GENERALIZATION_REPO / "paper_data_freeze" / "raw" / "local_model_generalization"
_REAL_SOURCES = {
    "gpt-4.1-mini (API)": (
        _TWO_STAGE_PROMPTING_REPO / "paper_results" / "eval_ready" / "paper_api_main" /
        "20260603_154649_text_extraction_gpt-4.1-mini_arc_challenge.csv"
    ),
    "gemini-2.5-flash (API)": (
        _TWO_STAGE_PROMPTING_REPO / "paper_results" / "eval_ready" / "paper_api_main" /
        "20260603_154649_text_extraction_gemini-2.5-flash_arc_challenge.csv"
    ),
    "llama-3.1-8b-instant (API)": (
        _TWO_STAGE_PROMPTING_REPO / "paper_results" / "eval_ready" / "paper_api_main" /
        "20260603_154649_text_extraction_llama-3.1-8b-instant_arc_challenge.csv"
    ),
    "Qwen-Turbo (API)": (
        _TWO_STAGE_PROMPTING_REPO / "paper_results" / "eval_ready" / "paper_api_main" /
        "20260603_154649_text_extraction_Qwen_Qwen2.5-7B-Instruct-Turbo_arc_challenge.csv"
    ),
    "Qwen2.5-7B-Instruct (local, MG)": (
        _MODEL_GENERALIZATION_REPO / "runs" / "20260617_162624" /
        "20260617_162624_text_extraction_Qwen_Qwen2.5-7B-Instruct_arc_challenge.csv"
    ),
    "Llama-3.1-8B-Instruct (local, MG)": (
        _MODEL_GENERALIZATION_REPO / "runs" / "20260617_162624" /
        "20260617_162624_text_extraction_meta-llama_Llama-3.1-8B-Instruct_arc_challenge.csv"
    ),
}

_freeze_available = _FREEZE_ROOT.is_dir() and all(p.is_file() for p in _REAL_SOURCES.values())
requires_freeze = pytest.mark.skipif(
    not _freeze_available,
    reason="Immutable Stage 1 freeze / historical reference repos not present on this machine.",
)


class TestFreezeFileIntegrity:
    @requires_freeze
    def test_frozen_arc_dataset_loads_and_checksum_matches(self):
        df = load_frozen_arc_dataset(_FREEZE_ROOT)
        assert len(df) == 1172  # full normalized pool, not yet split-filtered

    @requires_freeze
    def test_frozen_robustness_ids_loads_and_checksum_matches(self):
        ids = load_frozen_robustness_ids(_FREEZE_ROOT)
        assert len(ids) == EXPECTED_ROBUSTNESS_ROW_COUNT
        assert len(set(ids)) == EXPECTED_ROBUSTNESS_ROW_COUNT

    def test_tampered_file_fails_checksum(self, tmp_path):
        (tmp_path / "data" / "processed").mkdir(parents=True)
        bad_path = tmp_path / "data" / "processed" / "arc_challenge_normalized.csv"
        bad_path.write_text("question_id,question_text\nnot_the_real_freeze,x\n")

        with pytest.raises(FreezeValidationError, match="does not match the recorded immutable"):
            load_frozen_arc_dataset(tmp_path)


@requires_freeze
class TestAllThousandRowsMatchTheFreeze:
    """Requirement: all 1,000 reused rows match the frozen expected dataset
    — re-run against the correct authority (the immutable Stage 1 freeze),
    not ChoiceBench's current, independently re-normalized ARC file, which
    is what an earlier version of this check incorrectly used.
    """

    @pytest.mark.parametrize("source_name", sorted(_REAL_SOURCES))
    def test_source_matches_freeze_exactly(self, source_name):
        df = pd.read_csv(_REAL_SOURCES[source_name])
        assert len(df) == EXPECTED_ROBUSTNESS_ROW_COUNT

        # Should not raise: every question_id, and every compared column's
        # content, agrees with the frozen dataset for all 1000 rows.
        validate_against_freeze(df, _FREEZE_ROOT)

    def test_all_six_sources_use_identical_freeze_ids(self):
        """The 1000 question_ids are the SAME set across every model and
        every source repo (TSP API, TSP-local-copy excluded/MG-local) —
        one shared frozen question set, not six independently-sampled
        ones.
        """
        id_sets = []
        for path in _REAL_SOURCES.values():
            df = pd.read_csv(path)
            id_sets.append(frozenset(df["question_id"]))
        assert len(set(id_sets)) == 1


class TestKnownUpstreamMissingOptionE:
    def test_three_known_ids(self):
        assert KNOWN_UPSTREAM_MISSING_OPTION_E_IDS == {
            "f87cb129d9aa26c0",
            "e970a6b50d905595",
            "8aec8773c1d6b508",
        }

    @requires_freeze
    def test_these_rows_are_full_four_option_in_the_freeze(self):
        """Confirms the frozen dataset does NOT carry the upstream 5th
        option for these 3 rows — the A-D representation is what every
        historical cell (and this experiment) actually used.
        """
        frozen_df = load_frozen_arc_dataset(_FREEZE_ROOT)
        for qid in KNOWN_UPSTREAM_MISSING_OPTION_E_IDS:
            row = frozen_df.loc[qid]
            assert isinstance(row["choice_d"], str) and row["choice_d"].strip() != ""


class TestExactlyOneThousandRows:
    """Requirement: the final condition remains exactly 1,000 rows."""

    def _minimal_valid_freeze(self, tmp_path):
        """Builds a tiny synthetic freeze (1 question) so row-count
        validation can be tested in isolation from real 1000-row data.
        """
        proc_dir = tmp_path / "data" / "processed"
        split_dir = tmp_path / "data" / "splits" / "arc_challenge"
        proc_dir.mkdir(parents=True)
        split_dir.mkdir(parents=True)

        frozen_row = {
            "question_id": "q0000",
            "subject": "arc_challenge",
            "question_text": "Q?",
            "choice_a": "a",
            "choice_b": "b",
            "choice_c": "c",
            "choice_d": "d",
            "correct_option": "A",
            "correct_answer_text": "a",
        }
        pd.DataFrame([frozen_row]).to_csv(proc_dir / "arc_challenge_normalized.csv", index=False)
        (split_dir / "robustness_ids.json").write_text('["q0000"]')

        import hashlib

        monkeypatch_targets = {
            "FREEZE_ARC_NORMALIZED_SHA256": hashlib.sha256(
                (proc_dir / "arc_challenge_normalized.csv").read_bytes()
            ).hexdigest(),
            "FREEZE_ROBUSTNESS_IDS_SHA256": hashlib.sha256(
                (split_dir / "robustness_ids.json").read_bytes()
            ).hexdigest(),
        }
        return monkeypatch_targets

    def test_wrong_row_count_raises(self, tmp_path, monkeypatch):
        targets = self._minimal_valid_freeze(tmp_path)
        monkeypatch.setattr(
            "experiments.visible_llm_matcher.arc_freeze_validation.FREEZE_ARC_NORMALIZED_SHA256",
            targets["FREEZE_ARC_NORMALIZED_SHA256"],
        )
        monkeypatch.setattr(
            "experiments.visible_llm_matcher.arc_freeze_validation.FREEZE_ROBUSTNESS_IDS_SHA256",
            targets["FREEZE_ROBUSTNESS_IDS_SHA256"],
        )
        monkeypatch.setattr(
            "experiments.visible_llm_matcher.arc_freeze_validation.EXPECTED_ROBUSTNESS_ROW_COUNT",
            1,
        )

        too_few = pd.DataFrame([])
        with pytest.raises(FreezeValidationError, match="Expected exactly"):
            validate_against_freeze(too_few, tmp_path)

    def test_missing_frozen_id_raises(self, tmp_path, monkeypatch):
        targets = self._minimal_valid_freeze(tmp_path)
        monkeypatch.setattr(
            "experiments.visible_llm_matcher.arc_freeze_validation.FREEZE_ARC_NORMALIZED_SHA256",
            targets["FREEZE_ARC_NORMALIZED_SHA256"],
        )
        monkeypatch.setattr(
            "experiments.visible_llm_matcher.arc_freeze_validation.FREEZE_ROBUSTNESS_IDS_SHA256",
            targets["FREEZE_ROBUSTNESS_IDS_SHA256"],
        )
        monkeypatch.setattr(
            "experiments.visible_llm_matcher.arc_freeze_validation.EXPECTED_ROBUSTNESS_ROW_COUNT",
            1,
        )

        # A row count of 1 but for a DIFFERENT question than the frozen
        # split expects -- must still raise (missing frozen id).
        wrong_row = pd.DataFrame(
            [
                {
                    "question_id": "not_the_frozen_id",
                    "question_text": "Q?",
                    "choice_a": "a",
                    "choice_b": "b",
                    "choice_c": "c",
                    "choice_d": "d",
                    "correct_option": "A",
                }
            ]
        )
        with pytest.raises(FreezeValidationError, match="not in the frozen robustness_ids"):
            validate_against_freeze(wrong_row, tmp_path)


class TestCurrentChoicebenchIdsCannotSubstitute:
    """Requirement: current ChoiceBench dataset IDs cannot silently replace
    freeze IDs. ChoiceBench's own re-normalized arc_challenge_normalized.csv
    produces a DIFFERENT id for these same 3 genuinely-3-option questions
    (e.g. 79e8c959bbeb74a0 -> e3d2e85eb821d276 under ChoiceBench's current
    normalizer) -- that id must be rejected, not accepted as if it were the
    freeze id.
    """

    @requires_freeze
    def test_a_current_choicebench_normalized_id_is_rejected(self, monkeypatch):
        # e3d2e85eb821d276 is ChoiceBench's current-normalizer id for the
        # same question whose FREEZE id is 79e8c959bbeb74a0. It must not be
        # in the frozen robustness_ids split.
        choicebench_id = "e3d2e85eb821d276"
        frozen_ids = set(load_frozen_robustness_ids(_FREEZE_ROOT))
        assert choicebench_id not in frozen_ids

        # Row count is a separate, already-covered concern (see
        # TestExactlyOneThousandRows) -- relax it here so this test isolates
        # the id-rejection check specifically, against the real freeze.
        monkeypatch.setattr(
            "experiments.visible_llm_matcher.arc_freeze_validation.EXPECTED_ROBUSTNESS_ROW_COUNT",
            1,
        )
        row = pd.DataFrame(
            [
                {
                    "question_id": choicebench_id,
                    "question_text": "x",
                    "choice_a": "a",
                    "choice_b": "b",
                    "choice_c": "c",
                    "choice_d": "d",
                    "correct_option": "A",
                }
            ]
        )
        with pytest.raises(FreezeValidationError, match="not in the frozen robustness_ids"):
            validate_against_freeze(row, _FREEZE_ROOT)
