# tests/experiments/test_stage1_sources.py

import math
from pathlib import Path

import pandas as pd
import pytest

from experiments.visible_llm_matcher.stage1_sources import (
    KNOWN_3OPTION_ARC_QUESTION_IDS,
    Stage1Source,
    Stage1ValidationError,
    load_and_validate_stage1,
)

_REQUIRED_COLS = [
    "question_id",
    "question_text",
    "choice_a",
    "choice_b",
    "choice_c",
    "choice_d",
    "correct_option",
    "free_text_response",
    "subject",
]


def _make_clean_mmlu_csv(tmp_path, n=1000) -> Path:
    rows = []
    for i in range(n):
        rows.append(
            {
                "question_id": f"q{i:04d}",
                "question_text": f"Question {i}?",
                "choice_a": "a",
                "choice_b": "b",
                "choice_c": "c",
                "choice_d": "d",
                "correct_option": "A",
                "free_text_response": "a",
                "subject": "misc",
            }
        )
    path = tmp_path / "clean_mmlu.csv"
    pd.DataFrame(rows, columns=_REQUIRED_COLS).to_csv(path, index=False)
    return path


def _make_arc_csv_with_known_contamination(tmp_path, n=1000) -> Path:
    known_ids = sorted(KNOWN_3OPTION_ARC_QUESTION_IDS)
    rows = []
    for i in range(n):
        qid = known_ids[i] if i < len(known_ids) else f"arc{i:04d}"
        is_contaminated = qid in KNOWN_3OPTION_ARC_QUESTION_IDS
        rows.append(
            {
                "question_id": qid,
                "question_text": f"ARC question {i}?",
                "choice_a": "a",
                "choice_b": "b",
                "choice_c": "c",
                "choice_d": math.nan if is_contaminated else "d",
                "correct_option": "A",
                "free_text_response": "a",
                "subject": "arc_challenge",
            }
        )
    path = tmp_path / "contaminated_arc.csv"
    pd.DataFrame(rows, columns=_REQUIRED_COLS).to_csv(path, index=False)
    return path


class TestLoadAndValidateStage1Clean:
    def test_clean_1000_row_mmlu_passes(self, tmp_path):
        path = _make_clean_mmlu_csv(tmp_path)
        source = Stage1Source(path=path, model_name="m", benchmark="mmlu", repo="two-stage-prompting")

        df = load_and_validate_stage1([source])

        assert len(df) == 1000
        assert df["question_id"].is_unique
        assert not df["choice_d"].isna().any()

    def test_source_repo_and_path_are_tagged(self, tmp_path):
        path = _make_clean_mmlu_csv(tmp_path)
        source = Stage1Source(path=path, model_name="m", benchmark="mmlu", repo="two-stage-prompting")

        df = load_and_validate_stage1([source])

        assert (df["_source_repo"] == "two-stage-prompting").all()
        assert (df["_source_path"] == str(path)).all()


class TestFailClosedOnContamination:
    def test_known_contaminated_rows_without_replacement_raise(self, tmp_path):
        path = _make_arc_csv_with_known_contamination(tmp_path)
        source = Stage1Source(path=path, model_name="m", benchmark="arc_challenge", repo="two-stage-prompting")

        with pytest.raises(Stage1ValidationError, match="contaminated"):
            load_and_validate_stage1([source])

    def test_unknown_nan_row_raises_even_with_replacement_for_known_ids(self, tmp_path):
        """A NaN choice_d for a question_id NOT in the known 3-option set
        must fail closed regardless of what replacement_rows supplies for
        the known IDs — this loader never silently tolerates an unexpected
        contamination shape.
        """
        known_ids = sorted(KNOWN_3OPTION_ARC_QUESTION_IDS)
        rows = []
        for i in range(1000):
            qid = known_ids[i] if i < len(known_ids) else (f"arc{i:04d}" if i != len(known_ids) else "unexpected_nan_id")
            is_known_contaminated = qid in KNOWN_3OPTION_ARC_QUESTION_IDS
            is_unexpected = qid == "unexpected_nan_id"
            rows.append(
                {
                    "question_id": qid,
                    "question_text": f"ARC question {i}?",
                    "choice_a": "a",
                    "choice_b": "b",
                    "choice_c": "c",
                    "choice_d": math.nan if (is_known_contaminated or is_unexpected) else "d",
                    "correct_option": "A",
                    "free_text_response": "a",
                    "subject": "arc_challenge",
                }
            )
        path = tmp_path / "unexpected_contamination.csv"
        pd.DataFrame(rows, columns=_REQUIRED_COLS).to_csv(path, index=False)
        source = Stage1Source(path=path, model_name="m", benchmark="arc_challenge", repo="two-stage-prompting")

        replacement = pd.DataFrame(
            [
                {
                    "question_id": qid,
                    "question_text": "corrected",
                    "choice_a": "a",
                    "choice_b": "b",
                    "choice_c": "c",
                    "choice_d": math.nan,
                    "correct_option": "A",
                    "free_text_response": "corrected answer",
                    "subject": "arc_challenge",
                }
                for qid in known_ids
            ]
        )

        with pytest.raises(Stage1ValidationError, match="unexpected NaN"):
            load_and_validate_stage1([source], replacement_rows=replacement)

    def test_replacement_for_known_ids_resolves_contamination(self, tmp_path):
        path = _make_arc_csv_with_known_contamination(tmp_path)
        source = Stage1Source(path=path, model_name="m", benchmark="arc_challenge", repo="two-stage-prompting")

        replacement = pd.DataFrame(
            [
                {
                    "question_id": qid,
                    "question_text": "corrected question text",
                    "choice_a": "a",
                    "choice_b": "b",
                    "choice_c": "c",
                    "choice_d": math.nan,  # genuinely 3-option — allowed for known IDs
                    "correct_option": "A",
                    "free_text_response": "corrected free-text answer",
                    "subject": "arc_challenge",
                }
                for qid in sorted(KNOWN_3OPTION_ARC_QUESTION_IDS)
            ]
        )

        df = load_and_validate_stage1([source], replacement_rows=replacement)

        assert len(df) == 1000
        replaced = df[df["question_id"].isin(KNOWN_3OPTION_ARC_QUESTION_IDS)]
        assert (replaced["free_text_response"] == "corrected free-text answer").all()
        assert (replaced["_source_repo"] == "corrected_replacement").all()

    def test_bad_replacement_with_nan_for_unknown_id_raises(self, tmp_path):
        path = _make_clean_mmlu_csv(tmp_path)
        source = Stage1Source(path=path, model_name="m", benchmark="mmlu", repo="two-stage-prompting")

        bad_replacement = pd.DataFrame(
            [
                {
                    "question_id": "q0000",
                    "question_text": "bad",
                    "choice_a": "a",
                    "choice_b": "b",
                    "choice_c": "c",
                    "choice_d": math.nan,  # q0000 is NOT a known 3-option ID
                    "correct_option": "A",
                    "free_text_response": "x",
                    "subject": "misc",
                }
            ]
        )

        with pytest.raises(Stage1ValidationError, match="not in KNOWN_3OPTION_ARC_QUESTION_IDS"):
            load_and_validate_stage1([source], replacement_rows=bad_replacement)


class TestFailClosedOnIncompleteInput:
    def test_wrong_row_count_raises(self, tmp_path):
        path = _make_clean_mmlu_csv(tmp_path, n=997)
        source = Stage1Source(path=path, model_name="m", benchmark="mmlu", repo="two-stage-prompting")

        with pytest.raises(Stage1ValidationError, match="expected exactly"):
            load_and_validate_stage1([source])

    def test_duplicate_question_id_raises(self, tmp_path):
        path = _make_clean_mmlu_csv(tmp_path, n=999)
        df = pd.read_csv(path)
        dup_row = df.iloc[[0]]
        df = pd.concat([df, dup_row], ignore_index=True)
        df.to_csv(path, index=False)
        source = Stage1Source(path=path, model_name="m", benchmark="mmlu", repo="two-stage-prompting")

        with pytest.raises(Stage1ValidationError, match="duplicate question_id"):
            load_and_validate_stage1([source])

    def test_missing_free_text_response_raises(self, tmp_path):
        path = _make_clean_mmlu_csv(tmp_path)
        df = pd.read_csv(path)
        df.loc[0, "free_text_response"] = math.nan
        df.to_csv(path, index=False)
        source = Stage1Source(path=path, model_name="m", benchmark="mmlu", repo="two-stage-prompting")

        with pytest.raises(Stage1ValidationError, match="free_text_response"):
            load_and_validate_stage1([source])

    def test_missing_file_raises(self, tmp_path):
        source = Stage1Source(
            path=tmp_path / "does_not_exist.csv",
            model_name="m",
            benchmark="mmlu",
            repo="two-stage-prompting",
        )
        with pytest.raises(Stage1ValidationError, match="does not exist"):
            load_and_validate_stage1([source])

    def test_missing_required_column_raises(self, tmp_path):
        path = tmp_path / "missing_col.csv"
        pd.DataFrame([{"question_id": "q1"}]).to_csv(path, index=False)
        source = Stage1Source(path=path, model_name="m", benchmark="mmlu", repo="two-stage-prompting")

        with pytest.raises(Stage1ValidationError, match="missing required column"):
            load_and_validate_stage1([source])

    def test_no_sources_raises(self):
        with pytest.raises(Stage1ValidationError, match="no sources"):
            load_and_validate_stage1([])

    def test_mixed_model_benchmark_sources_raise(self, tmp_path):
        path_a = _make_clean_mmlu_csv(tmp_path)
        source_a = Stage1Source(path=path_a, model_name="model-a", benchmark="mmlu", repo="two-stage-prompting")
        source_b = Stage1Source(path=path_a, model_name="model-b", benchmark="mmlu", repo="two-stage-prompting")

        with pytest.raises(Stage1ValidationError, match="one \\(model, benchmark\\)"):
            load_and_validate_stage1([source_a, source_b])
