# tests/benchmarks/test_split.py

import pytest

import mcq_eval.benchmarks.split as split
from mcq_eval.benchmarks.split import (
    DuplicateSplitIdsError,
    InsufficientQuestionsError,
    SplitSizeMismatchError,
)
from mcq_eval.config.experiment import ROBUSTNESS_SPLIT_SEED


def _assert_split_ids_match_expected_subject_distribution(
    dataframe,
    split_ids,
    expected_subjects,
    per_subject,
) -> None:
    selected_rows = dataframe[dataframe["question_id"].isin(split_ids)]
    subject_counts = selected_rows["subject"].value_counts().to_dict()

    assert len(split_ids) == len(expected_subjects) * per_subject
    assert len(split_ids) == len(set(split_ids))
    assert set(subject_counts.keys()) == set(expected_subjects)
    assert all(count == per_subject for count in subject_counts.values())


class TestBuildStratifiedSplit:
    """Tests for build_robustness_split and build_review_split."""

    def test_build_robustness_split_returns_balanced_unique_ids_for_requested_subjects(
        self, split_test_dataframe, split_test_subjects,
    ) -> None:
        per_subject = 2
        split_ids = split.build_robustness_split(
            split_test_dataframe, subjects=split_test_subjects, per_subject=per_subject,
        )
        _assert_split_ids_match_expected_subject_distribution(
            dataframe=split_test_dataframe, split_ids=split_ids,
            expected_subjects=split_test_subjects, per_subject=per_subject,
        )

    def test_build_robustness_split_raises_when_subject_has_insufficient_questions(
        self, insufficient_split_dataframe,
    ) -> None:
        per_subject = 2
        subjects = ["computer_security", "high_school_physics"]
        with pytest.raises(InsufficientQuestionsError):
            split.build_robustness_split(
                insufficient_split_dataframe, subjects=subjects, per_subject=per_subject,
            )

    def test_build_review_split_returns_balanced_unique_ids_for_requested_subjects(
        self, split_test_dataframe, split_test_subjects,
    ) -> None:
        per_subject = 2
        review_ids = split.build_review_split(
            split_test_dataframe, subjects=split_test_subjects, per_subject=per_subject,
        )
        _assert_split_ids_match_expected_subject_distribution(
            dataframe=split_test_dataframe, split_ids=review_ids,
            expected_subjects=split_test_subjects, per_subject=per_subject,
        )

    def test_build_review_split_raises_when_subject_has_insufficient_questions(
        self, insufficient_split_dataframe,
    ) -> None:
        with pytest.raises(InsufficientQuestionsError):
            split.build_review_split(insufficient_split_dataframe)

    def test_build_robustness_split_excludes_excluded_ids(
        self, split_test_dataframe, split_test_subjects, split_test_exclude_ids,
    ) -> None:
        actual_ids = split.build_robustness_split(
            split_test_dataframe, split_test_subjects, per_subject=2,
            exclude_ids=split_test_exclude_ids,
        )
        intersection = set(actual_ids).intersection(split_test_exclude_ids)
        assert len(intersection) == 0

    def test_build_review_split_excludes_excluded_ids(
        self, split_test_dataframe, split_test_subjects, split_test_exclude_ids,
    ) -> None:
        actual_ids = split.build_review_split(
            split_test_dataframe, split_test_subjects, per_subject=2,
            exclude_ids=split_test_exclude_ids,
        )
        intersection = set(actual_ids).intersection(split_test_exclude_ids)
        assert len(intersection) == 0

    def test_build_robustness_split_is_deterministic_for_same_seed(
        self, split_test_dataframe, split_test_subjects,
    ) -> None:
        run1 = split.build_robustness_split(
            split_test_dataframe, split_test_subjects, per_subject=2, seed=1,
        )
        run2 = split.build_robustness_split(
            split_test_dataframe, split_test_subjects, per_subject=2, seed=1,
        )
        assert run1 == run2

    def test_build_review_split_is_deterministic_for_same_seed(
        self, split_test_dataframe, split_test_subjects,
    ) -> None:
        run1 = split.build_review_split(
            split_test_dataframe, split_test_subjects, per_subject=2, seed=1,
        )
        run2 = split.build_review_split(
            split_test_dataframe, split_test_subjects, per_subject=2, seed=1,
        )
        assert run1 == run2


class TestValidateSplitIds:
    """Tests for validate_split_ids."""

    def test_raises_for_wrong_size(self, split_test_dataframe, split_test_valid_ids) -> None:
        candidate_ids = split_test_valid_ids.copy()
        candidate_ids.pop()
        with pytest.raises(SplitSizeMismatchError):
            split.validate_split_ids(split_test_dataframe, candidate_ids, 4)

    def test_raises_for_duplicate_ids(self, split_test_dataframe, split_test_duplicate_ids) -> None:
        with pytest.raises(DuplicateSplitIdsError):
            split.validate_split_ids(split_test_dataframe, split_test_duplicate_ids, 3)

    def test_raises_for_unknown_ids(self, split_test_dataframe, split_test_unknown_ids) -> None:
        with pytest.raises(split.UnknownSplitIdsError):
            split.validate_split_ids(split_test_dataframe, split_test_unknown_ids, 2)

    def test_accepts_valid_ids(self, split_test_dataframe, split_test_valid_ids) -> None:
        split.validate_split_ids(split_test_dataframe, split_test_valid_ids, expected_size=4)


class TestAssertDisjoint:
    """Tests for assert_disjoint."""

    def test_raises_for_overlapping_splits(self, overlapping_split_map) -> None:
        with pytest.raises(split.OverlappingSplitIdsError):
            split.assert_disjoint(overlapping_split_map)


class TestBuildSplitMetadata:
    """Tests for build_split_metadata."""

    def test_returns_expected_fields(
        self, split_test_dataframe, split_test_valid_ids, split_test_subjects, split_test_exclude_ids,
    ) -> None:
        actual_metadata = split.build_split_metadata(
            split_test_dataframe, "robustness", split_test_valid_ids,
            split_test_subjects, per_subject=2, seed=ROBUSTNESS_SPLIT_SEED,
            strategy="robustness", exclude_ids=split_test_exclude_ids,
        )
        expected_metadata = {
            "split_name": "robustness",
            "subjects": ["computer_security", "high_school_physics", "anatomy"],
            "per_subject": 2,
            "seed": ROBUSTNESS_SPLIT_SEED,
            "strategy": "robustness",
            "actual_size": 4,
            "actual_subject_counts": {
                "computer_security": 2,
                "high_school_physics": 1,
                "anatomy": 1,
            },
            "eligible_pool_size": 13,
            "excluded_id_count": 2,
        }
        assert expected_metadata == actual_metadata


class TestBuildAllSplits:
    """Tests for build_all_splits."""

    def test_returns_both_split_artifacts(self, full_default_split_dataframe) -> None:
        artifacts = split.build_all_splits(full_default_split_dataframe)

        assert set(artifacts.keys()) == {"robustness", "review"}

        robustness_artifact = artifacts["robustness"]
        review_artifact = artifacts["review"]

        assert set(robustness_artifact.keys()) == {"ids", "metadata"}
        assert set(review_artifact.keys()) == {"ids", "metadata"}

        robustness_ids = robustness_artifact["ids"]
        review_ids = review_artifact["ids"]

        robustness_metadata = robustness_artifact["metadata"]
        review_metadata = review_artifact["metadata"]

        required_metadata_keys = {
            "split_name", "subjects", "per_subject", "seed", "strategy",
            "actual_size", "actual_subject_counts", "eligible_pool_size", "excluded_id_count",
        }

        assert required_metadata_keys <= set(robustness_metadata.keys())
        assert required_metadata_keys <= set(review_metadata.keys())

        robustness_expected_size = (
            len(robustness_metadata["subjects"]) * robustness_metadata["per_subject"]
        )
        review_expected_size = (
            len(review_metadata["subjects"]) * review_metadata["per_subject"]
        )

        split.validate_split_ids(full_default_split_dataframe, robustness_ids, robustness_expected_size)
        split.validate_split_ids(full_default_split_dataframe, review_ids, review_expected_size)

        split.assert_disjoint({"robustness": robustness_ids, "review": review_ids})

        assert robustness_metadata["split_name"] == "robustness"
        assert review_metadata["split_name"] == "review"
        assert robustness_metadata["actual_size"] == len(robustness_ids)
        assert review_metadata["actual_size"] == len(review_ids)
