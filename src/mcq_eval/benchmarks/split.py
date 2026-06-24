# src/mcq_eval/benchmarks/split.py

from typing import Any

import numpy as np
import pandas as pd

from mcq_eval.config.experiment import (
    REVIEW_QUESTIONS_PER_SUBJECT,
    REVIEW_SPLIT_SEED,
    REVIEW_SUBJECTS,
    REVIEW_TOTAL_QUESTIONS,
    REVIEW_TRACK_NAME,
    ROBUSTNESS_QUESTIONS_PER_SUBJECT,
    ROBUSTNESS_SPLIT_SEED,
    ROBUSTNESS_SUBJECTS,
    ROBUSTNESS_TOTAL_QUESTIONS,
    ROBUSTNESS_TRACK_NAME,
)


class InsufficientQuestionsError(ValueError):
    """Raised when a split cannot sample enough eligible questions for a subject."""
    pass


class SplitSizeMismatchError(ValueError):
    """Raised when a split does not contain the expected number of question IDs."""
    pass


class DuplicateSplitIdsError(ValueError):
    """Raised when a split contains duplicate question IDs."""
    pass


class UnknownSplitIdsError(ValueError):
    """Raised when a split contains question IDs not present in the dataset."""
    pass


class OverlappingSplitIdsError(ValueError):
    """Raised when two or more splits contain the same question IDs."""
    pass


def _build_stratified_split(
    df: pd.DataFrame,
    subjects: list[str],
    per_subject: int,
    seed: int,
    exclude_ids: set[str] | None = None,
) -> list[str]:
    """
    Build a stratified split by sampling a fixed number of questions per subject.

    The input DataFrame is expected to be the normalized canonical question table.
    This function filters the dataset to the requested subjects, removes any
    question IDs already present in ``exclude_ids``, and samples
    ``per_subject`` questions without replacement from each subject.

    Subject rows are sorted by ``question_id`` before sampling so that the
    output remains stable even if the upstream DataFrame row order changes.
    Sampling is deterministic for a given seed.

    Args:
        df: Normalized question DataFrame containing at least ``subject`` and
            ``question_id`` columns.
        subjects: Subjects to include in the split.
        per_subject: Number of questions to sample from each subject.
        seed: Random seed used for deterministic sampling.
        exclude_ids: Question IDs that must be excluded from the split.

    Returns:
        A flat list of sampled question IDs.

    Raises:
        InsufficientQuestionsError: If any subject has fewer than
            ``per_subject`` eligible questions after filtering.
    """
    rng = np.random.default_rng(seed)
    selected_ids: list[str] = []
    if exclude_ids is None:
        exclude_ids = set()
    df = df[df["subject"].isin(subjects)]
    df = df[~df["question_id"].isin(exclude_ids)]
    for subject in subjects:
        subject_rows = df[df["subject"] == subject]
        subject_rows = subject_rows.sort_values("question_id")

        if len(subject_rows) < per_subject:
            raise InsufficientQuestionsError(
                f"Subject '{subject}' has only {len(subject_rows)} eligible questions, "
                f"but {per_subject} are required."
            )

        sampled_rows = subject_rows.sample(
            n=per_subject,
            random_state=rng,
            replace=False,
        )

        selected_ids.extend(sampled_rows["question_id"].tolist())

    return selected_ids


def build_robustness_split(
    df: pd.DataFrame,
    subjects: list[str] = ROBUSTNESS_SUBJECTS,
    per_subject: int = ROBUSTNESS_QUESTIONS_PER_SUBJECT,
    seed: int = ROBUSTNESS_SPLIT_SEED,
    exclude_ids: set[str] | None = None,
) -> list[str]:
    """
    Build the robustness split using the configured robustness defaults.
    """
    return _build_stratified_split(df, subjects, per_subject, seed, exclude_ids)


def build_review_split(
    df: pd.DataFrame,
    subjects: list[str] = REVIEW_SUBJECTS,
    per_subject: int = REVIEW_QUESTIONS_PER_SUBJECT,
    seed: int = REVIEW_SPLIT_SEED,
    exclude_ids: set[str] | None = None,
) -> list[str]:
    """
    Build the review split using the configured review defaults.
    """
    return _build_stratified_split(df, subjects, per_subject, seed, exclude_ids)


def validate_split_ids(
    df: pd.DataFrame,
    split_ids: list[str],
    expected_size: int,
) -> None:
    """
    Validate that a split contains only valid question IDs and matches its expected size.

    This function performs three checks against the normalized canonical dataset:

    1. the split contains exactly ``expected_size`` question IDs,
    2. the split does not contain duplicate IDs,
    3. every split ID exists in the dataset's ``question_id`` column.

    Args:
        df: Normalized question DataFrame containing a ``question_id`` column.
        split_ids: Question IDs selected for one split.
        expected_size: Required number of IDs for the split.

    Raises:
        SplitSizeMismatchError: If the split size does not match ``expected_size``.
        DuplicateSplitIdsError: If the split contains duplicate question IDs.
        UnknownSplitIdsError: If the split contains IDs not present in the dataset.
    """
    actual_size = len(split_ids)
    if actual_size != expected_size:
        raise SplitSizeMismatchError(
            f"Split has {actual_size} IDs, but expected {expected_size}."
        )

    if len(split_ids) != len(set(split_ids)):
        duplicate_ids = sorted([qid for qid in set(split_ids) if split_ids.count(qid) > 1])
        raise DuplicateSplitIdsError(
            f"Split contains duplicate question IDs: {duplicate_ids}."
        )

    dataset_ids = set(df["question_id"])
    split_id_set = set(split_ids)

    if not split_id_set.issubset(dataset_ids):
        missing_ids = sorted(split_id_set - dataset_ids)
        raise UnknownSplitIdsError(
            f"Split contains question IDs not present in the dataset: {missing_ids}."
        )


def assert_disjoint(
    split_map: dict[str, list[str]],
) -> None:
    """
    Assert that all splits are pairwise disjoint.

    This function checks that no question ID appears in more than one split.
    Each entry in ``split_map`` maps a split name to its list of question IDs.
    If any two splits share one or more IDs, an exception is raised.

    Args:
        split_map: Mapping from split name to the list of question IDs in that split.

    Raises:
        OverlappingSplitIdsError: If any two splits contain overlapping question IDs.
    """
    split_items = list(split_map.items())
    for i in range(len(split_items)):
        current_row = set(split_items[i][1])
        for j in range(i + 1, len(split_items)):
            next_row = set(split_items[j][1])
            intersection = current_row.intersection(next_row)
            if intersection:
                raise OverlappingSplitIdsError(
                    f"split {split_items[i][0]} and split {split_items[j][0]} are overlapping."
                    f"Overlapping question IDS: {intersection}"
                )


def build_split_metadata(
    df: pd.DataFrame,
    split_name: str,
    split_ids: list[str],
    subjects: list[str],
    per_subject: int,
    seed: int,
    strategy: str,
    exclude_ids: set[str] | None = None,
) -> dict[str, Any]:
    """
    Build metadata for a generated split.

    This function assembles a reproducible metadata record for one split using the
    full normalized canonical dataset and the selected split IDs. It records both
    the intended split configuration and derived facts about the realized split,
    including its actual size, per-subject counts, and the size of the eligible
    candidate pool before sampling.

    Args:
        df: Normalized question DataFrame containing at least ``question_id`` and
            ``subject`` columns.
        split_name: Name of the split, such as ``robustness`` or ``review``.
        split_ids: Question IDs selected for this split.
        subjects: Subject list used to define the split's allowed scope.
        per_subject: Intended number of sampled questions per subject.
        seed: Random seed used during split generation.
        strategy: Human-readable description of the sampling strategy.
        exclude_ids: Question IDs excluded from the eligible pool because they were
            already used by earlier splits.

    Returns:
        A metadata dictionary describing the split configuration and its realized
        sampled result.
    """
    if exclude_ids is None:
        exclude_ids = set()
    df_selected_rows = df[df["question_id"].isin(split_ids)]
    actual_size = len(df_selected_rows)
    actual_subject_counts = df_selected_rows.groupby("subject")["question_id"].count().to_dict()
    df_eligible_pool = df[~df["question_id"].isin(exclude_ids) & df["subject"].isin(subjects)]
    eligible_pool_size = len(df_eligible_pool)
    excluded_id_count = len(exclude_ids)
    split_metadata = {
        "split_name": split_name,
        "subjects": subjects,
        "per_subject": per_subject,
        "seed": seed,
        "strategy": strategy,
        "actual_size": actual_size,
        "actual_subject_counts": actual_subject_counts,
        "eligible_pool_size": eligible_pool_size,
        "excluded_id_count": excluded_id_count,
    }
    return split_metadata


def build_all_splits(
    df: pd.DataFrame,
    robustness_seed: int = ROBUSTNESS_SPLIT_SEED,
    review_seed: int = REVIEW_SPLIT_SEED,
) -> dict[str, dict[str, Any]]:
    """
    Build, validate, and assemble all split artifacts for the experiment.

    This wrapper coordinates the full split-generation process using the
    normalized canonical question DataFrame. It builds the robustness split
    first, then builds the review split while excluding IDs already assigned
    to robustness. Each split is validated against the dataset, checked for
    cross-split overlap, and assembled into a split artifact containing the
    selected question IDs and the corresponding metadata.

    Args:
        df: Normalized question DataFrame containing at least ``question_id``
            and ``subject`` columns.
        robustness_seed: Random seed used to generate the robustness split.
        review_seed: Random seed used to generate the review split.

    Returns:
        A dictionary mapping split names to split artifacts. Each split
        artifact has the form::

            {
                "ids": list[str],
                "metadata": dict[str, Any],
            }

    Raises:
        SplitSizeMismatchError: If a generated split does not match its
            expected size.
        DuplicateSplitIdsError: If a generated split contains duplicate IDs.
        UnknownSplitIdsError: If a generated split contains IDs not present
            in the dataset.
        OverlappingSplitIdsError: If the generated splits are not disjoint.
        InsufficientQuestionsError: If a split cannot sample enough eligible
            questions for one or more subjects.
    """
    exclude_ids = set()

    robustness_exclude_ids = exclude_ids.copy()
    robustness_split_ids = build_robustness_split(
        df,
        seed=robustness_seed,
        exclude_ids=robustness_exclude_ids,
    )
    validate_split_ids(
        df,
        robustness_split_ids,
        expected_size=ROBUSTNESS_TOTAL_QUESTIONS,
    )
    robustness_metadata = build_split_metadata(
        df,
        split_name=ROBUSTNESS_TRACK_NAME,
        split_ids=robustness_split_ids,
        subjects=ROBUSTNESS_SUBJECTS,
        per_subject=ROBUSTNESS_QUESTIONS_PER_SUBJECT,
        seed=robustness_seed,
        strategy="stratified_by_subject",
        exclude_ids=robustness_exclude_ids,
    )
    exclude_ids = exclude_ids.union(robustness_split_ids)

    return_dict = {}
    return_dict[ROBUSTNESS_TRACK_NAME] = {
        "ids": robustness_split_ids,
        "metadata": robustness_metadata
    }

    review_exclude_ids = exclude_ids.copy()
    review_split_ids = build_review_split(
        df,
        seed=review_seed,
        exclude_ids=review_exclude_ids,
    )
    validate_split_ids(
        df,
        review_split_ids,
        expected_size=REVIEW_TOTAL_QUESTIONS,
    )
    review_metadata = build_split_metadata(
        df,
        split_name=REVIEW_TRACK_NAME,
        split_ids=review_split_ids,
        subjects=REVIEW_SUBJECTS,
        per_subject=REVIEW_QUESTIONS_PER_SUBJECT,
        seed=review_seed,
        strategy="stratified_by_subject",
        exclude_ids=review_exclude_ids,
    )

    assert_disjoint(
        {
            ROBUSTNESS_TRACK_NAME: robustness_split_ids,
            REVIEW_TRACK_NAME: review_split_ids,
        }
    )

    return_dict[REVIEW_TRACK_NAME] = {
        "ids": review_split_ids,
        "metadata": review_metadata
    }

    return return_dict
