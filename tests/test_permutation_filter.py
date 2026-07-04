import json
from pathlib import Path

import pandas as pd
import pytest

from choicebench.permutation_filter import (
    PermutationFilterExclusion,
    filter_permutation_unsafe,
    match_permutation_unsafe_pattern,
    permutation_filter_path_for,
    write_permutation_filter_report,
)


def _row(question_id: str, options: list[str], correct_index: int = 0) -> dict:
    choices_json = json.dumps(
        [{"text": opt, "source_index": i} for i, opt in enumerate(options)]
    )
    return {
        "question_id": question_id,
        "subject": "abstract_algebra",
        "question_text": "irrelevant",
        "choices_json": choices_json,
        "correct_index": correct_index,
        "correct_option": ["A", "B", "C", "D"][correct_index],
        "n_choices": len(options),
    }


class TestMatchPermutationUnsafePattern:
    """Positive fixtures: each must match, since permuting them changes meaning."""

    @pytest.mark.parametrize(
        "text",
        [
            "All of the above",
            "all of the above.",
            "None of the above",
            "none of the above.",
            "Both A and C",
            "both a and c",
            "A and B",
            "b and d",
            "Neither A nor C",
            "neither b nor d.",
        ],
    )
    def test_matches_meta_referential_option(self, text):
        assert match_permutation_unsafe_pattern(text) is not None

    """Negative fixtures: ordinary option content that happens to contain
    conjunction-like substrings must NOT match — the filter matches the
    *entire* option text, not a substring buried in normal prose, to bias
    toward false negatives over dropping valid items."""

    @pytest.mark.parametrize(
        "text",
        [
            "Iron and B12 deficiency",
            "Mitochondria and ribosomes",
            "Vitamin A and vitamin B are both required",
            "The mixture of A and B produces a precipitate",
            "42",
            "A rapid increase in temperature",
        ],
    )
    def test_does_not_match_ordinary_option_text(self, text):
        assert match_permutation_unsafe_pattern(text) is None


class TestFilterPermutationUnsafe:
    def test_excludes_question_with_unsafe_option(self):
        df = pd.DataFrame(
            [
                _row("q1", ["4", "5", "6", "7"]),
                _row("q2", ["oxygen", "nitrogen", "A and B", "carbon"]),
            ]
        )

        filtered, exclusions = filter_permutation_unsafe(df)

        assert filtered["question_id"].tolist() == ["q1"]
        assert len(exclusions) == 1
        assert exclusions[0].question_id == "q2"
        assert exclusions[0].matched_pattern == "letter_and_letter"
        assert exclusions[0].offending_option_text == "A and B"

    def test_keeps_all_rows_when_none_are_unsafe(self):
        df = pd.DataFrame(
            [
                _row("q1", ["4", "5", "6", "7"]),
                _row("q2", ["oxygen", "nitrogen", "hydrogen", "carbon"]),
            ]
        )

        filtered, exclusions = filter_permutation_unsafe(df)

        assert len(filtered) == 2
        assert exclusions == []

    def test_exclusion_is_dataclass_with_as_dict(self):
        df = pd.DataFrame([_row("q1", ["none of the above", "5", "6", "7"])])

        _, exclusions = filter_permutation_unsafe(df)

        assert isinstance(exclusions[0], PermutationFilterExclusion)
        assert exclusions[0].as_dict() == {
            "question_id": "q1",
            "matched_pattern": "none_of_the_above",
            "offending_option_text": "none of the above",
        }


class TestSidecarIO:
    def test_permutation_filter_path_for_replaces_normalized_suffix(self):
        path = permutation_filter_path_for(Path("/data/mmlu_filtered_normalized.csv"))
        assert path == Path("/data/mmlu_filtered_permutation_filter.json")

    def test_write_permutation_filter_report_round_trips(self, tmp_path):
        exclusions = [
            PermutationFilterExclusion("q2", "letter_and_letter", "A and B"),
        ]
        path = write_permutation_filter_report(exclusions, n_total=2, path=tmp_path / "x_permutation_filter.json")

        report = json.loads(path.read_text())
        assert report["n_total"] == 2
        assert report["n_excluded"] == 1
        assert report["n_kept"] == 1
        assert report["exclusions"] == [
            {"question_id": "q2", "matched_pattern": "letter_and_letter", "offending_option_text": "A and B"}
        ]
