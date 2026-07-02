# tests/benchmarks/test_mmlu.py

import numpy as np
import pandas as pd
import pytest

from choicebench.benchmarks.base import make_normalized_row
from choicebench.benchmarks.mmlu import build_normalized_dataframe, normalize_row
from choicebench.pipeline.options import build_option_map


class TestNormalizeRow:
    """Tests for normalize_row."""

    def test_normalize_row(self, sample_raw_row):
        result = normalize_row(sample_raw_row)
        # question_id stays stable: a 4-choice row hashes the same content as
        # the legacy choice_a..choice_d format.
        assert result["question_id"] == "4865890d7f0efae8"
        assert result["subject"] == "computer_security"
        assert result["question_text"] == (
            "Which protocol is primarily used to securely browse websites?"
        )
        assert result["correct_option"] == "C"
        assert result["correct_index"] == 2
        assert result["correct_answer_text"] == "HTTPS"
        assert result["n_choices"] == 4
        assert build_option_map(result) == {"A": "FTP", "B": "HTTP", "C": "HTTPS", "D": "SMTP"}

    def test_normalize_row_accepts_numpy_choices_array(self):
        row = {
            "subject": "abstract_algebra",
            "question": "Find the degree for the given field extension.",
            "choices": np.array(["0", "4", "2", "6"], dtype=object),
            "answer": 1,
        }

        result = normalize_row(row)

        assert build_option_map(result) == {"A": "0", "B": "4", "C": "2", "D": "6"}
        assert result["correct_option"] == "B"
        assert result["correct_answer_text"] == "4"

    def test_normalize_row_accepts_stringified_choices_list(self):
        row = {
            "subject": "abstract_algebra",
            "question": "Find the degree for the given field extension.",
            "choices": "['0', '4', '2', '6']",
            "answer": 1,
        }

        result = normalize_row(row)

        assert result["correct_option"] == "B"
        assert result["correct_answer_text"] == "4"

    def test_normalize_row_rejects_invalid_choice_count(self):
        row = {
            "subject": "abstract_algebra",
            "question": "Find the degree for the given field extension.",
            "choices": ["0", "4", "2"],
            "answer": 1,
        }

        with pytest.raises(ValueError) as exc_info:
            normalize_row(row)

        msg = str(exc_info.value)
        assert "exactly 4 choices" in msg
        assert "found 3 choices" in msg
        assert "['0', '4', '2']" in msg


class TestBuildNormalizedDataframe:
    """Tests for build_normalized_dataframe."""

    def test_build_normalize_dataframe(self, sample_raw_dataframe):
        expected_raw = [
            make_normalized_row(
                subject="computer_security",
                question_text="Which protocol is primarily used to securely browse websites?",
                choices=["FTP", "HTTP", "HTTPS", "SMTP"],
                correct_index=2,
            ),
            make_normalized_row(
                subject="high_school_physics",
                question_text="What is the SI unit of force?",
                choices=["Joule", "Newton", "Watt", "Pascal"],
                correct_index=1,
            ),
        ]
        df = pd.DataFrame(expected_raw)
        pd.testing.assert_frame_equal(build_normalized_dataframe(sample_raw_dataframe), df)
