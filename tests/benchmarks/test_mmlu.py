# tests/benchmarks/test_mmlu.py

import numpy as np
import pandas as pd
import pytest

from choicebench.benchmarks.mmlu import build_normalized_dataframe, normalize_row


class TestNormalizeRow:
    """Tests for normalize_row."""

    def test_normalize_row(self, sample_raw_row):
        assert normalize_row(sample_raw_row) == {
            "question_id": "4865890d7f0efae8",
            "subject": "computer_security",
            "question_text": "Which protocol is primarily used to securely browse websites?",
            "choice_a": "FTP",
            "choice_b": "HTTP",
            "choice_c": "HTTPS",
            "choice_d": "SMTP",
            "correct_option": "C",
            "correct_answer_text": "HTTPS",
        }

    def test_normalize_row_accepts_numpy_choices_array(self):
        row = {
            "subject": "abstract_algebra",
            "question": "Find the degree for the given field extension.",
            "choices": np.array(["0", "4", "2", "6"], dtype=object),
            "answer": 1,
        }

        result = normalize_row(row)

        assert result["choice_a"] == "0"
        assert result["choice_b"] == "4"
        assert result["choice_c"] == "2"
        assert result["choice_d"] == "6"
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
            {
                "question_id": "4865890d7f0efae8",
                "subject": "computer_security",
                "question_text": "Which protocol is primarily used to securely browse websites?",
                "choice_a": "FTP",
                "choice_b": "HTTP",
                "choice_c": "HTTPS",
                "choice_d": "SMTP",
                "correct_option": "C",
                "correct_answer_text": "HTTPS",
            },
            {
                "question_id": "5e9876049bf053f9",
                "subject": "high_school_physics",
                "question_text": "What is the SI unit of force?",
                "choice_a": "Joule",
                "choice_b": "Newton",
                "choice_c": "Watt",
                "choice_d": "Pascal",
                "correct_option": "B",
                "correct_answer_text": "Newton",
            },
        ]
        df = pd.DataFrame(expected_raw)
        pd.testing.assert_frame_equal(build_normalized_dataframe(sample_raw_dataframe), df)
