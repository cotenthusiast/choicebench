# tests/conftest.py

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from mcq_eval.clients.types import (
    SUCCESS_STATUS,
    FAILURE_STATUS,
    ErrorInfo,
    ModelRequest,
    ModelResponse,
    UsageInfo,
)
from mcq_eval.config.experiment import (
    MMLU_QUESTIONS_PER_SUBJECT,
    REVIEW_SUBJECTS,
    ROBUSTNESS_SUBJECTS,
)


# ---------------------------------------------------------------------------
# Raw and normalized question fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_raw_row() -> dict[str, object]:
    return {
        "subject": "computer_security",
        "question": "Which protocol is primarily used to securely browse websites?",
        "choices": '["FTP", "HTTP", "HTTPS", "SMTP"]',
        "answer": 2,
    }


@pytest.fixture
def sample_raw_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "subject": "computer_security",
                "question": "Which protocol is primarily used to securely browse websites?",
                "choices": '["FTP", "HTTP", "HTTPS", "SMTP"]',
                "answer": 2,
            },
            {
                "subject": "high_school_physics",
                "question": "What is the SI unit of force?",
                "choices": '["Joule", "Newton", "Watt", "Pascal"]',
                "answer": 1,
            },
        ]
    )


@pytest.fixture
def sample_normalized_row() -> dict[str, object]:
    return {
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


@pytest.fixture
def sample_normalized_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        [
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
    )


# ---------------------------------------------------------------------------
# Split test helpers and fixtures
# ---------------------------------------------------------------------------


def _make_normalized_rows(subjects: list[str], n_per_subject: int) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    options = ["A", "B", "C", "D"]

    for subject in subjects:
        for i in range(n_per_subject):
            correct_option = options[i % 4]
            rows.append(
                {
                    "question_id": f"{subject}__{i:02d}",
                    "subject": subject,
                    "question_text": f"{subject} question {i}",
                    "choice_a": f"{subject} choice A {i}",
                    "choice_b": f"{subject} choice B {i}",
                    "choice_c": f"{subject} choice C {i}",
                    "choice_d": f"{subject} choice D {i}",
                    "correct_option": correct_option,
                    "correct_answer_text": f"{subject} answer {correct_option} {i}",
                }
            )

    return rows


def _make_variable_count_rows(subject_counts: dict[str, int]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    options = ["A", "B", "C", "D"]

    for subject, count in subject_counts.items():
        for i in range(count):
            correct_option = options[i % 4]
            rows.append(
                {
                    "question_id": f"{subject}__{i:02d}",
                    "subject": subject,
                    "question_text": f"{subject} question {i}",
                    "choice_a": f"{subject} choice A {i}",
                    "choice_b": f"{subject} choice B {i}",
                    "choice_c": f"{subject} choice C {i}",
                    "choice_d": f"{subject} choice D {i}",
                    "correct_option": correct_option,
                    "correct_answer_text": f"{subject} answer {correct_option} {i}",
                }
            )

    return rows


@pytest.fixture
def split_test_subjects() -> list[str]:
    return [
        "computer_security",
        "high_school_physics",
        "anatomy",
    ]


@pytest.fixture
def split_test_dataframe(split_test_subjects: list[str]) -> pd.DataFrame:
    return pd.DataFrame(_make_normalized_rows(split_test_subjects, 5))


@pytest.fixture
def split_test_exclude_ids() -> set[str]:
    return {
        "computer_security__00",
        "high_school_physics__00",
    }


@pytest.fixture
def split_test_valid_ids() -> list[str]:
    return [
        "computer_security__01",
        "computer_security__02",
        "high_school_physics__01",
        "anatomy__01",
    ]


@pytest.fixture
def split_test_duplicate_ids() -> list[str]:
    return [
        "computer_security__01",
        "computer_security__01",
        "high_school_physics__01",
    ]


@pytest.fixture
def split_test_unknown_ids() -> list[str]:
    return [
        "computer_security__01",
        "missing_question_id",
    ]


@pytest.fixture
def disjoint_split_map() -> dict[str, list[str]]:
    return {
        "robustness": [
            "computer_security__01",
            "computer_security__02",
            "high_school_physics__01",
        ],
        "review": [
            "anatomy__01",
            "anatomy__02",
            "high_school_physics__02",
        ],
    }


@pytest.fixture
def overlapping_split_map() -> dict[str, list[str]]:
    return {
        "robustness": [
            "computer_security__01",
            "computer_security__02",
            "high_school_physics__01",
        ],
        "review": [
            "anatomy__01",
            "computer_security__02",
            "high_school_physics__02",
        ],
    }


@pytest.fixture
def insufficient_split_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        _make_variable_count_rows(
            {
                "computer_security": 3,
                "high_school_physics": 1,
            }
        )
    )


@pytest.fixture
def full_default_split_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        _make_normalized_rows(ROBUSTNESS_SUBJECTS, MMLU_QUESTIONS_PER_SUBJECT)
    )


@pytest.fixture
def full_default_review_subjects() -> list[str]:
    return REVIEW_SUBJECTS


@pytest.fixture
def full_default_robustness_subjects() -> list[str]:
    return ROBUSTNESS_SUBJECTS


# ---------------------------------------------------------------------------
# Client type fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def valid_request() -> ModelRequest:
    return ModelRequest(
        provider="openai",
        model_name="gpt-4.1-mini",
        payload="Question: What is 2 + 2?\nA. 3\nB. 4\nC. 5\nD. 6",
    )


@pytest.fixture
def successful_response() -> ModelResponse:
    return ModelResponse(
        provider="openai",
        model_name="gpt-4.1-mini",
        status=SUCCESS_STATUS,
        latency_seconds=0.25,
        raw_text="B",
        finish_reason="stop",
        usage=UsageInfo(
            prompt_tokens=25,
            completion_tokens=3,
            total_tokens=28,
        ),
        error=None,
        timestamp_utc="2026-03-13T21:00:00Z",
    )


@pytest.fixture
def failed_response() -> ModelResponse:
    return ModelResponse(
        provider="openai",
        model_name="gpt-4.1-mini",
        status=FAILURE_STATUS,
        latency_seconds=0.40,
        raw_text=None,
        finish_reason=None,
        usage=None,
        error=ErrorInfo(
            error_type="ProviderTimeoutError",
            message="Request timed out.",
            retryable=True,
            stage="provider_call",
        ),
        timestamp_utc=None,
    )


# ---------------------------------------------------------------------------
# Parser / scorer test fixtures
# ---------------------------------------------------------------------------


FIXTURES_DIR = Path(__file__).parent / "fixtures"
SAMPLE_MCQ_OUTPUTS_PATH = FIXTURES_DIR / "sample_mcq_outputs.json"


@pytest.fixture(scope="session")
def sample_mcq_outputs() -> dict[str, Any]:
    """
    Load sample parser/scoring cases from disk.

    Returns:
        Dictionary containing options, gold choice, and raw model-output cases.
    """
    with SAMPLE_MCQ_OUTPUTS_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


@pytest.fixture(scope="session")
def sample_options(sample_mcq_outputs: dict[str, Any]) -> dict[str, str]:
    """
    Return the sample answer options mapping used by parser tests.
    """
    return sample_mcq_outputs["options"]


@pytest.fixture(scope="session")
def sample_gold_choice(sample_mcq_outputs: dict[str, Any]) -> str:
    """
    Return the sample gold choice used by scoring tests.
    """
    return sample_mcq_outputs["gold_choice"]


@pytest.fixture(scope="session")
def sample_case_map(sample_mcq_outputs: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """
    Index sample cases by case name for convenient lookup in tests.
    """
    return {case["name"]: case for case in sample_mcq_outputs["cases"]}
