# tests/conftest.py

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from choicebench.clients.types import (
    SUCCESS_STATUS,
    FAILURE_STATUS,
    ErrorInfo,
    ModelRequest,
    ModelResponse,
    UsageInfo,
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
