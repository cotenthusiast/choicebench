# tests/conftest.py

import atexit
import os
import shutil
import sys
import tempfile
from pathlib import Path

# Isolate the ChoiceBench workspace BEFORE any choicebench import below:
# config.paths resolves ROOT_DIR / PROCESSED_DIR / RUNS_DIR / REPORTS_DIR from
# CHOICEBENCH_HOME at import time. Without this, the suite would read and
# write the repository's own data/, runs/ and reports/ directories and depend
# on previously prepared repo state. Each pytest process (including xdist
# workers) gets its own private workspace.
_TEST_HOME = Path(tempfile.mkdtemp(prefix="choicebench-test-home-"))
os.environ["CHOICEBENCH_HOME"] = str(_TEST_HOME)
atexit.register(shutil.rmtree, _TEST_HOME, ignore_errors=True)

import json
from typing import Any

import pandas as pd
import pytest

from choicebench.benchmarks.base import make_normalized_row
from choicebench.clients.types import (
    SUCCESS_STATUS,
    FAILURE_STATUS,
    ErrorInfo,
    ModelRequest,
    ModelResponse,
    UsageInfo,
)


@pytest.fixture(scope="session", autouse=True)
def prepared_toy_dataset():
    """Prepare the deterministic toy dataset once per test session.

    Uses the official preparation entry point so tests exercise the same
    artifact layout users get from ``python scripts/prepare_toy_data.py``. The
    artifact lands in this session's private CHOICEBENCH_HOME workspace, so a
    clean clone passes ``pytest`` with no manual preparation step and nothing
    is written into the repository.
    """
    from unittest import mock

    from choicebench.cli.prepare_toy_data import main as prepare_toy_main

    with mock.patch.object(sys, "argv", ["scripts/prepare_toy_data.py"]):
        prepare_toy_main()


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
    return make_normalized_row(
        "computer_security",
        "Which protocol is primarily used to securely browse websites?",
        ["FTP", "HTTP", "HTTPS", "SMTP"],
        correct_index=2,
    )


@pytest.fixture
def sample_normalized_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        [
            make_normalized_row(
                "computer_security",
                "Which protocol is primarily used to securely browse websites?",
                ["FTP", "HTTP", "HTTPS", "SMTP"],
                correct_index=2,
            ),
            make_normalized_row(
                "high_school_physics",
                "What is the SI unit of force?",
                ["Joule", "Newton", "Watt", "Pascal"],
                correct_index=1,
            ),
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
