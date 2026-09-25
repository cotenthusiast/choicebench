"""Shared fixtures for the ChoiceBench conformance suite.

This suite is an executable specification of the ChoiceBench paper
protocol (see README.md). Expected scientific behavior comes from the
frozen experiment specification, not from implementation details.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
PROMPTS_ROOT = REPO_ROOT / "prompts"
SCRIPTS_PAPER_DIR = REPO_ROOT / "scripts" / "paper"

if str(SCRIPTS_PAPER_DIR) not in sys.path:
    # scripts/paper/*.py are standalone scripts with no package __init__.py;
    # this is the same sys.path-based import mechanism used elsewhere in this
    # repo's own tooling to import them as plain modules.
    sys.path.insert(0, str(SCRIPTS_PAPER_DIR))


@pytest.fixture
def prompts_root() -> Path:
    return PROMPTS_ROOT


@pytest.fixture
def base_runner_kwargs(prompts_root: Path) -> dict:
    """Common ExperimentRunner constructor kwargs for conformance tests.

    seed=42 matches the frozen experiment seed; benchmark_name/run_id are
    arbitrary but fixed strings, not read by any test for their content
    (only for consistency across calls within one test).
    """
    return dict(
        split_name="test",
        prompt_version="v1",
        prompts_dir=prompts_root,
        run_id="conformance-test-run",
        seed=42,
        benchmark_name="diag_bench",
    )


@pytest.fixture
def reasoning_runner_kwargs(prompts_root: Path) -> dict:
    return dict(
        split_name="test",
        prompt_version="v1_reasoning",
        prompts_dir=prompts_root,
        run_id="conformance-test-run",
        seed=42,
        benchmark_name="diag_bench",
    )


@pytest.fixture
def ihs_runner_kwargs(prompts_root: Path) -> dict:
    return dict(
        split_name="test",
        prompt_version="v1_ihs",
        prompts_dir=prompts_root,
        run_id="conformance-test-run",
        seed=42,
        benchmark_name="diag_bench",
    )
