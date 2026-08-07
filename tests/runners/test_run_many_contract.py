# tests/runners/test_run_many_contract.py
#
# Regression for FC-2 / MF-B: every library runner's run_many() must accept a
# pandas DataFrame (what the orchestrator always passes), not just a list of
# dicts. PriDe previously overrode run_many() to iterate the argument directly,
# so it crashed on the real call path while its own tests passed a list.

import json
from pathlib import Path

import pandas as pd
import pytest

from choicebench.backends.dummy_backend import DummyBackend
from choicebench.methods import (
    CyclicLogprobRunner,
    DirectMCQRunner,
    PermutationRunner,
    PriDeRunner,
    TwoStageRunner,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
_PROMPTS_DIR = REPO_ROOT / "prompts"


_CHOICES_JSON = json.dumps(
    [{"text": t, "source_index": i} for i, t in enumerate(["alpha", "bravo", "charlie", "delta"])]
)


def _question_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "question_id": f"q{i}",
                "subject": "demo",
                "question_text": f"Question {i}?",
                "choices_json": _CHOICES_JSON,
                "correct_option": ["A", "B", "C", "D"][i % 4],
                "correct_answer_text": ["alpha", "bravo", "charlie", "delta"][i % 4],
            }
            for i in range(3)
        ]
    )


def _build_runner(cls, tmp_path: Path):
    common = dict(
        backend=DummyBackend(),
        method_name=cls.__name__,
        split_name="robustness",
        prompt_version="v1",
        prompts_dir=_PROMPTS_DIR,
        run_id="run_many_contract",
    )
    if cls is PriDeRunner:
        common.update(
            calibration_n=0,
            calibration_seed=0,
            calibration_benchmark="mmlu",
            calibration_runs_dir=tmp_path,
            calibration_questions=[],
        )
    return cls(**common)


@pytest.mark.parametrize(
    "runner_cls",
    [
        DirectMCQRunner,
        PermutationRunner,
        TwoStageRunner,
        PriDeRunner,
        CyclicLogprobRunner,
    ],
)
def test_run_many_accepts_dataframe(runner_cls, tmp_path: Path):
    df = _question_df()
    runner = _build_runner(runner_cls, tmp_path)

    results = runner.run_many(df)

    assert isinstance(results, list)
    assert len(results) == len(df)
    for row in results:
        assert isinstance(row, dict)
        assert "question_id" in row
