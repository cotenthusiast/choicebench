# tests/runners/test_benchmark_name.py
#
# ExperimentRunner needs the benchmark's registry name (e.g. "mmlu",
# "arc_challenge") available on the instance, not just split_name (e.g.
# "test"/"robustness") -- the canonical tie-break protocol's benchmark_id
# input requires it, and result-row provenance should record it.

from pathlib import Path

from choicebench.methods.direct_mcq import DirectMCQRunner

from tests.runners.conftest import MockBackend

REPO_ROOT = Path(__file__).resolve().parents[2]
_PROMPTS_DIR = REPO_ROOT / "prompts"


def _make_runner(benchmark_name="mmlu"):
    return DirectMCQRunner(
        backend=MockBackend(responses=["A"]),
        method_name="baseline",
        split_name="test",
        prompt_version="v1",
        prompts_dir=_PROMPTS_DIR,
        run_id="test_run_001",
        benchmark_name=benchmark_name,
    )


def test_benchmark_name_is_stored_on_the_instance():
    runner = _make_runner(benchmark_name="arc_challenge")
    assert runner.benchmark_name == "arc_challenge"


def test_benchmark_name_defaults_to_empty_string_when_not_supplied():
    runner = DirectMCQRunner(
        backend=MockBackend(responses=["A"]),
        method_name="baseline",
        split_name="test",
        prompt_version="v1",
        prompts_dir=_PROMPTS_DIR,
        run_id="test_run_001",
    )
    assert runner.benchmark_name == ""


def test_benchmark_name_is_persisted_in_the_result_row(runner_question_row):
    runner = _make_runner(benchmark_name="mmlu")
    row = runner.run_one(runner_question_row, sample_index=0)
    assert row["benchmark_name"] == "mmlu"
