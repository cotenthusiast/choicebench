# tests/scripts/test_paper_export_benchmark_questions.py

import importlib.util
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT_PATH = REPO_ROOT / "scripts" / "paper" / "export_benchmark_questions.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("export_benchmark_questions", _SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


mod = _load_module()


class TestExportBenchmarkQuestions:
    def test_writes_the_selection_questions_to_csv(self, tmp_path, monkeypatch):
        fake_questions = pd.DataFrame([
            {"question_id": "q1", "question_text": "Q1?", "choices_json": "[]", "correct_option": "A"},
            {"question_id": "q2", "question_text": "Q2?", "choices_json": "[]", "correct_option": "B"},
        ])
        fake_selection = MagicMock(questions=fake_questions)
        monkeypatch.setattr(mod, "load_benchmark_selection", lambda benchmark_cfg, seed: fake_selection)

        config_path = tmp_path / "config.yaml"
        config_path.write_text("""
experiment: {name: unit_test}
models: [{backend: dummy, model_name_or_path: d1}]
benchmarks: [{name: mmlu, split: test, n_samples: 2}]
methods: [{name: direct_mcq}]
metrics: [accuracy]
run: {seed: 7}
""")
        output = tmp_path / "exported.csv"

        n_rows = mod.export(config_path, 0, output)

        assert n_rows == 2
        result_df = pd.read_csv(output)
        assert len(result_df) == 2

    def test_injects_a_benchmark_name_column_from_the_benchmark_config(self, tmp_path, monkeypatch):
        """The raw questions DataFrame load_benchmark_selection() returns
        has NO benchmark_name column at all (confirmed against real
        prepared MMLU data) -- but run_text_extraction_rotations.py and
        run_stochasticity_repeats.py both read
        source_df["benchmark_name"].iloc[0]. The exporter must inject it,
        or both of those scripts crash the first time they're pointed at
        real exported data instead of a synthetic test fixture that
        happened to already include the column."""
        fake_questions = pd.DataFrame([
            {"question_id": "q1", "question_text": "Q1?", "choices_json": "[]", "correct_option": "A"},
        ])
        fake_selection = MagicMock(questions=fake_questions)
        monkeypatch.setattr(mod, "load_benchmark_selection", lambda benchmark_cfg, seed: fake_selection)

        config_path = tmp_path / "config.yaml"
        config_path.write_text("""
experiment: {name: unit_test}
models: [{backend: dummy, model_name_or_path: d1}]
benchmarks: [{name: mmlu, split: test, n_samples: 1}]
methods: [{name: direct_mcq}]
metrics: [accuracy]
run: {seed: 7}
""")
        output = tmp_path / "exported.csv"

        mod.export(config_path, 0, output)

        result_df = pd.read_csv(output)
        assert (result_df["benchmark_name"] == "mmlu").all()

    def test_does_not_mutate_the_selections_own_dataframe(self, tmp_path, monkeypatch):
        fake_questions = pd.DataFrame([
            {"question_id": "q1", "question_text": "Q1?", "choices_json": "[]", "correct_option": "A"},
        ])
        fake_selection = MagicMock(questions=fake_questions)
        monkeypatch.setattr(mod, "load_benchmark_selection", lambda benchmark_cfg, seed: fake_selection)

        config_path = tmp_path / "config.yaml"
        config_path.write_text("""
experiment: {name: unit_test}
models: [{backend: dummy, model_name_or_path: d1}]
benchmarks: [{name: mmlu, split: test, n_samples: 1}]
methods: [{name: direct_mcq}]
metrics: [accuracy]
run: {seed: 7}
""")
        output = tmp_path / "exported.csv"

        mod.export(config_path, 0, output)

        assert "benchmark_name" not in fake_questions.columns

    def test_uses_config_run_seed_when_none_given(self, tmp_path, monkeypatch):
        fake_questions = pd.DataFrame([{"question_id": "q1"}])
        fake_selection = MagicMock(questions=fake_questions)
        captured_seed = {}

        def _fake_load(benchmark_cfg, seed):
            captured_seed["seed"] = seed
            return fake_selection

        monkeypatch.setattr(mod, "load_benchmark_selection", _fake_load)

        config_path = tmp_path / "config.yaml"
        config_path.write_text("""
experiment: {name: unit_test}
models: [{backend: dummy, model_name_or_path: d1}]
benchmarks: [{name: mmlu, split: test, n_samples: 1}]
methods: [{name: direct_mcq}]
metrics: [accuracy]
run: {seed: 99}
""")
        mod.export(config_path, 0, tmp_path / "out.csv")

        assert captured_seed["seed"] == 99

    def test_explicit_run_seed_overrides_config_default(self, tmp_path, monkeypatch):
        fake_questions = pd.DataFrame([{"question_id": "q1"}])
        fake_selection = MagicMock(questions=fake_questions)
        captured_seed = {}

        def _fake_load(benchmark_cfg, seed):
            captured_seed["seed"] = seed
            return fake_selection

        monkeypatch.setattr(mod, "load_benchmark_selection", _fake_load)

        config_path = tmp_path / "config.yaml"
        config_path.write_text("""
experiment: {name: unit_test}
models: [{backend: dummy, model_name_or_path: d1}]
benchmarks: [{name: mmlu, split: test, n_samples: 1}]
methods: [{name: direct_mcq}]
metrics: [accuracy]
run: {seed: 99}
""")
        mod.export(config_path, 0, tmp_path / "out.csv", run_seed=5)

        assert captured_seed["seed"] == 5
