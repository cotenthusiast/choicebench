# tests/integration/test_paper_rotation_scripts_chain.py
#
# Integration coverage for the flip-rate standalone-script chains, run
# end-to-end through the REAL runner classes and REAL script run()
# functions (DummyBackend only, no network/GPU) -- not the hand-written
# fixture CSVs each script's own unit test file uses. Those unit tests
# prove each script's own logic in isolation against an assumed schema;
# this proves the ACTUAL producer's output schema is what the ACTUAL
# consumer script expects, catching a column-name/shape drift between the
# two that isolated unit tests (each trusting its own fixture) cannot.
#
# Two chains:
#   text_extraction rotations -> visible_llm_matcher rotations
#   two_stage (real run_one) -> two_stage rotations

import importlib.util
import json
from pathlib import Path

import pandas as pd

from choicebench.config.schema import GenerationKwargsConfig, ModelConfig
from choicebench.methods.library.two_stage import TwoStageRunner

REPO_ROOT = Path(__file__).resolve().parents[2]
_PROMPTS_DIR = REPO_ROOT / "prompts"

_DUMMY_MODEL_CONFIG = ModelConfig(
    backend="dummy", model_name_or_path="dummy-model",
    generation_kwargs=GenerationKwargsConfig(max_new_tokens=16, temperature=0.0),
)


def _load_script(name: str):
    path = REPO_ROOT / "scripts" / "paper" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


text_extraction_rotations = _load_script("run_text_extraction_rotations")
visible_llm_matcher_rotations = _load_script("run_visible_llm_matcher_rotations")
two_stage_rotations = _load_script("run_two_stage_rotations")


def _questions_df() -> pd.DataFrame:
    choices_json = json.dumps([
        {"text": "FTP", "source_index": 0}, {"text": "HTTP", "source_index": 1},
        {"text": "HTTPS", "source_index": 2}, {"text": "SMTP", "source_index": 3},
    ])
    return pd.DataFrame([
        dict(question_id="q1", benchmark_name="mmlu", subject="computer_security",
             question_text="Which protocol is primarily used to securely browse websites?",
             choices_json=choices_json, correct_option="C"),
        dict(question_id="q2", benchmark_name="mmlu", subject="computer_security",
             question_text="Which protocol transfers files without encryption?",
             choices_json=choices_json, correct_option="A"),
    ])


class TestTextExtractionToVisibleLLMMatcherChain:
    def test_real_text_extraction_output_feeds_visible_llm_matcher_rotations(self, tmp_path):
        questions_csv = tmp_path / "questions.csv"
        _questions_df().to_csv(questions_csv, index=False)

        te_output = tmp_path / "text_extraction_rotations.csv"
        n_te = text_extraction_rotations.run(
            questions_csv, _DUMMY_MODEL_CONFIG, "smoke_run", te_output, run_seed=42,
        )
        assert n_te == 2

        te_df = pd.read_csv(te_output)
        # The exact columns visible_llm_matcher_rotations.run() reads.
        assert "per_rotation_raw_text_json" in te_df.columns
        assert "benchmark_name" in te_df.columns

        vlm_output = tmp_path / "visible_llm_matcher_rotations.csv"
        n_vlm = visible_llm_matcher_rotations.run(
            te_output, _DUMMY_MODEL_CONFIG, "smoke_run", vlm_output, run_seed=42,
        )
        assert n_vlm == 2

        vlm_df = pd.read_csv(vlm_output)
        assert set(vlm_df["question_id"]) == {"q1", "q2"}
        assert (vlm_df["method_name"] == "visible_llm_matcher").all()
        for per_rotation_json in vlm_df["per_rotation_choices_json"]:
            assert len(json.loads(per_rotation_json)) == 4


class TestTwoStageToTwoStageRotationsChain:
    def test_real_two_stage_output_feeds_two_stage_rotations(self, tmp_path):
        from choicebench.backends.dummy_backend import DummyBackend

        runner = TwoStageRunner(
            backend=DummyBackend(), method_name="two_stage", split_name="test",
            prompt_version="v1", prompts_dir=_PROMPTS_DIR, run_id="smoke_run",
            seed=42, benchmark_name="mmlu",
        )
        rows = _questions_df().to_dict(orient="records")
        results = [runner.run_one(row, i) for i, row in enumerate(rows)]
        two_stage_csv = tmp_path / "two_stage_result.csv"
        pd.DataFrame(results).to_csv(two_stage_csv, index=False)

        # The exact column two_stage_rotations.run() reads.
        assert "free_text_response" in pd.read_csv(two_stage_csv).columns

        output = tmp_path / "two_stage_rotations.csv"
        n_written = two_stage_rotations.run(
            two_stage_csv, _DUMMY_MODEL_CONFIG, "smoke_run", output, run_seed=42,
        )
        assert n_written == 2

        result_df = pd.read_csv(output)
        assert set(result_df["question_id"]) == {"q1", "q2"}
        assert (result_df["method_name"] == "two_stage").all()
        for per_rotation_json in result_df["per_rotation_choices_json"]:
            assert len(json.loads(per_rotation_json)) == 4
