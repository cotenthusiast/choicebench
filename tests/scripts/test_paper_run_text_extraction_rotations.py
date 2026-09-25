# tests/scripts/test_paper_run_text_extraction_rotations.py

import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

from choicebench.config.schema import GenerationKwargsConfig, ModelConfig

REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT_PATH = REPO_ROOT / "scripts" / "paper" / "run_text_extraction_rotations.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("run_text_extraction_rotations", _SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


mod = _load_module()

_DUMMY_MODEL_CONFIG = ModelConfig(
    backend="dummy", model_name_or_path="dummy-model",
    generation_kwargs=GenerationKwargsConfig(max_new_tokens=1024, temperature=0.0),
)


def _questions_csv(tmp_path, question_ids) -> Path:
    choices_json = json.dumps([
        {"text": "FTP", "source_index": 0}, {"text": "HTTP", "source_index": 1},
        {"text": "HTTPS", "source_index": 2}, {"text": "SMTP", "source_index": 3},
    ])
    df = pd.DataFrame([
        dict(question_id=qid, benchmark_name="mmlu", subject="computer_security",
             question_text="Which protocol is primarily used to securely browse websites?",
             choices_json=choices_json, correct_option="C")
        for qid in question_ids
    ])
    path = tmp_path / "questions.csv"
    df.to_csv(path, index=False)
    return path


class TestRunTextExtractionRotations:
    def test_writes_one_row_per_question(self, tmp_path):
        source = _questions_csv(tmp_path, ["q1", "q2"])
        output = tmp_path / "out.csv"

        n_written = mod.run(source, _DUMMY_MODEL_CONFIG, "test_run", output, run_seed=42)

        assert n_written == 2
        result_df = pd.read_csv(output)
        assert len(result_df) == 2
        assert set(result_df["question_id"]) == {"q1", "q2"}

    def test_method_name_is_stamped_correctly(self, tmp_path):
        source = _questions_csv(tmp_path, ["q1"])
        output = tmp_path / "out.csv"
        mod.run(source, _DUMMY_MODEL_CONFIG, "test_run", output, run_seed=42)
        result_df = pd.read_csv(output)
        assert result_df.iloc[0]["method_name"] == "text_extraction"

    def test_per_rotation_fields_are_persisted(self, tmp_path):
        source = _questions_csv(tmp_path, ["q1"])
        output = tmp_path / "out.csv"
        mod.run(source, _DUMMY_MODEL_CONFIG, "test_run", output, run_seed=42)
        result_df = pd.read_csv(output)
        assert len(json.loads(result_df.iloc[0]["per_rotation_choices_json"])) == 4
        assert len(json.loads(result_df.iloc[0]["per_rotation_raw_text_json"])) == 4

    def test_resume_skips_already_completed_question_ids(self, tmp_path):
        source = _questions_csv(tmp_path, ["q1", "q2", "q3"])
        output = tmp_path / "out.csv"

        n_first = mod.run(source, _DUMMY_MODEL_CONFIG, "test_run", output, run_seed=42)
        assert n_first == 3

        n_second = mod.run(source, _DUMMY_MODEL_CONFIG, "test_run", output, run_seed=42)
        assert n_second == 0

        result_df = pd.read_csv(output)
        assert len(result_df) == 3
        assert sorted(result_df["question_id"]) == ["q1", "q2", "q3"]

    def test_resume_after_partial_completion_only_does_remaining_work(self, tmp_path):
        # Realistic partial state: an actual prior invocation of this same
        # script, not a hand-built fixture with a different column set --
        # a real resume's existing file always has the matching schema,
        # since only this script's own writer ever produced it.
        source = _questions_csv(tmp_path, ["q1", "q2"])
        output = tmp_path / "out.csv"

        n_first = mod.run(source, _DUMMY_MODEL_CONFIG, "test_run", output, run_seed=42)
        assert n_first == 2

        source2 = _questions_csv(tmp_path, ["q1", "q2", "q3"])
        n_written = mod.run(source2, _DUMMY_MODEL_CONFIG, "test_run", output, run_seed=42)
        assert n_written == 1

    def test_no_resume_against_existing_output_does_not_duplicate_rows(self, tmp_path):
        """Confirmed conformance gap: --no-resume previously appended fresh
        rows onto a pre-existing output without truncating it first,
        producing scientifically duplicate rows for the same questions."""
        source = _questions_csv(tmp_path, ["q1", "q2"])
        output = tmp_path / "out.csv"

        mod.run(source, _DUMMY_MODEL_CONFIG, "test_run", output, run_seed=42, resume=True)
        mod.run(source, _DUMMY_MODEL_CONFIG, "test_run", output, run_seed=42, resume=False)

        result_df = pd.read_csv(output)
        assert not result_df["question_id"].duplicated().any(), (
            f"--no-resume duplicated rows: {result_df['question_id'].value_counts().to_dict()}"
        )
        assert sorted(result_df["question_id"]) == ["q1", "q2"]

    def test_conflicting_duplicate_rows_in_existing_output_raise_on_resume(self, tmp_path):
        """Confirmed conformance gap: a corrupted/concatenated output file
        with two disagreeing rows for the same question_id was silently
        treated as "already completed" rather than surfaced as a
        conflict."""
        source = _questions_csv(tmp_path, ["q1", "q2"])
        output = tmp_path / "out.csv"
        conflicting = pd.DataFrame([
            {"question_id": "q1", "parsed_choice": "A", "method_name": "text_extraction"},
            {"question_id": "q1", "parsed_choice": "B", "method_name": "text_extraction"},
        ])
        conflicting.to_csv(output, index=False)

        with pytest.raises(ValueError, match="conflicting"):
            mod.run(source, _DUMMY_MODEL_CONFIG, "test_run", output, run_seed=42, resume=True)

    def test_missing_question_text_column_raises_clear_error(self, tmp_path):
        df = pd.DataFrame([{"question_id": "q1"}])
        path = tmp_path / "bad.csv"
        df.to_csv(path, index=False)
        with pytest.raises(ValueError, match="question_text"):
            mod.run(path, _DUMMY_MODEL_CONFIG, "test_run", tmp_path / "out.csv", run_seed=42)

    def test_a_schema_mismatched_existing_output_file_raises_clearly(self, tmp_path):
        """A stale output file from an older script version (fewer/
        different columns) must fail loudly, not silently have new-format
        rows appended into it, corrupting the CSV's per-row field count.
        Covers the sync (Dummy backend) write path."""
        source = _questions_csv(tmp_path, ["q1", "q2"])
        output = tmp_path / "out.csv"
        stale = pd.DataFrame([{"question_id": "q1", "method_name": "text_extraction"}])
        stale.to_csv(output, index=False)

        with pytest.raises(ValueError, match="schema"):
            mod.run(source, _DUMMY_MODEL_CONFIG, "test_run", output, run_seed=42)


class _FakeAsyncCapableBackend:
    """is_async_capable()=True double -- verifies run() dispatches to the
    BATCHED run_rotations_many_async() path, never the sequential one
    (which would silently submit one single-request "batch" per rotation
    call against a real BatchAPIBackend, defeating the point)."""

    def __init__(self, response_text: str = "HTTPS") -> None:
        self._response_text = response_text
        self.generate_batch_calls: list[list[str]] = []

    def is_async_capable(self) -> bool:
        return True

    @property
    def provider(self) -> str:
        return "fake"

    @property
    def model_name(self) -> str:
        return "fake-model"

    async def generate_batch(self, prompts):
        from choicebench.clients.types import ModelResponse, SUCCESS_STATUS
        self.generate_batch_calls.append(list(prompts))
        return [
            ModelResponse(
                provider="fake", model_name="fake-model", status=SUCCESS_STATUS,
                latency_seconds=0.0, raw_text=self._response_text,
            )
            for _ in prompts
        ]

    def generate(self, prompt, **kwargs):
        raise RuntimeError("must not be called for an async-capable backend")


class TestRunDispatchesToBatchedPathForAsyncCapableBackends:
    def test_uses_run_rotations_many_async_not_the_sequential_path(self, tmp_path, monkeypatch):
        fake_backend = _FakeAsyncCapableBackend()
        monkeypatch.setattr(mod, "build_backend", lambda *a, **k: fake_backend)

        source = _questions_csv(tmp_path, ["q1", "q2"])
        output = tmp_path / "out.csv"

        n_written = mod.run(source, _DUMMY_MODEL_CONFIG, "test_run", output, run_seed=42)

        assert n_written == 2
        # One generate_batch() call covering BOTH questions' rotations
        # (2 questions x 4 rotations = 8 prompts), not 8 separate calls.
        assert len(fake_backend.generate_batch_calls) == 1
        assert len(fake_backend.generate_batch_calls[0]) == 8
        result_df = pd.read_csv(output)
        assert len(result_df) == 2

    def test_resume_only_batches_the_still_pending_questions(self, tmp_path, monkeypatch):
        # Realistic partial state: a real prior invocation via this same
        # async-capable-backend path, not a hand-built fixture with a
        # different column set.
        fake_backend = _FakeAsyncCapableBackend()
        monkeypatch.setattr(mod, "build_backend", lambda *a, **k: fake_backend)

        source = _questions_csv(tmp_path, ["q1"])
        output = tmp_path / "out.csv"
        mod.run(source, _DUMMY_MODEL_CONFIG, "test_run", output, run_seed=42)
        fake_backend.generate_batch_calls.clear()

        source2 = _questions_csv(tmp_path, ["q1", "q2"])
        n_written = mod.run(source2, _DUMMY_MODEL_CONFIG, "test_run", output, run_seed=42)

        assert n_written == 1
        # Only q2's 4 rotation prompts should have been submitted.
        assert len(fake_backend.generate_batch_calls[0]) == 4

    def test_nothing_pending_makes_no_generate_batch_call(self, tmp_path, monkeypatch):
        fake_backend = _FakeAsyncCapableBackend()
        monkeypatch.setattr(mod, "build_backend", lambda *a, **k: fake_backend)

        source = _questions_csv(tmp_path, ["q1"])
        output = tmp_path / "out.csv"
        mod.run(source, _DUMMY_MODEL_CONFIG, "test_run", output, run_seed=42)
        fake_backend.generate_batch_calls.clear()

        n_written = mod.run(source, _DUMMY_MODEL_CONFIG, "test_run", output, run_seed=42)

        assert n_written == 0
        assert fake_backend.generate_batch_calls == []

    def test_a_schema_mismatched_existing_output_file_raises_clearly(self, tmp_path, monkeypatch):
        """Same guard as the sync path, but covers the batched
        _write_results() path used by async-capable backends."""
        fake_backend = _FakeAsyncCapableBackend()
        monkeypatch.setattr(mod, "build_backend", lambda *a, **k: fake_backend)

        source = _questions_csv(tmp_path, ["q1", "q2"])
        output = tmp_path / "out.csv"
        stale = pd.DataFrame([{"question_id": "q1", "method_name": "text_extraction"}])
        stale.to_csv(output, index=False)

        with pytest.raises(ValueError, match="schema"):
            mod.run(source, _DUMMY_MODEL_CONFIG, "test_run", output, run_seed=42)
