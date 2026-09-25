# tests/scripts/test_paper_run_stochasticity_repeats.py

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from choicebench.clients.types import ModelResponse, SUCCESS_STATUS
from choicebench.config.schema import GenerationKwargsConfig, ModelConfig

REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT_PATH = REPO_ROOT / "scripts" / "paper" / "run_stochasticity_repeats.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("run_stochasticity_repeats", _SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


mod = _load_module()

_MODEL_CONFIG = ModelConfig(
    backend="api", provider="openai", model_name_or_path="gpt-4.1-mini",
    generation_kwargs=GenerationKwargsConfig(max_new_tokens=1024, temperature=0.0),
)


class _FakeAsyncBackend:
    """is_async_capable()=True double: generate_batch() returns a fixed
    answer for every prompt and records exactly how it was called, so
    tests can assert on call shape (one call per repetition, not one per
    question) without needing a real provider."""

    def __init__(self, response_text: str = "C") -> None:
        self._response_text = response_text
        self.generate_batch_calls: list[list[str]] = []

    def is_async_capable(self) -> bool:
        return True

    @property
    def provider(self) -> str:
        return "openai"

    @property
    def model_name(self) -> str:
        return "gpt-4.1-mini"

    async def generate_batch(self, prompts):
        self.generate_batch_calls.append(list(prompts))
        return [
            ModelResponse(
                provider="openai", model_name="gpt-4.1-mini", status=SUCCESS_STATUS,
                latency_seconds=0.0, raw_text=self._response_text,
            )
            for _ in prompts
        ]

    def generate(self, prompt, **kwargs):
        raise RuntimeError("must not be called for an async-capable backend")


def _questions_csv(tmp_path, question_ids, n_options=4) -> Path:
    options = [
        {"text": "FTP", "source_index": 0}, {"text": "HTTP", "source_index": 1},
        {"text": "HTTPS", "source_index": 2}, {"text": "SMTP", "source_index": 3},
    ][:n_options]
    choices_json = json.dumps(options)
    df = pd.DataFrame([
        dict(question_id=qid, benchmark_name="mmlu", subject="computer_security",
             question_text="Which protocol is primarily used to securely browse websites?",
             choices_json=choices_json, correct_option="C")
        for qid in question_ids
    ])
    path = tmp_path / "questions.csv"
    df.to_csv(path, index=False)
    return path


class TestRunStochasticityRepeatsBaseline:
    """direct_mcq / reasoning_mcq: single-stage, batch-safe."""

    def test_writes_n_repetitions_times_n_questions_rows(self, tmp_path, monkeypatch):
        backend = _FakeAsyncBackend()
        monkeypatch.setattr(mod, "build_backend", lambda *a, **k: backend)
        source = _questions_csv(tmp_path, ["q1", "q2"])
        output = tmp_path / "out.csv"

        n_written = mod.run(
            source, _MODEL_CONFIG, "test_run", output, method_name="direct_mcq",
            prompt_version="v1", n_repetitions=4, run_seed=42,
        )

        assert n_written == 8  # 2 questions x 4 repetitions
        result_df = pd.read_csv(output)
        assert len(result_df) == 8
        assert sorted(result_df["repetition_index"].unique().tolist()) == [0, 1, 2, 3]
        assert set(result_df["question_id"]) == {"q1", "q2"}

    def test_makes_one_generate_batch_call_per_repetition(self, tmp_path, monkeypatch):
        backend = _FakeAsyncBackend()
        monkeypatch.setattr(mod, "build_backend", lambda *a, **k: backend)
        source = _questions_csv(tmp_path, ["q1", "q2", "q3"])
        output = tmp_path / "out.csv"

        mod.run(
            source, _MODEL_CONFIG, "test_run", output, method_name="direct_mcq",
            prompt_version="v1", n_repetitions=4, run_seed=42,
        )

        # 4 repetitions -> 4 generate_batch() calls, each covering all 3
        # questions at once (not one call per question).
        assert len(backend.generate_batch_calls) == 4
        assert all(len(call) == 3 for call in backend.generate_batch_calls)

    def test_each_repetition_gets_a_distinct_model_identity(self, tmp_path, monkeypatch):
        """Repetitions must never share a cache bucket -- otherwise
        repetition 2/3/4 would silently return repetition 1's cached
        response instead of a genuinely independent call, defeating the
        whole point of a stochasticity measurement."""
        recorded_identities = []

        def _fake_build_backend(*args, **kwargs):
            recorded_identities.append(kwargs.get("model_identity"))
            return _FakeAsyncBackend()

        monkeypatch.setattr(mod, "build_backend", _fake_build_backend)
        source = _questions_csv(tmp_path, ["q1"])
        output = tmp_path / "out.csv"

        mod.run(
            source, _MODEL_CONFIG, "test_run", output, method_name="direct_mcq",
            prompt_version="v1", n_repetitions=4, run_seed=42,
        )

        assert len(recorded_identities) == 4
        assert len(set(recorded_identities)) == 4  # all distinct
        assert all(i is not None for i in recorded_identities)

    def test_method_name_is_stamped_correctly(self, tmp_path, monkeypatch):
        backend = _FakeAsyncBackend()
        monkeypatch.setattr(mod, "build_backend", lambda *a, **k: backend)
        source = _questions_csv(tmp_path, ["q1"])
        output = tmp_path / "out.csv"

        mod.run(
            source, _MODEL_CONFIG, "test_run", output, method_name="reasoning_mcq",
            prompt_version="v1_reasoning", n_repetitions=4, run_seed=42,
        )

        result_df = pd.read_csv(output)
        assert (result_df["method_name"] == "reasoning_mcq").all()

    def test_resume_skips_already_completed_question_repetition_pairs(self, tmp_path, monkeypatch):
        backend = _FakeAsyncBackend()
        monkeypatch.setattr(mod, "build_backend", lambda *a, **k: backend)
        source = _questions_csv(tmp_path, ["q1", "q2"])
        output = tmp_path / "out.csv"

        n_first = mod.run(
            source, _MODEL_CONFIG, "test_run", output, method_name="direct_mcq",
            prompt_version="v1", n_repetitions=4, run_seed=42,
        )
        assert n_first == 8

        backend2 = _FakeAsyncBackend()
        monkeypatch.setattr(mod, "build_backend", lambda *a, **k: backend2)
        n_second = mod.run(
            source, _MODEL_CONFIG, "test_run", output, method_name="direct_mcq",
            prompt_version="v1", n_repetitions=4, run_seed=42,
        )

        assert n_second == 0
        assert backend2.generate_batch_calls == []
        result_df = pd.read_csv(output)
        assert len(result_df) == 8  # no duplicates

    def test_resume_after_partial_completion_only_redoes_remaining_pairs(self, tmp_path, monkeypatch):
        # Realistic partial state: an actual prior invocation of this same
        # script, not a hand-built fixture with a different column set --
        # a real resume's existing file always has the matching schema,
        # since only this script's own writer ever produced it.
        backend = _FakeAsyncBackend()
        monkeypatch.setattr(mod, "build_backend", lambda *a, **k: backend)
        source = _questions_csv(tmp_path, ["q1", "q2"])
        output = tmp_path / "out.csv"

        n_partial = mod.run(
            source, _MODEL_CONFIG, "test_run", output, method_name="direct_mcq",
            prompt_version="v1", n_repetitions=2, run_seed=42,
        )
        assert n_partial == 4  # 2 questions x reps 0-1

        backend2 = _FakeAsyncBackend()
        monkeypatch.setattr(mod, "build_backend", lambda *a, **k: backend2)
        n_written = mod.run(
            source, _MODEL_CONFIG, "test_run", output, method_name="direct_mcq",
            prompt_version="v1", n_repetitions=4, run_seed=42,
        )

        # Both questions still need reps 2,3 (2 more each) = 4.
        assert n_written == 4
        result_df = pd.read_csv(output)
        assert len(result_df) == 8
        assert sorted(result_df["repetition_index"].unique().tolist()) == [0, 1, 2, 3]

    def test_a_schema_mismatch_against_the_existing_output_file_raises_clearly(self, tmp_path, monkeypatch):
        """An existing output file whose columns don't match what this run
        would write (e.g. produced by an older version of this script)
        must fail loudly, not silently corrupt the CSV with rows of
        differing field counts."""
        backend = _FakeAsyncBackend()
        monkeypatch.setattr(mod, "build_backend", lambda *a, **k: backend)
        source = _questions_csv(tmp_path, ["q1", "q2"])
        output = tmp_path / "out.csv"

        stale = pd.DataFrame([
            {"question_id": "q1", "repetition_index": 0, "method_name": "direct_mcq"},
        ])
        stale.to_csv(output, index=False)

        with pytest.raises(ValueError, match="schema"):
            mod.run(
                source, _MODEL_CONFIG, "test_run", output, method_name="direct_mcq",
                prompt_version="v1", n_repetitions=4, run_seed=42,
            )

    def test_missing_question_text_column_raises_clear_error(self, tmp_path):
        df = pd.DataFrame([{"question_id": "q1"}])
        path = tmp_path / "bad.csv"
        df.to_csv(path, index=False)
        with pytest.raises(ValueError, match="question_text"):
            mod.run(
                path, _MODEL_CONFIG, "test_run", tmp_path / "out.csv",
                method_name="direct_mcq", prompt_version="v1",
            )


class TestRunStochasticityRepeatsTwoStage:
    """two_stage / reasoning_two_stage: dependent, stays synchronous --
    still uses run_many_async (concurrent dispatch, not provider Batch
    API), one call per repetition just like the baseline path."""

    def test_writes_rows_with_free_text_response_preserved(self, tmp_path, monkeypatch):
        backend = _FakeAsyncBackend(response_text="C")
        monkeypatch.setattr(mod, "build_backend", lambda *a, **k: backend)
        source = _questions_csv(tmp_path, ["q1"])
        output = tmp_path / "out.csv"

        n_written = mod.run(
            source, _MODEL_CONFIG, "test_run", output, method_name="two_stage",
            prompt_version="v1", n_repetitions=2, run_seed=42, execution_mode="sync",
        )

        assert n_written == 2
        result_df = pd.read_csv(output)
        assert "free_text_response" in result_df.columns
        assert (result_df["method_name"] == "two_stage").all()

    def test_each_repetition_reruns_both_stages_independently(self, tmp_path, monkeypatch):
        """"Complete independent Stage1->Stage2 repeats" -- stage 1 must
        be re-elicited every repetition, never reused across repetitions
        the way the flip-rate rotation scripts reuse it across rotations."""
        backend = _FakeAsyncBackend(response_text="C")
        monkeypatch.setattr(mod, "build_backend", lambda *a, **k: backend)
        source = _questions_csv(tmp_path, ["q1"])
        output = tmp_path / "out.csv"

        mod.run(
            source, _MODEL_CONFIG, "test_run", output, method_name="two_stage",
            prompt_version="v1", n_repetitions=3, run_seed=42, execution_mode="sync",
        )

        # 3 repetitions x 2 calls (stage1 + stage2) = 6 total prompts, as
        # 3 pairs of generate_batch() calls (one pair per repetition).
        assert len(backend.generate_batch_calls) == 6
