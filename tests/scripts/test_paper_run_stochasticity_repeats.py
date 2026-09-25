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
    tests can assert on call shape (one call per FRESH repetition, not
    one per question, and never one for observation 0) without needing
    a real provider."""

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


_DEFAULT_OBS0_IDENTITY = dict(
    provider="openai", model_name="gpt-4.1-mini", benchmark_name="mmlu",
    prompt_version="v1", temperature=0.0, max_tokens=1024,
)


def _canonical_obs0_csv(tmp_path, rows) -> Path:
    """rows: list of dicts, each a full canonical row as the main
    accuracy run's own saved output would contain it -- must include
    question_id and method_name, and may include extra
    condition_metadata columns (experiment_id, condition_id, ...) that
    a freshly computed repetition row never has. Identity columns
    (provider/model_name/benchmark_name/prompt_version/temperature/
    max_tokens) default to matching _MODEL_CONFIG + the standard v1/mmlu
    test setup unless a row overrides them -- most tests care about
    something else and shouldn't have to restate the whole identity."""
    filled_rows = [{**_DEFAULT_OBS0_IDENTITY, **row} for row in rows]
    df = pd.DataFrame(filled_rows)
    path = tmp_path / "canonical_obs0.csv"
    df.to_csv(path, index=False)
    return path


class TestRunStochasticityRepeatsBaseline:
    """direct_mcq / reasoning_mcq: single-stage, batch-safe. Observation
    0 is always reused from the canonical artifact -- only repetitions
    1..n_repetitions-1 are fresh calls."""

    def test_writes_n_repetitions_times_n_questions_rows(self, tmp_path, monkeypatch):
        backend = _FakeAsyncBackend()
        monkeypatch.setattr(mod, "build_backend", lambda *a, **k: backend)
        source = _questions_csv(tmp_path, ["q1", "q2"])
        output = tmp_path / "out.csv"
        canonical = _canonical_obs0_csv(tmp_path, [
            {"question_id": "q1", "method_name": "direct_mcq", "parsed_choice": "C", "is_correct": True},
            {"question_id": "q2", "method_name": "direct_mcq", "parsed_choice": "A", "is_correct": False},
        ])

        n_written = mod.run(
            source, _MODEL_CONFIG, "test_run", output, method_name="direct_mcq",
            prompt_version="v1", canonical_obs0_csv=canonical, n_repetitions=4, run_seed=42,
        )

        assert n_written == 8  # 2 questions x 4 repetitions (1 reused + 3 fresh)
        result_df = pd.read_csv(output)
        assert len(result_df) == 8
        assert sorted(result_df["repetition_index"].unique().tolist()) == [0, 1, 2, 3]
        assert set(result_df["question_id"]) == {"q1", "q2"}

    def test_makes_one_generate_batch_call_per_fresh_repetition(self, tmp_path, monkeypatch):
        backend = _FakeAsyncBackend()
        monkeypatch.setattr(mod, "build_backend", lambda *a, **k: backend)
        source = _questions_csv(tmp_path, ["q1", "q2", "q3"])
        output = tmp_path / "out.csv"
        canonical = _canonical_obs0_csv(tmp_path, [
            {"question_id": qid, "method_name": "direct_mcq", "parsed_choice": "C", "is_correct": True}
            for qid in ["q1", "q2", "q3"]
        ])

        mod.run(
            source, _MODEL_CONFIG, "test_run", output, method_name="direct_mcq",
            prompt_version="v1", canonical_obs0_csv=canonical, n_repetitions=4, run_seed=42,
        )

        # 4 repetitions, but repetition 0 is reused (no call) -- only
        # reps 1,2,3 are fresh -> 3 generate_batch() calls, each covering
        # all 3 questions at once (not one call per question).
        assert len(backend.generate_batch_calls) == 3
        assert all(len(call) == 3 for call in backend.generate_batch_calls)

    def test_each_fresh_repetition_gets_a_distinct_model_identity(self, tmp_path, monkeypatch):
        """Repetitions must never share a cache bucket -- otherwise
        repetition 2/3 would silently return repetition 1's cached
        response instead of a genuinely independent call, defeating the
        whole point of a stochasticity measurement."""
        recorded_identities = []

        def _fake_build_backend(*args, **kwargs):
            recorded_identities.append(kwargs.get("model_identity"))
            return _FakeAsyncBackend()

        monkeypatch.setattr(mod, "build_backend", _fake_build_backend)
        source = _questions_csv(tmp_path, ["q1"])
        output = tmp_path / "out.csv"
        canonical = _canonical_obs0_csv(tmp_path, [
            {"question_id": "q1", "method_name": "direct_mcq", "parsed_choice": "C", "is_correct": True},
        ])

        mod.run(
            source, _MODEL_CONFIG, "test_run", output, method_name="direct_mcq",
            prompt_version="v1", canonical_obs0_csv=canonical, n_repetitions=4, run_seed=42,
        )

        # Only the 3 FRESH repetitions (1,2,3) ever build a backend --
        # observation 0 is reused and never calls build_backend at all.
        assert len(recorded_identities) == 3
        assert len(set(recorded_identities)) == 3  # all distinct
        assert all(i is not None for i in recorded_identities)

    def test_fresh_repetitions_use_the_same_run_seed_never_a_varied_one(self, tmp_path, monkeypatch):
        """"Same request configuration + separate real invocations + no
        response-cache reuse" -- NOT explicitly different provider seeds
        per repetition, which would deliberately change the request and
        measure a different phenomenon. The distinct cache/invocation
        identity across fresh repetitions comes entirely from
        model_identity (see the sibling test above); run_seed itself must
        be identical across every fresh repetition."""
        recorded_calls = []

        def _fake_build_backend(*args, **kwargs):
            recorded_calls.append(kwargs)
            return _FakeAsyncBackend()

        monkeypatch.setattr(mod, "build_backend", _fake_build_backend)
        source = _questions_csv(tmp_path, ["q1"])
        output = tmp_path / "out.csv"
        canonical = _canonical_obs0_csv(tmp_path, [
            {"question_id": "q1", "method_name": "direct_mcq", "parsed_choice": "C", "is_correct": True},
        ])

        mod.run(
            source, _MODEL_CONFIG, "test_run", output, method_name="direct_mcq",
            prompt_version="v1", canonical_obs0_csv=canonical, n_repetitions=4, run_seed=42,
        )

        assert len(recorded_calls) == 3  # reps 1, 2, 3
        run_seeds = [call.get("run_seed") for call in recorded_calls]
        assert run_seeds == [42, 42, 42]  # identical -- never varied per repetition
        # Distinct identity comes from model_identity alone, not from seed.
        model_identities = [call.get("model_identity") for call in recorded_calls]
        assert len(set(model_identities)) == 3

    def test_build_backend_never_receives_a_repetition_varying_provider_seed_kwarg(
            self, tmp_path, monkeypatch,
    ):
        """Same production seed policy for obs0 and every fresh repetition:
        this script must never pass a provider-seed-like override that
        varies per repetition -- provider_seed is purely a model_config
        field, resolved once by build_backend() itself, identically for
        every call sharing that model_config."""
        recorded_calls = []

        def _fake_build_backend(*args, **kwargs):
            recorded_calls.append(kwargs)
            return _FakeAsyncBackend()

        monkeypatch.setattr(mod, "build_backend", _fake_build_backend)
        source = _questions_csv(tmp_path, ["q1"])
        output = tmp_path / "out.csv"
        canonical = _canonical_obs0_csv(tmp_path, [
            {"question_id": "q1", "method_name": "direct_mcq", "parsed_choice": "C", "is_correct": True},
        ])

        mod.run(
            source, _MODEL_CONFIG, "test_run", output, method_name="direct_mcq",
            prompt_version="v1", canonical_obs0_csv=canonical, n_repetitions=4, run_seed=42,
        )

        for call in recorded_calls:
            assert "provider_seed" not in call
            assert "seed" not in call  # only run_seed/model_identity vary the call

    def test_method_name_is_stamped_correctly(self, tmp_path, monkeypatch):
        backend = _FakeAsyncBackend()
        monkeypatch.setattr(mod, "build_backend", lambda *a, **k: backend)
        source = _questions_csv(tmp_path, ["q1"])
        output = tmp_path / "out.csv"
        canonical = _canonical_obs0_csv(tmp_path, [
            {"question_id": "q1", "method_name": "reasoning_mcq", "parsed_choice": "C", "is_correct": True,
             "prompt_version": "v1_reasoning"},
        ])

        mod.run(
            source, _MODEL_CONFIG, "test_run", output, method_name="reasoning_mcq",
            prompt_version="v1_reasoning", canonical_obs0_csv=canonical, n_repetitions=4, run_seed=42,
        )

        result_df = pd.read_csv(output)
        assert (result_df["method_name"] == "reasoning_mcq").all()

    def test_resume_skips_already_completed_question_repetition_pairs(self, tmp_path, monkeypatch):
        backend = _FakeAsyncBackend()
        monkeypatch.setattr(mod, "build_backend", lambda *a, **k: backend)
        source = _questions_csv(tmp_path, ["q1", "q2"])
        output = tmp_path / "out.csv"
        canonical = _canonical_obs0_csv(tmp_path, [
            {"question_id": "q1", "method_name": "direct_mcq", "parsed_choice": "C", "is_correct": True},
            {"question_id": "q2", "method_name": "direct_mcq", "parsed_choice": "A", "is_correct": False},
        ])

        n_first = mod.run(
            source, _MODEL_CONFIG, "test_run", output, method_name="direct_mcq",
            prompt_version="v1", canonical_obs0_csv=canonical, n_repetitions=4, run_seed=42,
        )
        assert n_first == 8

        backend2 = _FakeAsyncBackend()
        monkeypatch.setattr(mod, "build_backend", lambda *a, **k: backend2)
        n_second = mod.run(
            source, _MODEL_CONFIG, "test_run", output, method_name="direct_mcq",
            prompt_version="v1", canonical_obs0_csv=canonical, n_repetitions=4, run_seed=42,
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
        canonical = _canonical_obs0_csv(tmp_path, [
            {"question_id": "q1", "method_name": "direct_mcq", "parsed_choice": "C", "is_correct": True},
            {"question_id": "q2", "method_name": "direct_mcq", "parsed_choice": "A", "is_correct": False},
        ])

        n_partial = mod.run(
            source, _MODEL_CONFIG, "test_run", output, method_name="direct_mcq",
            prompt_version="v1", canonical_obs0_csv=canonical, n_repetitions=2, run_seed=42,
        )
        assert n_partial == 4  # 2 questions x reps 0(reused),1(fresh)

        backend2 = _FakeAsyncBackend()
        monkeypatch.setattr(mod, "build_backend", lambda *a, **k: backend2)
        n_written = mod.run(
            source, _MODEL_CONFIG, "test_run", output, method_name="direct_mcq",
            prompt_version="v1", canonical_obs0_csv=canonical, n_repetitions=4, run_seed=42,
        )

        # Both questions still need reps 2,3 (2 more each) = 4. Rep 0 is
        # already done, so no canonical lookup happens on this call.
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
        canonical = _canonical_obs0_csv(tmp_path, [
            {"question_id": "q1", "method_name": "direct_mcq", "parsed_choice": "C", "is_correct": True},
            {"question_id": "q2", "method_name": "direct_mcq", "parsed_choice": "A", "is_correct": False},
        ])

        # An older-version file missing the repetition_index column
        # entirely -- required for this script's (question_id,
        # repetition_index) resumability, so it can't legitimately be
        # this script's own prior output.
        stale = pd.DataFrame([{"question_id": "q1", "method_name": "direct_mcq"}])
        stale.to_csv(output, index=False)

        with pytest.raises(ValueError, match="schema"):
            mod.run(
                source, _MODEL_CONFIG, "test_run", output, method_name="direct_mcq",
                prompt_version="v1", canonical_obs0_csv=canonical, n_repetitions=4, run_seed=42,
            )

    def test_missing_question_text_column_raises_clear_error(self, tmp_path):
        df = pd.DataFrame([{"question_id": "q1"}])
        path = tmp_path / "bad.csv"
        df.to_csv(path, index=False)
        with pytest.raises(ValueError, match="question_text"):
            mod.run(
                path, _MODEL_CONFIG, "test_run", tmp_path / "out.csv",
                method_name="direct_mcq", prompt_version="v1",
                canonical_obs0_csv=tmp_path / "unused_canonical.csv",
            )


class TestRunStochasticityRepeatsTwoStage:
    """two_stage / reasoning_two_stage: dependent, stays synchronous --
    still uses run_many_async (concurrent dispatch, not provider Batch
    API), one call pair per FRESH repetition just like the baseline
    path's one call per fresh repetition."""

    def test_writes_rows_with_free_text_response_preserved(self, tmp_path, monkeypatch):
        backend = _FakeAsyncBackend(response_text="C")
        monkeypatch.setattr(mod, "build_backend", lambda *a, **k: backend)
        source = _questions_csv(tmp_path, ["q1"])
        output = tmp_path / "out.csv"
        canonical = _canonical_obs0_csv(tmp_path, [
            {
                "question_id": "q1", "method_name": "two_stage", "parsed_choice": "C",
                "is_correct": True, "free_text_response": "HTTPS",
            },
        ])

        n_written = mod.run(
            source, _MODEL_CONFIG, "test_run", output, method_name="two_stage",
            prompt_version="v1", canonical_obs0_csv=canonical, n_repetitions=2,
            run_seed=42, execution_mode="sync",
        )

        assert n_written == 2  # rep 0 reused + rep 1 fresh
        result_df = pd.read_csv(output)
        assert "free_text_response" in result_df.columns
        assert (result_df["method_name"] == "two_stage").all()

    def test_each_fresh_repetition_reruns_both_stages_independently(self, tmp_path, monkeypatch):
        """"Complete independent Stage1->Stage2 repeats" -- stage 1 must
        be re-elicited every FRESH repetition, never reused across
        repetitions the way the flip-rate rotation scripts reuse it
        across rotations of the same call."""
        backend = _FakeAsyncBackend(response_text="C")
        monkeypatch.setattr(mod, "build_backend", lambda *a, **k: backend)
        source = _questions_csv(tmp_path, ["q1"])
        output = tmp_path / "out.csv"
        canonical = _canonical_obs0_csv(tmp_path, [
            {
                "question_id": "q1", "method_name": "two_stage", "parsed_choice": "C",
                "is_correct": True, "free_text_response": "HTTPS",
            },
        ])

        mod.run(
            source, _MODEL_CONFIG, "test_run", output, method_name="two_stage",
            prompt_version="v1", canonical_obs0_csv=canonical, n_repetitions=4,
            run_seed=42, execution_mode="sync",
        )

        # Reps 1,2,3 are fresh (rep 0 reused, no calls) x 2 calls
        # (stage1 + stage2) each = 6 total prompts, as 3 pairs of
        # generate_batch() calls (one pair per fresh repetition).
        assert len(backend.generate_batch_calls) == 6


class TestRunStochasticityRepeatsObs0Reuse:
    """Observation 0 must reuse the main accuracy run's own saved result,
    never a freshly generated replacement -- and must fail loudly, not
    silently substitute a fresh call, if that canonical artifact is
    missing or doesn't actually cover what's needed."""

    def test_missing_canonical_obs0_artifact_fails_loudly(self, tmp_path, monkeypatch):
        backend = _FakeAsyncBackend()
        monkeypatch.setattr(mod, "build_backend", lambda *a, **k: backend)
        source = _questions_csv(tmp_path, ["q1"])
        output = tmp_path / "out.csv"
        missing_canonical = tmp_path / "does_not_exist.csv"

        with pytest.raises(ValueError, match="does not exist"):
            mod.run(
                source, _MODEL_CONFIG, "test_run", output, method_name="direct_mcq",
                prompt_version="v1", canonical_obs0_csv=missing_canonical,
                n_repetitions=1, run_seed=42,
            )

        assert backend.generate_batch_calls == []  # never silently falls back to a fresh call

    def test_canonical_obs0_missing_question_id_fails_loudly(self, tmp_path, monkeypatch):
        backend = _FakeAsyncBackend()
        monkeypatch.setattr(mod, "build_backend", lambda *a, **k: backend)
        source = _questions_csv(tmp_path, ["q1", "q2"])
        output = tmp_path / "out.csv"
        # Canonical only covers q1 -- q2's observation 0 has no canonical
        # row to reuse.
        canonical = _canonical_obs0_csv(tmp_path, [
            {"question_id": "q1", "method_name": "direct_mcq", "parsed_choice": "C", "is_correct": True},
        ])

        with pytest.raises(ValueError, match="missing"):
            mod.run(
                source, _MODEL_CONFIG, "test_run", output, method_name="direct_mcq",
                prompt_version="v1", canonical_obs0_csv=canonical,
                n_repetitions=1, run_seed=42,
            )

    def test_canonical_obs0_wrong_method_name_fails_loudly(self, tmp_path, monkeypatch):
        """The canonical file exists and covers q1 -- but only under a
        DIFFERENT method_name. Must not be mistaken for direct_mcq's
        observation 0."""
        backend = _FakeAsyncBackend()
        monkeypatch.setattr(mod, "build_backend", lambda *a, **k: backend)
        source = _questions_csv(tmp_path, ["q1"])
        output = tmp_path / "out.csv"
        canonical = _canonical_obs0_csv(tmp_path, [
            {"question_id": "q1", "method_name": "two_stage", "parsed_choice": "C", "is_correct": True},
        ])

        with pytest.raises(ValueError, match="missing"):
            mod.run(
                source, _MODEL_CONFIG, "test_run", output, method_name="direct_mcq",
                prompt_version="v1", canonical_obs0_csv=canonical,
                n_repetitions=1, run_seed=42,
            )

    def test_canonical_obs0_wrong_provider_fails_loudly(self, tmp_path, monkeypatch):
        """A Llama sync (openrouter) canonical row must never be reused as
        observation 0 for a batch (deepinfra) stochasticity run -- same
        nominal model, different deployment, different provider."""
        backend = _FakeAsyncBackend()
        monkeypatch.setattr(mod, "build_backend", lambda *a, **k: backend)
        source = _questions_csv(tmp_path, ["q1"])
        output = tmp_path / "out.csv"
        canonical = _canonical_obs0_csv(tmp_path, [
            {"question_id": "q1", "method_name": "direct_mcq", "parsed_choice": "C",
             "is_correct": True, "provider": "openrouter"},
        ])

        with pytest.raises(ValueError, match="provider"):
            mod.run(
                source, _MODEL_CONFIG, "test_run", output, method_name="direct_mcq",
                prompt_version="v1", canonical_obs0_csv=canonical,
                n_repetitions=1, run_seed=42,
            )

    def test_canonical_obs0_wrong_model_name_fails_loudly(self, tmp_path, monkeypatch):
        backend = _FakeAsyncBackend()
        monkeypatch.setattr(mod, "build_backend", lambda *a, **k: backend)
        source = _questions_csv(tmp_path, ["q1"])
        output = tmp_path / "out.csv"
        canonical = _canonical_obs0_csv(tmp_path, [
            {"question_id": "q1", "method_name": "direct_mcq", "parsed_choice": "C",
             "is_correct": True, "model_name": "claude-haiku-4-5"},
        ])

        with pytest.raises(ValueError, match="model_name"):
            mod.run(
                source, _MODEL_CONFIG, "test_run", output, method_name="direct_mcq",
                prompt_version="v1", canonical_obs0_csv=canonical,
                n_repetitions=1, run_seed=42,
            )

    def test_canonical_obs0_wrong_benchmark_fails_loudly(self, tmp_path, monkeypatch):
        backend = _FakeAsyncBackend()
        monkeypatch.setattr(mod, "build_backend", lambda *a, **k: backend)
        source = _questions_csv(tmp_path, ["q1"])
        output = tmp_path / "out.csv"
        canonical = _canonical_obs0_csv(tmp_path, [
            {"question_id": "q1", "method_name": "direct_mcq", "parsed_choice": "C",
             "is_correct": True, "benchmark_name": "arc_challenge"},
        ])

        with pytest.raises(ValueError, match="benchmark_name"):
            mod.run(
                source, _MODEL_CONFIG, "test_run", output, method_name="direct_mcq",
                prompt_version="v1", canonical_obs0_csv=canonical,
                n_repetitions=1, run_seed=42,
            )

    def test_canonical_obs0_wrong_prompt_version_fails_loudly(self, tmp_path, monkeypatch):
        backend = _FakeAsyncBackend()
        monkeypatch.setattr(mod, "build_backend", lambda *a, **k: backend)
        source = _questions_csv(tmp_path, ["q1"])
        output = tmp_path / "out.csv"
        canonical = _canonical_obs0_csv(tmp_path, [
            {"question_id": "q1", "method_name": "direct_mcq", "parsed_choice": "C",
             "is_correct": True, "prompt_version": "v1_reasoning"},
        ])

        with pytest.raises(ValueError, match="prompt_version"):
            mod.run(
                source, _MODEL_CONFIG, "test_run", output, method_name="direct_mcq",
                prompt_version="v1", canonical_obs0_csv=canonical,
                n_repetitions=1, run_seed=42,
            )

    def test_canonical_obs0_wrong_temperature_fails_loudly(self, tmp_path, monkeypatch):
        backend = _FakeAsyncBackend()
        monkeypatch.setattr(mod, "build_backend", lambda *a, **k: backend)
        source = _questions_csv(tmp_path, ["q1"])
        output = tmp_path / "out.csv"
        canonical = _canonical_obs0_csv(tmp_path, [
            {"question_id": "q1", "method_name": "direct_mcq", "parsed_choice": "C",
             "is_correct": True, "temperature": 0.7},
        ])

        with pytest.raises(ValueError, match="temperature"):
            mod.run(
                source, _MODEL_CONFIG, "test_run", output, method_name="direct_mcq",
                prompt_version="v1", canonical_obs0_csv=canonical,
                n_repetitions=1, run_seed=42,
            )

    def test_canonical_obs0_wrong_max_tokens_fails_loudly(self, tmp_path, monkeypatch):
        backend = _FakeAsyncBackend()
        monkeypatch.setattr(mod, "build_backend", lambda *a, **k: backend)
        source = _questions_csv(tmp_path, ["q1"])
        output = tmp_path / "out.csv"
        canonical = _canonical_obs0_csv(tmp_path, [
            {"question_id": "q1", "method_name": "direct_mcq", "parsed_choice": "C",
             "is_correct": True, "max_tokens": 512},
        ])

        with pytest.raises(ValueError, match="max_tokens"):
            mod.run(
                source, _MODEL_CONFIG, "test_run", output, method_name="direct_mcq",
                prompt_version="v1", canonical_obs0_csv=canonical,
                n_repetitions=1, run_seed=42,
            )

    def test_canonical_obs0_missing_identity_column_fails_loudly(self, tmp_path, monkeypatch):
        """A canonical artifact missing a whole identity column (e.g. an
        older or hand-built file) must fail loudly, not silently skip
        that check."""
        backend = _FakeAsyncBackend()
        monkeypatch.setattr(mod, "build_backend", lambda *a, **k: backend)
        source = _questions_csv(tmp_path, ["q1"])
        output = tmp_path / "out.csv"
        canonical_path = tmp_path / "canonical_obs0.csv"
        pd.DataFrame([
            {"question_id": "q1", "method_name": "direct_mcq", "parsed_choice": "C", "is_correct": True},
        ]).to_csv(canonical_path, index=False)  # no provider/model_name/... columns at all

        with pytest.raises(ValueError, match="identity column"):
            mod.run(
                source, _MODEL_CONFIG, "test_run", output, method_name="direct_mcq",
                prompt_version="v1", canonical_obs0_csv=canonical_path,
                n_repetitions=1, run_seed=42,
            )

    def test_canonical_obs0_ambiguous_duplicate_rows_fail_loudly(self, tmp_path, monkeypatch):
        """Two conflicting rows for the same question_id under the same
        method_name -- must never silently pick one via keep='first'."""
        backend = _FakeAsyncBackend()
        monkeypatch.setattr(mod, "build_backend", lambda *a, **k: backend)
        source = _questions_csv(tmp_path, ["q1"])
        output = tmp_path / "out.csv"
        canonical = _canonical_obs0_csv(tmp_path, [
            {"question_id": "q1", "method_name": "direct_mcq", "parsed_choice": "C", "is_correct": True},
            {"question_id": "q1", "method_name": "direct_mcq", "parsed_choice": "A", "is_correct": False},
        ])

        with pytest.raises(ValueError, match="more than one row"):
            mod.run(
                source, _MODEL_CONFIG, "test_run", output, method_name="direct_mcq",
                prompt_version="v1", canonical_obs0_csv=canonical,
                n_repetitions=1, run_seed=42,
            )

    def test_reused_obs0_row_uses_canonical_values_without_a_fresh_call(self, tmp_path, monkeypatch):
        backend = _FakeAsyncBackend()
        monkeypatch.setattr(mod, "build_backend", lambda *a, **k: backend)
        source = _questions_csv(tmp_path, ["q1"])
        output = tmp_path / "out.csv"
        canonical = _canonical_obs0_csv(tmp_path, [
            {
                "question_id": "q1", "method_name": "direct_mcq", "parsed_choice": "C",
                "is_correct": True, "experiment_id": "exp1", "condition_id": "cond1",
            },
        ])

        n_written = mod.run(
            source, _MODEL_CONFIG, "test_run", output, method_name="direct_mcq",
            prompt_version="v1", canonical_obs0_csv=canonical,
            n_repetitions=1, run_seed=42,
        )

        assert n_written == 1
        assert backend.generate_batch_calls == []  # no fresh call for observation 0
        result_df = pd.read_csv(output)
        assert result_df.iloc[0]["repetition_index"] == 0
        assert result_df.iloc[0]["parsed_choice"] == "C"
        assert result_df.iloc[0]["experiment_id"] == "exp1"  # condition metadata carried through

    def test_reused_obs0_row_and_fresh_rows_coexist_in_the_same_output_file(self, tmp_path, monkeypatch):
        """Reused observation-0 rows (extra condition_metadata columns
        from the main grid run) and freshly computed rows (without them)
        are legitimately heterogeneous -- both must land in the same
        output file without corrupting it."""
        backend = _FakeAsyncBackend()
        monkeypatch.setattr(mod, "build_backend", lambda *a, **k: backend)
        source = _questions_csv(tmp_path, ["q1"])
        output = tmp_path / "out.csv"
        canonical = _canonical_obs0_csv(tmp_path, [
            {
                "question_id": "q1", "method_name": "direct_mcq", "parsed_choice": "C",
                "is_correct": True, "experiment_id": "exp1",
            },
        ])

        n_written = mod.run(
            source, _MODEL_CONFIG, "test_run", output, method_name="direct_mcq",
            prompt_version="v1", canonical_obs0_csv=canonical,
            n_repetitions=2, run_seed=42,
        )

        assert n_written == 2
        result_df = pd.read_csv(output)
        assert len(result_df) == 2
        obs0_row = result_df[result_df["repetition_index"] == 0].iloc[0]
        obs1_row = result_df[result_df["repetition_index"] == 1].iloc[0]
        assert obs0_row["experiment_id"] == "exp1"
        assert pd.isna(obs1_row["experiment_id"])  # fresh row never had this column


class TestRunStochasticityExecutionModePolicy:
    """two_stage/reasoning_two_stage have their own stage-1/stage-2
    dependency per repetition and no multi-wave batching is built for
    them -- execution_mode='batch' must never be silently honored for
    them, whether it arrives as an explicit argument or (previously) as
    the script's own unconditional default."""

    def test_two_stage_under_batch_execution_mode_raises(self, tmp_path, monkeypatch):
        backend = _FakeAsyncBackend()
        monkeypatch.setattr(mod, "build_backend", lambda *a, **k: backend)
        source = _questions_csv(tmp_path, ["q1"])
        output = tmp_path / "out.csv"
        canonical = _canonical_obs0_csv(tmp_path, [
            {"question_id": "q1", "method_name": "two_stage", "parsed_choice": "C", "is_correct": True},
        ])

        with pytest.raises(ValueError, match="execution_mode"):
            mod.run(
                source, _MODEL_CONFIG, "test_run", output, method_name="two_stage",
                prompt_version="v1", canonical_obs0_csv=canonical,
                n_repetitions=1, run_seed=42, execution_mode="batch",
            )

    def test_reasoning_two_stage_under_batch_execution_mode_raises(self, tmp_path, monkeypatch):
        backend = _FakeAsyncBackend()
        monkeypatch.setattr(mod, "build_backend", lambda *a, **k: backend)
        source = _questions_csv(tmp_path, ["q1"])
        output = tmp_path / "out.csv"
        canonical = _canonical_obs0_csv(tmp_path, [
            {"question_id": "q1", "method_name": "reasoning_two_stage", "parsed_choice": "C",
             "is_correct": True, "prompt_version": "v1_reasoning"},
        ])

        with pytest.raises(ValueError, match="execution_mode"):
            mod.run(
                source, _MODEL_CONFIG, "test_run", output, method_name="reasoning_two_stage",
                prompt_version="v1_reasoning", canonical_obs0_csv=canonical,
                n_repetitions=1, run_seed=42, execution_mode="batch",
            )

    def test_direct_mcq_is_not_restricted_to_batch(self, tmp_path, monkeypatch):
        """The sync-only restriction is specific to the dependent methods
        -- direct_mcq/reasoning_mcq are batch-safe either way, so sync is
        still accepted (just not the frozen-policy default)."""
        backend = _FakeAsyncBackend()
        monkeypatch.setattr(mod, "build_backend", lambda *a, **k: backend)
        source = _questions_csv(tmp_path, ["q1"])
        output = tmp_path / "out.csv"
        canonical = _canonical_obs0_csv(tmp_path, [
            {"question_id": "q1", "method_name": "direct_mcq", "parsed_choice": "C", "is_correct": True},
        ])

        n_written = mod.run(
            source, _MODEL_CONFIG, "test_run", output, method_name="direct_mcq",
            prompt_version="v1", canonical_obs0_csv=canonical,
            n_repetitions=1, run_seed=42, execution_mode="sync",
        )
        assert n_written == 1

    def test_validate_execution_mode_directly(self):
        mod._validate_execution_mode("two_stage", "sync")  # must not raise
        mod._validate_execution_mode("direct_mcq", "batch")  # must not raise
        mod._validate_execution_mode("direct_mcq", "sync")  # must not raise
        with pytest.raises(ValueError, match="execution_mode"):
            mod._validate_execution_mode("two_stage", "batch")
        with pytest.raises(ValueError, match="execution_mode"):
            mod._validate_execution_mode("reasoning_two_stage", "batch")
