# tests/remediation/test_prefix_counterexamples.py
#
# BATCH 0 — pre-fix counterexamples for the consolidated remediation.
#
# Protocol (Phase-10 A13):
#   * Every test below is marked xfail(strict=True, reason="pre-fix <FINDING-ID>").
#     On current trunk it must FAIL for the documented reason (shown as XFAIL).
#     After its remediation batch lands, strict xfail turns XPASS into an ERROR,
#     forcing the marker's removal; the test then guards the fixed behavior
#     permanently.
#   * "Guard" tests (no xfail) encode behavior that must hold BOTH before and
#     after a fix — they prevent overreach (e.g. naive "always last letter").
#
# Existing-test specification changes REQUIRED by ratified policies (none made
# in Batch 0; each lands with its implementing batch and moves TOWARD the
# ratified semantics — no weakening):
#   S1 (Batch 1, mechanical API migration, not semantic):
#      - tests/runners/test_permutation.py: TestGeneratePermutations /
#        TestBuildPermutedPrompt / un-permute tests re-target the Rotation
#        API returned by generate_permutations() (mapping + slot map).
#      - Any direct callers of PermutationRunner._generate_permutations /
#        _unpermute_choice in runner tests follow the same signature.
#   S2 (Batch 2): tests/fixtures/sample_mcq_outputs.json case
#      "conflicting_letters" ("A or C") flips expected_status parse_ok ->
#      parse_ambiguous, expected_choice C -> None (ratified P6-5: compound
#      answer constructions are noncommittal). Assertion rename follows:
#      test_rejects_multiple_conflicting_letters finally matches its name.
#   S3 (Batch 2): test_last_weak_candidate_wins_when_multiple_weak_candidates
#      ("A or C or B" -> B) flips to PARSE_AMBIGUOUS for the same reason.
#   Verified NOT needing changes (checked against ratified semantics):
#      - test_prefers_strong_candidate_over_weak_mentions
#        ("The final answer is B, not A or C" -> B): selection language beats
#        bare weak letters under the new ranking too.
#      - test_last_strong_cue_wins_when_multiple_strong_candidates
#        ("The final answer is B or maybe the answer is C" -> C): two separate
#        single-candidate constructions; the later one supersedes. Encoded as
#        GuardF2::test_supersession_keeps_later_construction.
#      - test_handles_reasoning_plus_final_answer_format
#        ("The symptoms fit option A best. Final answer: A"): discussion cue
#        demotion cannot flip this — the trailing selection construction wins.

from pathlib import Path
from types import SimpleNamespace

import pytest

from choicebench.backends.dummy_backend import DummyBackend
from choicebench.benchmarks.base import make_normalized_row
from choicebench.backends.hf_backend import HuggingFaceBackend
from choicebench.config.schema import ConfigError, load_config
from choicebench.methods import (
    DirectMCQRunner,
    PriDeRunner,
    ShuffledBaselineRunner,
    TwoStageRunner,
)
from choicebench.parsing.parser import parse_model_answer
from choicebench.parsing.types import PARSE_AMBIGUOUS, PARSE_MISSING

REPO_ROOT = Path(__file__).resolve().parents[2]
PROMPTS_DIR = REPO_ROOT / "prompts"


def _row(subject="abstract_algebra", choices=("Paris", "London", "Berlin", "Madrid"),
         correct_index=0):
    return make_normalized_row(subject, "Q?", list(choices), correct_index=correct_index)


_OPTIONS = {"A": "Paris", "B": "London", "C": "Berlin", "D": "Madrid"}


# ---------------------------------------------------------------------------
# F1 — duplicate option text mis-scored by cyclic_permutation
# ---------------------------------------------------------------------------

class _PickTextBackend(DummyBackend):
    """Returns the displayed letter whose option text equals `wanted`."""

    def __init__(self, wanted: str):
        super().__init__()
        self._wanted = wanted

    def generate(self, prompt: str, **kwargs) -> str:
        for line in prompt.splitlines():
            if len(line) > 2 and line[1] == "." and line[3:].strip() == self._wanted:
                return line[0]
        raise AssertionError(f"option text {self._wanted!r} not rendered in prompt")


class _GoldSlotBackend(DummyBackend):
    """Selects the DISPLAYED SLOT whose canonical source is the gold identity.

    Two identical option texts carry no semantic information distinguishing
    their canonical twins, so no backend can "track gold content" through
    text alone. This synthetic backend instead exercises the ratified
    invariant directly: in rotation i it selects the displayed slot holding
    the canonical gold identity (letter[(c_star - i) % n]) and the test
    verifies that positional inversion preserves that identity (every vote
    maps back to the gold slot's own canonical position).

    Pre-fix, the text-scan inverse redirected every such vote to the
    earliest textual twin; post-fix, attribution stays positional.
    """

    def __init__(self, letters, c_star: int):
        super().__init__()
        self._letters = letters
        self._c_star = c_star
        self._call = 0

    def generate(self, prompt: str, **kwargs) -> str:
        letter = self._letters[(self._c_star - self._call) % len(self._letters)]
        self._call += 1
        return letter


class TestF1DuplicateOptionText:
    # A13: xfail marker REMOVED post-Batch-1 — this reproducer failed on
    # pre-fix trunk (text-scan redirected every gold-slot vote to the
    # earliest twin) and must stay green against the positional inverse.
    def test_gold_twin_scores_correct_when_model_picks_gold_content(self):
        from choicebench.methods.library.permutation import PermutationRunner

        row = _row(choices=("x", "dup", "dup", "y"), correct_index=2)  # gold = C (later twin)
        runner = PermutationRunner(
            backend=_GoldSlotBackend(list("ABCD"), c_star=2),
            method_name="cyclic_permutation",
            split_name="test", prompt_version="v1", prompts_dir=PROMPTS_DIR,
            run_id="r", temperature=0.0, max_tokens=8, seed=42,
        )
        out = runner.run_one(row, 0)
        assert out["parsed_choice"] == "C"
        assert out["is_correct"] is True
        assert out["answer_status"] == "success"


class TestF1GuardsMustStayGreen:
    def test_perfect_tracker_recovers_content_for_distinct_texts(self):
        """Distinct texts: every rotation, every content slot -> correct vote."""
        from choicebench.methods.library.permutation import PermutationRunner

        for n in (2, 3, 4, 5):
            letters = [chr(65 + i) for i in range(n)]
            texts = [f"c{i}" for i in range(n)]
            for c_star in range(n):
                row = make_normalized_row("s", "q", texts, correct_index=c_star)
                runner = PermutationRunner(
                    backend=_PickTextBackend(f"c{c_star}"), method_name="cyclic_permutation",
                    split_name="test", prompt_version="v1", prompts_dir=PROMPTS_DIR,
                    run_id="r", temperature=0.0, max_tokens=8, seed=42,
                )
                out = runner.run_one(row, 0)
                assert out["parsed_choice"] == letters[c_star], (n, c_star)


# ---------------------------------------------------------------------------
# F4 — pride_repro {subject} KeyError in four runners
# ---------------------------------------------------------------------------

class TestF4SubjectRenderingMatrix:
    def _pride_repro_row(self):
        return _row()  # subject abstract_algebra -> renders "about abstract algebra"

    def test_direct_mcq_renders_pride_repro_prompt(self):
        runner = DirectMCQRunner(
            backend=DummyBackend(), method_name="direct_mcq", split_name="test",
            prompt_version="pride_repro", prompts_dir=PROMPTS_DIR,
            run_id="r", temperature=0.0, max_tokens=8, seed=42,
        )
        prompt = runner._build_prompt(self._pride_repro_row())
        assert "about abstract algebra" in prompt

    def test_pride_renders_pride_repro_prompt(self):
        runner = PriDeRunner(
            backend=DummyBackend(), method_name="pride", split_name="test",
            prompt_version="pride_repro", prompts_dir=PROMPTS_DIR,
            run_id="r", temperature=0.0, max_tokens=8, seed=42,
        )
        prompt = runner._build_prompt(self._pride_repro_row())
        assert "about abstract algebra" in prompt

    def test_shuffled_baseline_renders_pride_repro_prompt(self):
        runner = ShuffledBaselineRunner(
            backend=DummyBackend(), method_name="shuffled_baseline", split_name="test",
            prompt_version="pride_repro", prompts_dir=PROMPTS_DIR,
            run_id="r", temperature=0.0, max_tokens=8, seed=42,
        )
        out = runner.run_one(self._pride_repro_row(), 0)
        assert "about abstract algebra" in out["prompt"]

    # A13: xfail marker REMOVED post-Batch-4 — the TwoStage fallback now
    # passes subject when rendering direct_mcq templates; failed on pre-fix
    # trunk with KeyError('subject') at fallback-prompt build time.
    def test_two_stage_fallback_renders_pride_repro_prompt(self):
        from tests.runners.conftest import MockBackend

        # Three backend calls flow through run_one: stage-1 free text,
        # stage-2 option matching (unparseable), then the direct-MCQ
        # fallback. On trunk the KeyError fired while BUILDING the third
        # prompt, before call 3 — hence two responses suffice there but not
        # post-fix.
        backend = MockBackend(responses=["Paris", "!!!unparseable!!!", "B"])
        runner = TwoStageRunner(
            backend=backend, method_name="two_stage", split_name="test",
            prompt_version="pride_repro", prompts_dir=PROMPTS_DIR,
            run_id="r", temperature=0.0, max_tokens=8, seed=42,
            fallback_on_parse_failure=True,
        )
        out = runner.run_one(self._pride_repro_row(), 0)
        assert out["fallback_used"] is True
        # The row's prompt column records the stage-2 matching prompt by
        # long-standing design (sync and async alike); the fallback
        # prompt's {subject} rendering is asserted against the request the
        # backend actually received for the scored fallback answer.
        assert "about abstract algebra" in backend.requests_received[-1]


# ---------------------------------------------------------------------------
# F5 / P3-F1 — MMLU-Pro null options stringified to "nan"
# ---------------------------------------------------------------------------

class TestF5NullOptionPolicy:
    # A13: xfail markers REMOVED post-Batch-5 — normalize_row now drops null
    # distractors and rejects null gold; both reproducers failed on pre-fix
    # trunk ('nan' stringified silently).
    def test_null_distractor_is_dropped(self):
        import json as _json

        from choicebench.benchmarks.mmlu_pro import normalize_row

        row = normalize_row({"category": "c", "question": "q",
                             "options": ["a", None, "c"], "answer_index": 0})
        texts = [item["text"] for item in _json.loads(row["choices_json"])]
        assert texts == ["a", "c"]
        assert "nan" not in texts

    def test_null_gold_option_is_rejected(self):
        from choicebench.benchmarks.mmlu_pro import normalize_row

        with pytest.raises(ValueError):
            normalize_row({"category": "c", "question": "q",
                           "options": ["a", None], "answer_index": 1})

    def test_null_distractor_before_gold_remaps_answer(self):
        import json as _json

        from choicebench.benchmarks.mmlu_pro import normalize_row

        row = normalize_row({"category": "c", "question": "q",
                             "options": [None, "a", "b"], "answer_index": 1})
        records = _json.loads(row["choices_json"])
        assert [rec["text"] for rec in records] == ["a", "b"]
        assert row["correct_option"] == "A"
        assert row["correct_answer_text"] == "a"

    def test_pandas_nan_distractor_treated_like_none(self):
        import json as _json

        from choicebench.benchmarks.mmlu_pro import normalize_row

        row = normalize_row({"category": "c", "question": "q",
                             "options": ["a", float("nan"), "c"],
                             "answer_index": 0})
        texts = [item["text"] for item in _json.loads(row["choices_json"])]
        assert texts == ["a", "c"]

    def test_validator_warns_on_legacy_nan_artifact_and_scanner_classifies(
        self, tmp_path, recwarn
    ):
        """F5 scanner: legacy 'nan' artifacts stay loadable but warn loudly."""
        from choicebench.benchmarks.base import make_normalized_row
        from choicebench.datasets import (
            DatasetSpec,
            artifact_csv_path,
            scan_null_option_rows,
            write_prepared_dataset,
        )
        import pandas as pd

        clean = make_normalized_row("s", "clean question", ["x", "y"], 0)
        legacy = make_normalized_row("s", "legacy question", ["x", "nan"], 0)
        df = pd.DataFrame([clean, legacy])
        spec = DatasetSpec(benchmark="toy", split="test")
        processed = tmp_path / "processed"
        write_prepared_dataset(df, processed, spec)

        loaded_df = pd.read_csv(
            artifact_csv_path(processed, spec), dtype={"question_id": "string"}
        )
        findings = scan_null_option_rows(loaded_df)
        assert len(findings) == 1
        assert findings[0]["kind"] == "distractor"

        from choicebench.datasets import validate_normalized_dataset

        validate_normalized_dataset(loaded_df)
        assert any("'nan' option text" in str(w.message) for w in recwarn.list)

    def test_scanner_flags_nan_gold_more_severely(self):
        from choicebench.datasets import scan_null_option_rows
        import pandas as pd

        row = make_normalized_row("s", "q", ["nan", "y"], 0)
        df = pd.DataFrame([row])
        findings = scan_null_option_rows(df)
        assert len(findings) == 1
        assert findings[0]["kind"] == "gold"


class TestP3F3ConsumedColumnNAGuard:
    """P3-F3: pandas NA-token coercion on read must never reach consumed
    columns. Guard asserts the write->load round trip keeps every normalized
    schema column NaN-free, so any future coercion becomes loud here."""

    CONSUMED_COLUMNS = [
        "question_id", "subject", "question_text", "choices_json",
        "correct_index", "correct_option", "correct_answer_text", "n_choices",
    ]

    def test_loaded_artifact_consumed_columns_contain_no_na(self, tmp_path):
        import pandas as pd

        from choicebench.datasets import (
            DatasetSpec,
            artifact_csv_path,
            load_prepared_dataset,
            write_prepared_dataset,
        )

        rows = [
            make_normalized_row("algebra", "What is 2+2?", ["3", "4"], 1),
            # Legit content mentioning NA-like tokens inside longer text must
            # survive verbatim (only full-cell tokens are coercible).
            make_normalized_row("chemistry", "Is NaCl related to NaN?", ["yes", "no"], 0),
        ]
        spec = DatasetSpec(benchmark="toy", split="test")
        processed = tmp_path / "processed"
        write_prepared_dataset(pd.DataFrame(rows), processed, spec)

        prepared = load_prepared_dataset(processed, spec)
        df = prepared.dataframe
        for col in self.CONSUMED_COLUMNS:
            assert not df[col].isna().any(), f"column {col} contains NA after load"
        assert df["question_text"].iloc[1] == "Is NaCl related to NaN?"


# ---------------------------------------------------------------------------
# P6-1 — tier-4 option-text fallback requires answer-like structure
# ---------------------------------------------------------------------------

class TestP61MentionIsNotSelection:
    # A13: xfails removed post-Batch-2 (all three flipped).
    def test_refusal_single_mention_is_missing(self):
        result = parse_model_answer("I cannot help with questions about Madrid.", _OPTIONS)
        assert result.status == PARSE_MISSING
        assert result.final_choice is None

    def test_meta_single_mention_is_missing(self):
        result = parse_model_answer(
            "This reminds me of an old question about London.", _OPTIONS)
        assert result.status == PARSE_MISSING

    def test_mention_then_explicit_selection_resolves_to_selection(self):
        result = parse_model_answer(
            "Madrid is discussed in option D, but my pick is B.", _OPTIONS)
        assert result.final_choice == "B"


class TestP61GuardsMustStayGreen:
    @pytest.mark.parametrize("text", [
        "Madrid",
        '"Madrid"',
        "**Madrid**",
        "<answer>Madrid</answer>",
        "Answer: Madrid",
        "My answer is Madrid.",
    ])
    def test_answer_like_structures_still_parse(self, text):
        result = parse_model_answer(text, _OPTIONS)
        assert result.final_choice == "D", text


# ---------------------------------------------------------------------------
# P6-4 — wrapper/typography blind spots
# ---------------------------------------------------------------------------

class TestP64Wrappers:
    # A13: xfails removed post-Batch-2 (all three flipped).
    def test_markdown_bold_letter(self):
        assert parse_model_answer("**B**", _OPTIONS).final_choice == "B"

    def test_xml_tagged_letter(self):
        assert parse_model_answer("<answer>A</answer>", _OPTIONS).final_choice == "A"

    def test_curly_quoted_letter(self):
        assert parse_model_answer("\u201cC\u201d is my choice", _OPTIONS).final_choice == "C"


# ---------------------------------------------------------------------------
# P6-5 — compound answer constructions are ambiguous
# ---------------------------------------------------------------------------

class TestP65CompoundAnswers:
    # A13: xfails removed post-Batch-2 (all three flipped).
    def test_cued_compound_is_ambiguous(self):
        result = parse_model_answer("Answer: A or B", _OPTIONS)
        assert result.status == PARSE_AMBIGUOUS
        assert result.final_choice is None

    def test_either_or_is_ambiguous(self):
        result = parse_model_answer("Either C or D", _OPTIONS)
        assert result.status == PARSE_AMBIGUOUS

    def test_hedged_compound_is_ambiguous(self):
        result = parse_model_answer("My answer is probably B or C", _OPTIONS)
        assert result.status == PARSE_AMBIGUOUS


class TestP65GuardsMustStayGreen:
    def test_coordination_then_later_selection_supersedes(self):
        result = parse_model_answer(
            "The final answer is B or maybe the answer is C", _OPTIONS)
        assert result.final_choice == "C"

    def test_enumeration_then_explicit_pick(self):
        result = parse_model_answer("Looking at A, B and C, my final pick is B.", _OPTIONS)
        assert result.final_choice == "B"


# ---------------------------------------------------------------------------
# F2 — incidental option reference vs explicit answer-selection language
# ---------------------------------------------------------------------------

class TestF2DiscussionVersusSelection:
    # A13: xfail removed post-Batch-2 (flipped).
    def test_incidental_reference_then_selection(self):
        result = parse_model_answer(
            "Option C refers to the capital of Germany. My pick is B.", _OPTIONS)
        assert result.final_choice == "B"


class TestF2GuardsMustStayGreen:
    def test_speculation_then_explicit_cue_wins(self):
        result = parse_model_answer("Maybe A? Hmm, hard. Final answer: C", _OPTIONS)
        assert result.final_choice == "C"


# ---------------------------------------------------------------------------
# F3 — run.seed plumbing for HF sampled decoding (ratified design)
# ---------------------------------------------------------------------------

class TestF3SeedPlumbing:
    # A13: xfail marker REMOVED post-Batch-3 — build_backend now forwards
    # run.seed into HF defaults; this reproducer failed on pre-fix trunk.
    def test_build_backend_forwards_run_seed_to_hf_defaults(self, monkeypatch):
        import choicebench.cli.run_experiment as run_exp
        from choicebench.config.schema import GenerationKwargsConfig, ModelConfig

        model_cfg = ModelConfig(
            backend="huggingface", model_name_or_path="fake/model", device="cpu",
            generation_kwargs=GenerationKwargsConfig(do_sample=True, temperature=0.7),
        )
        captured = {}

        class _ProbeBackend(run_exp.HuggingFaceBackend):
            def load(self):
                captured["defaults"] = dict(self._default_generation_kwargs)

        monkeypatch.setattr(run_exp, "HuggingFaceBackend", _ProbeBackend)
        run_exp.build_backend(model_cfg, "r", run_seed=123)
        assert captured["defaults"].get("seed") == 123

    # A13: xfail marker REMOVED post-Batch-3 — generate() now seeds torch
    # only when do_sample is set, and seed stays an internal control
    # parameter (never forwarded to transformers). Failed on pre-fix trunk
    # (RNG touched without sampling).
    def test_generate_seeds_only_when_sampling_and_seed_stays_internal(self):
        import sys

        from tests.backends.test_hf_backend import _FakeModel, _FakeTokenizer, _FakeTorch

        torch = _FakeTorch()
        backend = HuggingFaceBackend(
            "fake-model", "cpu", max_new_tokens=4,
            temperature=0.7, do_sample=False, seed=123,
        )
        tokenizer = _FakeTokenizer()
        model = _FakeModel()
        backend._loaded = True
        backend._tokenizer = tokenizer
        backend._model = model
        backend._torch = torch

        backend.generate("prompt")

        assert torch.seed is None                      # greedy: RNG untouched
        assert "seed" not in model.generate_kwargs     # internal control param


# ---------------------------------------------------------------------------
# P6-3 — OpenAI client drops finish/status provenance
# ---------------------------------------------------------------------------

class TestP63OpenAIStatusProvenance:
    @staticmethod
    def _generate_with_response(response):
        import asyncio

        from choicebench.clients.openai_client import OpenAIClient
        from choicebench.clients.types import ModelRequest

        class _FakeResponses:
            async def create(self, **kwargs):
                return response

        client = OpenAIClient(model_name="gpt-x", api_key="test-key")
        client.client.responses = _FakeResponses()
        request = ModelRequest(provider="openai", model_name="gpt-x",
                               payload="prompt", temperature=0.0, max_tokens=8)
        return asyncio.run(client._generate_provider_response(request))

    @staticmethod
    def _fake_response(status=..., details_reason=None):
        class _R:
            output_text = "B"
            usage = None

        _R.status = None if status is ... else status
        if details_reason is not None:
            _R.incomplete_details = SimpleNamespace(reason=details_reason)
        return _R()

    # A13: xfail marker REMOVED post-Batch-3 — this reproducer failed on
    # pre-fix trunk (finish_reason always None); it now passes.
    def test_incomplete_response_surfaces_truncation(self):
        response = self._generate_with_response(
            self._fake_response(status="incomplete")
        )
        assert response.finish_reason is not None
        assert response.finish_reason != "stop"

    def test_completed_maps_to_provider_truth_not_stop(self):
        response = self._generate_with_response(
            self._fake_response(status="completed")
        )
        assert response.finish_reason == "completed"

    def test_incomplete_max_output_tokens_maps_to_length(self):
        response = self._generate_with_response(
            self._fake_response(status="incomplete",
                                details_reason="max_output_tokens")
        )
        assert response.finish_reason == "length"

    def test_incomplete_content_filter_preserved_verbatim(self):
        response = self._generate_with_response(
            self._fake_response(status="incomplete",
                                details_reason="content_filter")
        )
        assert response.finish_reason == "content_filter"

    def test_incomplete_unknown_reason_preserved_verbatim(self):
        response = self._generate_with_response(
            self._fake_response(status="incomplete",
                                details_reason="some_future_reason")
        )
        assert response.finish_reason == "some_future_reason"

    @pytest.mark.parametrize("status", ["failed", "in_progress", "cancelled", "queued"])
    def test_other_lifecycle_statuses_preserved_coarsely(self, status):
        response = self._generate_with_response(self._fake_response(status=status))
        assert response.finish_reason == status

    def test_missing_legacy_status_yields_none_safely(self):
        response = self._generate_with_response(self._fake_response())
        assert response.finish_reason is None


# ---------------------------------------------------------------------------
# P9-1 — legacy loader ingests sibling-run CSVs via substring matching
# ---------------------------------------------------------------------------

class TestP91LoaderGlobIsolation:
    # A13: xfail marker REMOVED post-Batch-6 — the loader now requires run_id
    # as a filename-stem prefix; failed on pre-fix trunk (sibling "_dedup_"
    # runs were ingested via substring match).
    def test_sibling_run_csvs_are_not_ingested(self, tmp_path):
        import pandas as pd

        from examples.pride_from_artifacts import _read_all_run_results

        rows_orig = pd.DataFrame([
            {"question_id": "q1", "raw_text": "orig", "parsed_choice": "A",
             "correct_option": "A", "answer_status": "success"},
            {"question_id": "q2", "raw_text": "orig", "parsed_choice": "B",
             "correct_option": "B", "answer_status": "success"},
        ])
        rows_dedup = rows_orig.copy()
        rows_dedup["raw_text"] = "dedup"
        rows_orig.to_csv(tmp_path / "20260705_204054_direct_logprob_m_mmlu.csv", index=False)
        rows_dedup.to_csv(
            tmp_path / "20260705_204054_dedup_direct_logprob_m_mmlu.csv", index=False)

        df = _read_all_run_results(tmp_path, run_id="20260705_204054",
                                   method_name="direct_logprob")
        assert set(df["raw_text"]) == {"orig"}


# ---------------------------------------------------------------------------
# D1/D2 — documentation-as-spec drift (README vs registries)
# ---------------------------------------------------------------------------

def _readme_section(start_marker: str) -> str:
    text = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    start = text.index(start_marker)
    end = text.find("\n### ", start + len(start_marker))
    return text[start:end if end != -1 else len(text)]


class TestBatch6HygieneGuards:
    """Permanent guards for the Batch-6 hygiene fixes (P5-1, P3-F4, P3-F5)."""

    def test_p51_skip_path_gcs_stale_checkpoint(self, tmp_path):
        import pandas as pd

        import choicebench.cli.run_experiment as run_exp
        from choicebench.config.schema import ExperimentConfig
        from choicebench.io.writers import write_run_results

        identity = {
            "experiment_id": "exp", "condition_id": "cond",
            "dataset_artifact_id": "ds", "dataset_selection_id": "sel",
            "model_id": "model", "method_id": "method", "prompt_id": "prompt",
            "benchmark_split": "test",
        }
        questions = pd.DataFrame([{"question_id": "q1"}])
        write_run_results(
            [{"question_id": "q1", "is_correct": True, **identity}],
            tmp_path, "cond", "toy",
        )
        checkpoint = tmp_path / "checkpoints" / "cond.json"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_text('{"stale": true}')

        condition = {
            "condition_id": identity["condition_id"],
            "experiment_id": identity["experiment_id"],
            "selection_id": "sel", "artifact_id": "ds",
            "model_id": "model", "method_id": "method", "prompt_id": "prompt",
            "split": "test",
            "checkpoint_path": f"checkpoints/{identity['condition_id']}.json",
        }
        config = ExperimentConfig(
            name="unit", models=[], benchmarks=[], methods=[], metrics=[],
            run=RunConfigForTest().build(),
        )
        config.run.resume = True
        skipped = run_exp._check_existing_result(tmp_path, condition, questions, config)
        assert skipped is True
        assert not checkpoint.exists()

    def test_p51_no_checkpoint_is_fine_on_skip(self, tmp_path):
        import pandas as pd

        import choicebench.cli.run_experiment as run_exp
        from choicebench.config.schema import ExperimentConfig
        from choicebench.io.writers import write_run_results

        identity = {
            "experiment_id": "exp", "condition_id": "cond2",
            "dataset_artifact_id": "ds", "dataset_selection_id": "sel",
            "model_id": "model", "method_id": "method", "prompt_id": "prompt",
            "benchmark_split": "test",
        }
        questions = pd.DataFrame([{"question_id": "q1"}])
        write_run_results(
            [{"question_id": "q1", "is_correct": True, **identity}],
            tmp_path, "cond2", "toy",
        )
        condition = {
            "condition_id": identity["condition_id"],
            "experiment_id": identity["experiment_id"],
            "selection_id": "sel", "artifact_id": "ds",
            "model_id": "model", "method_id": "method", "prompt_id": "prompt",
            "split": "test",
            "checkpoint_path": "checkpoints/cond2.json",
        }
        config = ExperimentConfig(
            name="unit", models=[], benchmarks=[], methods=[], metrics=[],
            run=RunConfigForTest().build(),
        )
        config.run.resume = True
        assert run_exp._check_existing_result(tmp_path, condition, questions, config) is True

    def test_p34_arc_duplicate_labels_warn_and_resolve_first(self, caplog):
        import logging

        from choicebench.benchmarks.arc import normalize_row

        row = {
            "id": "dup-1", "question": "q",
            "choices": {"text": ["one", "two", "three"],
                        "label": ["A", "A", "B"]},
            "answerKey": "A",
        }
        with caplog.at_level(logging.WARNING, logger="choicebench.benchmarks.arc"):
            normalized = normalize_row(row)
        assert normalized["correct_option"] == "A"
        assert normalized["correct_answer_text"] == "one"
        assert any("duplicate choice labels" in rec.message for rec in caplog.records)

    def test_p34_arc_unique_labels_do_not_warn(self, caplog):
        import logging

        from choicebench.benchmarks.arc import normalize_row

        row = {
            "id": "ok-1", "question": "q",
            "choices": {"text": ["one", "two"], "label": ["A", "B"]},
            "answerKey": "B",
        }
        with caplog.at_level(logging.WARNING, logger="choicebench.benchmarks.arc"):
            normalize_row(row)
        assert not any("duplicate choice labels" in rec.message for rec in caplog.records)

    def test_p35_truthfulqa_multiple_gold_labels_warn_and_resolve_first(self, caplog):
        import logging

        from choicebench.benchmarks.truthful_qa import normalize_row

        row = {
            "question": "q",
            "mc1_targets": {"choices": ["a", "b", "c"], "labels": [1, 0, 1]},
        }
        with caplog.at_level(logging.WARNING, logger="choicebench.benchmarks.truthful_qa"):
            normalized = normalize_row(row)
        assert normalized["correct_index"] == 0
        assert any("gold (label==1) choices" in rec.message for rec in caplog.records)


class RunConfigForTest:
    @staticmethod
    def build():
        from choicebench.config.schema import RunConfig

        return RunConfig(seed=1, prompt_version="v1")


class TestDocsMatchRegistries:
    # A13: xfail marker REMOVED post-Batch-7 — README metrics table now
    # lists every built-in metric; failed on pre-fix trunk.
    def test_metrics_table_lists_every_builtin_metric(self):
        from choicebench.metrics import BUILTIN_METRICS

        section = _readme_section("### Metrics")
        listed = {line.split("`")[1] for line in section.splitlines()
                  if line.startswith("| `")}
        assert set(BUILTIN_METRICS) <= listed

    # A13: xfail marker REMOVED post-Batch-7 — README methods table now
    # lists every registered method; failed on pre-fix trunk.
    def test_methods_table_lists_every_registered_method(self):
        from choicebench.registry import METHOD_REGISTRY

        section = _readme_section("### Methods")
        listed = {line.split("`")[1] for line in section.splitlines()
                  if line.startswith("| `")}
        assert set(METHOD_REGISTRY) <= listed
