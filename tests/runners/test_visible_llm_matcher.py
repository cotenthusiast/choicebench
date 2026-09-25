# tests/runners/test_visible_llm_matcher.py

import json
from pathlib import Path

from choicebench.methods.library.permutation import PermutationRunner
from choicebench.methods.library.visible_llm_matcher import VisibleLLMMatcherRunner
from choicebench.scoring.types import SCORE_CORRECT, SCORE_INCORRECT, SCORE_UNSCORABLE

from tests.runners.conftest import MockBackend

REPO_ROOT = Path(__file__).resolve().parents[2]
_PROMPTS_DIR = REPO_ROOT / "prompts"


def _make_runner(backend):
    return VisibleLLMMatcherRunner(
        backend=backend,
        method_name="visible_llm_matcher",
        split_name="test",
        prompt_version="v1",
        prompts_dir=_PROMPTS_DIR,
        run_id="test_run_001",
        seed=42,
        benchmark_name="mmlu",
    )


def _row_with_extracted_text(base_row, extracted_text):
    row = dict(base_row)
    row["extracted_text"] = extracted_text
    return row


class TestVisibleLLMMatcherRunOne:
    def test_one_new_call_per_question(self, runner_question_row):
        """Stage 1 (text_extraction) is reused, never re-called -- exactly
        one new call here, the Stage-2 LLM match."""
        backend = MockBackend(responses=["C"])
        row = _row_with_extracted_text(runner_question_row, "HTTPS")
        _make_runner(backend).run_one(row, sample_index=0)
        assert len(backend.requests_received) == 1

    def test_prompt_includes_the_reused_extracted_text_and_visible_options(self, runner_question_row):
        backend = MockBackend(responses=["C"])
        row = _row_with_extracted_text(runner_question_row, "HTTPS")
        _make_runner(backend).run_one(row, sample_index=0)
        prompt = backend.requests_received[0]
        assert "HTTPS" in prompt
        assert "A. FTP" in prompt
        assert "B. HTTP" in prompt
        assert "D. SMTP" in prompt

    def test_correct_match_scores_correct(self, runner_question_row):
        backend = MockBackend(responses=["C"])
        row = _row_with_extracted_text(runner_question_row, "HTTPS")
        result = _make_runner(backend).run_one(row, sample_index=0)
        assert result["parsed_choice"] == "C"
        assert result["is_correct"] is True
        assert result["score_status"] == SCORE_CORRECT

    def test_incorrect_match(self, runner_question_row):
        backend = MockBackend(responses=["A"])
        row = _row_with_extracted_text(runner_question_row, "HTTPS")
        result = _make_runner(backend).run_one(row, sample_index=0)
        assert result["is_correct"] is False
        assert result["score_status"] == SCORE_INCORRECT

    def test_unparseable_response_is_unscorable(self, runner_question_row):
        backend = MockBackend(responses=["I cannot determine this."])
        row = _row_with_extracted_text(runner_question_row, "HTTPS")
        result = _make_runner(backend).run_one(row, sample_index=0)
        assert result["parsed_choice"] is None
        assert result["score_status"] == SCORE_UNSCORABLE

    def test_missing_extracted_text_column_raises(self, runner_question_row):
        import pytest
        with pytest.raises(KeyError, match="extracted_text"):
            _make_runner(MockBackend(responses=["C"])).run_one(runner_question_row, sample_index=0)

    def test_reused_extracted_text_is_persisted_on_the_row(self, runner_question_row):
        backend = MockBackend(responses=["C"])
        row = _row_with_extracted_text(runner_question_row, "HTTPS")
        result = _make_runner(backend).run_one(row, sample_index=0)
        assert result["reused_extracted_text"] == "HTTPS"


class TestExtractedTextSourceValidation:
    """Confirmed conformance gap: extracted_text's own producer identity
    (model/benchmark/method) was never validated -- a cross-model/cross-
    condition artifact would silently mix in. All three fields are
    OPTIONAL/additive: a row without them is unvalidated, exactly as
    before this existed."""

    def test_matching_source_model_is_accepted(self, runner_question_row):
        backend = MockBackend(responses=["C"])
        row = _row_with_extracted_text(runner_question_row, "HTTPS")
        row["extracted_text_source_model"] = "gpt-4.1-mini"  # matches MockBackend's default model_name
        result = _make_runner(backend).run_one(row, sample_index=0)
        assert result["parsed_choice"] == "C"

    def test_mismatched_source_model_is_rejected(self, runner_question_row):
        import pytest
        backend = MockBackend(responses=["C"])
        row = _row_with_extracted_text(runner_question_row, "HTTPS")
        row["extracted_text_source_model"] = "some-other-model-entirely"
        with pytest.raises(ValueError, match="extracted_text_source_model"):
            _make_runner(backend).run_one(row, sample_index=0)

    def test_mismatched_source_benchmark_is_rejected(self, runner_question_row):
        import pytest
        backend = MockBackend(responses=["C"])
        row = _row_with_extracted_text(runner_question_row, "HTTPS")
        row["extracted_text_source_benchmark"] = "arc_challenge"  # runner's own benchmark_name is "mmlu"
        with pytest.raises(ValueError, match="extracted_text_source_benchmark"):
            _make_runner(backend).run_one(row, sample_index=0)

    def test_mismatched_source_method_is_rejected(self, runner_question_row):
        import pytest
        backend = MockBackend(responses=["C"])
        row = _row_with_extracted_text(runner_question_row, "HTTPS")
        row["extracted_text_source_method"] = "reasoning_two_stage"
        with pytest.raises(ValueError, match="extracted_text_source_method"):
            _make_runner(backend).run_one(row, sample_index=0)

    def test_absent_source_fields_are_unvalidated_backward_compatible(self, runner_question_row):
        """A row from before this validation existed (no source_* fields
        at all) must keep working exactly as before."""
        backend = MockBackend(responses=["C"])
        row = _row_with_extracted_text(runner_question_row, "HTTPS")
        result = _make_runner(backend).run_one(row, sample_index=0)
        assert result["parsed_choice"] == "C"


class TestRunMatchingRotations:
    """Reruns ONLY the Stage-2 LLM match under every cyclic rotation of the
    options, reusing text_extraction's OWN per-rotation extracted text (one
    entry per rotation, already elicited under that exact rotation) --
    never re-eliciting Stage 1."""

    def test_makes_one_call_per_non_null_rotation_text(self, runner_question_row):
        backend = MockBackend(responses=["C", "A", "B", "D"])
        per_rotation_text = ["HTTPS", "FTP", "HTTP", "SMTP"]
        _make_runner(backend).run_matching_rotations(
            runner_question_row, per_rotation_text, sample_index=0
        )
        assert len(backend.requests_received) == 4

    def test_skips_a_rotation_with_no_extracted_text_no_call_made(self, runner_question_row):
        """A rotation where text_extraction's own Stage-1 call failed has
        nothing to match -- skip it (no call, no vote), don't crash."""
        backend = MockBackend(responses=["C", "B", "D"])
        per_rotation_text = ["HTTPS", None, "HTTP", "SMTP"]
        result = _make_runner(backend).run_matching_rotations(
            runner_question_row, per_rotation_text, sample_index=0
        )
        assert len(backend.requests_received) == 3
        per_rotation_choices = json.loads(result["per_rotation_choices_json"])
        assert per_rotation_choices[1] is None

    def test_persists_one_canonical_choice_per_rotation(self, runner_question_row):
        canonical = {"A": "FTP", "B": "HTTP", "C": "HTTPS", "D": "SMTP"}
        perms = [r.mapping for r in PermutationRunner._generate_rotations(canonical)]
        responses = []
        for perm in perms:
            for letter, text in perm.items():
                if text == "HTTPS":
                    responses.append(letter)
                    break

        backend = MockBackend(responses=responses)
        per_rotation_text = ["HTTPS"] * 4
        result = _make_runner(backend).run_matching_rotations(
            runner_question_row, per_rotation_text, sample_index=0
        )

        per_rotation_choices = json.loads(result["per_rotation_choices_json"])
        assert per_rotation_choices == ["C", "C", "C", "C"]
        assert result["parsed_choice"] == "C"
        assert result["is_correct"] is True

    def test_all_rotations_null_text_yields_no_parsed_choice(self, runner_question_row):
        backend = MockBackend(responses=[])
        result = _make_runner(backend).run_matching_rotations(
            runner_question_row, [None, None, None, None], sample_index=0
        )
        assert len(backend.requests_received) == 0
        assert result["parsed_choice"] is None
        per_rotation_choices = json.loads(result["per_rotation_choices_json"])
        assert per_rotation_choices == [None, None, None, None]

    def test_result_row_has_metadata(self, runner_question_row):
        backend = MockBackend(responses=["C", "C", "C", "C"])
        result = _make_runner(backend).run_matching_rotations(
            runner_question_row, ["HTTPS"] * 4, sample_index=0
        )
        assert result["run_id"] == "test_run_001"
        assert result["method_name"] == "visible_llm_matcher"
        assert result["split_name"] == "test"

    def test_trace_length_mismatch_fails_loudly(self, runner_question_row):
        """Confirmed conformance gap: a per_rotation_extracted_text list
        shorter than this question's n_choices previously silently
        truncated to the shorter length via zip(), rather than raising."""
        import pytest
        backend = MockBackend(responses=["C", "A", "B"])
        short_trace = ["only_three", "entries", "here"]  # runner_question_row has 4 options
        with pytest.raises(ValueError, match="expected 4"):
            _make_runner(backend).run_matching_rotations(
                runner_question_row, short_trace, sample_index=0
            )
        # Must fail BEFORE making any calls, not partway through.
        assert len(backend.requests_received) == 0

    def test_trace_length_too_long_also_fails_loudly(self, runner_question_row):
        import pytest
        backend = MockBackend(responses=["C"] * 5)
        long_trace = ["HTTPS", "FTP", "HTTP", "SMTP", "EXTRA"]
        with pytest.raises(ValueError, match="expected 4"):
            _make_runner(backend).run_matching_rotations(
                runner_question_row, long_trace, sample_index=0
            )

    def test_exact_matching_trace_length_still_works(self, runner_question_row):
        """Regression guard: the length check itself must not reject a
        correctly-sized trace."""
        backend = MockBackend(responses=["C", "A", "B", "D"])
        result = _make_runner(backend).run_matching_rotations(
            runner_question_row, ["HTTPS"] * 4, sample_index=0
        )
        assert len(backend.requests_received) == 4
        per_rotation_choices = json.loads(result["per_rotation_choices_json"])
        assert len(per_rotation_choices) == 4
