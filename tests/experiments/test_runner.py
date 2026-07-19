# tests/experiments/test_runner.py
#
# Proves the only experimental delta introduced by VisibleLlmMatcherRunner,
# relative to the two existing sibling cells, is the matcher mechanism:
#
#   text_extraction (options visible + embedding matcher):
#     1 backend call (Stage 1 only) -> offline embedding match, no 2nd call.
#   two_stage / two_prompt (options hidden + LLM matcher):
#     2 backend calls (Stage 1 hidden, then Stage 2 LLM match).
#   VisibleLlmMatcherRunner (options visible + LLM matcher, the missing cell):
#     0 Stage-1 backend calls (free text is REUSED, injected via
#     stage1_lookup) + exactly 1 backend call (Stage 2 LLM match) — i.e.
#     text_extraction's own Stage-1 output fed through two_stage's own
#     Stage-2 protocol, with nothing else added.

import math
from pathlib import Path

import pytest

from choicebench.scoring.types import SCORE_CORRECT, SCORE_INCORRECT, SCORE_UNSCORABLE

from experiments.visible_llm_matcher.runner import VisibleLlmMatcherRunner
from tests.runners.conftest import MockBackend

_PROMPTS_DIR = Path(__file__).resolve().parents[2] / "experiments" / "visible_llm_matcher" / "prompts"


@pytest.fixture
def question_row() -> dict:
    return {
        "question_id": "4865890d7f0efae8",
        "subject": "computer_security",
        "question_text": "Which protocol is primarily used to securely browse websites?",
        "choice_a": "FTP",
        "choice_b": "HTTP",
        "choice_c": "HTTPS",
        "choice_d": "SMTP",
        "correct_option": "C",
    }


@pytest.fixture
def stage1_lookup() -> dict:
    """Stands in for the output of stage1_sources.load_and_validate_stage1(),
    reshaped to question_id -> row dict, exactly as VisibleLlmMatcherRunner
    expects. free_text_response is what a REUSED text_extraction Stage 1
    would have produced — this fixture never calls a backend to get it.
    """
    return {
        "4865890d7f0efae8": {
            "free_text_response": "HTTPS",
            "_source_repo": "two-stage-prompting",
            "_source_path": "paper_results/eval_ready/paper_api_main/20260603_154649_text_extraction_gpt-4.1-mini_mmlu.csv",
        }
    }


def _make_runner(backend, stage1_lookup):
    return VisibleLlmMatcherRunner(
        backend=backend,
        method_name="visible_llm_matcher",
        split_name="robustness",
        prompt_version="v1",
        prompts_dir=_PROMPTS_DIR,
        run_id="test_run_001",
        stage1_lookup=stage1_lookup,
    )


class TestExactlyOneBackendCall:
    def test_single_call_not_two(self, question_row, stage1_lookup):
        """Unlike TwoStageRunner (2 calls), this runner must make exactly 1
        — Stage 1 is injected, not generated.
        """
        backend = MockBackend(responses=["C"])
        _make_runner(backend, stage1_lookup).run_one(question_row, sample_index=0)

        assert len(backend.requests_received) == 1

    def test_the_one_call_is_the_matching_prompt_not_a_free_text_prompt(self, question_row, stage1_lookup):
        backend = MockBackend(responses=["C"])
        _make_runner(backend, stage1_lookup).run_one(question_row, sample_index=0)

        sent_prompt = backend.requests_received[0]
        assert "Reference answer: HTTPS" in sent_prompt
        assert "Select the option that best matches" in sent_prompt


class TestStage1IsInjectedNotGenerated:
    def test_free_text_response_comes_from_lookup(self, question_row, stage1_lookup):
        backend = MockBackend(responses=["C"])
        result = _make_runner(backend, stage1_lookup).run_one(question_row, sample_index=0)

        assert result["free_text_response"] == "HTTPS"

    def test_provenance_columns_are_carried_through(self, question_row, stage1_lookup):
        backend = MockBackend(responses=["C"])
        result = _make_runner(backend, stage1_lookup).run_one(question_row, sample_index=0)

        assert result["free_text_source_repo"] == "two-stage-prompting"
        assert "text_extraction" in result["free_text_source_path"]

    def test_missing_stage1_row_raises_rather_than_silently_generating(self, question_row, stage1_lookup):
        empty_lookup: dict = {}
        backend = MockBackend(responses=["C"])

        with pytest.raises(KeyError, match="No reused Stage-1 row"):
            _make_runner(backend, empty_lookup).run_one(question_row, sample_index=0)

        # And critically: no backend call was made trying to "fill the gap".
        assert len(backend.requests_received) == 0


class TestScoringMatchesLlmMatcherOutcome:
    def test_correct(self, question_row, stage1_lookup):
        backend = MockBackend(responses=["C"])
        result = _make_runner(backend, stage1_lookup).run_one(question_row, sample_index=0)

        assert result["parsed_choice"] == "C"
        assert result["is_correct"] is True
        assert result["score_status"] == SCORE_CORRECT

    def test_incorrect(self, question_row, stage1_lookup):
        backend = MockBackend(responses=["A"])
        result = _make_runner(backend, stage1_lookup).run_one(question_row, sample_index=0)

        assert result["is_correct"] is False
        assert result["score_status"] == SCORE_INCORRECT

    def test_unparseable_llm_matcher_response_is_unscorable(self, question_row, stage1_lookup):
        backend = MockBackend(responses=["I think it might be one of those"])
        result = _make_runner(backend, stage1_lookup).run_one(question_row, sample_index=0)

        assert result["parsed_choice"] is None
        assert result["score_status"] == SCORE_UNSCORABLE

    def test_backend_failure_yields_no_parse_or_score(self, question_row, stage1_lookup):
        from choicebench.clients.types import ProviderTimeoutError

        backend = MockBackend(responses=[ProviderTimeoutError("Request timed out.")])
        result = _make_runner(backend, stage1_lookup).run_one(question_row, sample_index=0)

        assert result["parsed_choice"] is None
        assert result["is_correct"] is None
        assert result["score_status"] is None


class TestHistoricalNanSerializationIsPreserved:
    def test_missing_option_d_renders_literal_nan_not_dropped(self, stage1_lookup):
        """The sibling cells' own Stage-2/Stage-1 protocol renders a missing
        4th option as literal "D. nan" (verified against real historical
        data in test_historical_protocol.py). VisibleLlmMatcherRunner must
        reproduce that, not "fix" it — see historical_protocol.py and
        stage1_sources.py module docstrings for why.
        """
        row = {
            "question_id": "79e8c959bbeb74a0",
            "subject": "arc_challenge",
            "question_text": "A toy truck rolls over a smooth surface...",
            "choice_a": "slower",
            "choice_b": "faster",
            "choice_c": "at the same speed",
            "choice_d": math.nan,
            "correct_option": "A",
        }
        lookup = {
            "79e8c959bbeb74a0": {
                "free_text_response": "The truck will most likely roll slower.",
                "_source_repo": "two-stage-prompting",
                "_source_path": "corrected_replacement_placeholder",
            }
        }
        backend = MockBackend(responses=["A"])
        result = _make_runner(backend, lookup).run_one(row, sample_index=0)

        assert "D. nan" in result["prompt"]

    def test_parser_does_not_restrict_to_real_option_letters(self, stage1_lookup):
        """Faithful reproduction of TSP's parser: even for this 3-option
        question, a bare "D" in the response is still accepted as
        final_choice == "D" (an unscorable-against-gold but PARSE_OK
        result) — matching two_prompt's own historical Stage-2 parsing,
        not choicebench.parsing.parser's stricter real-letters-only
        behavior.
        """
        row = {
            "question_id": "79e8c959bbeb74a0",
            "subject": "arc_challenge",
            "question_text": "A toy truck rolls over a smooth surface...",
            "choice_a": "slower",
            "choice_b": "faster",
            "choice_c": "at the same speed",
            "choice_d": math.nan,
            "correct_option": "A",
        }
        lookup = {
            "79e8c959bbeb74a0": {
                "free_text_response": "The truck will most likely roll slower.",
                "_source_repo": "two-stage-prompting",
                "_source_path": "x",
            }
        }
        backend = MockBackend(responses=["D"])
        result = _make_runner(backend, lookup).run_one(row, sample_index=0)

        assert result["parsed_choice"] == "D"
        assert result["is_correct"] is False


class TestIsolationFromSiblingCells:
    def test_reused_free_text_is_bitwise_identical_to_text_extraction_input(
        self, question_row, stage1_lookup
    ):
        """The free text this runner sends into Stage 2 is exactly what was
        stored for text_extraction (options visible + embedding matcher) —
        this test re-asserts that identity explicitly as a guard against a
        future edit accidentally normalizing/mutating it before use.
        """
        backend = MockBackend(responses=["C"])
        result = _make_runner(backend, stage1_lookup).run_one(question_row, sample_index=0)

        assert result["free_text_response"] == stage1_lookup["4865890d7f0efae8"]["free_text_response"]

    def test_stage2_prompt_matches_two_stage_option_matching_template_shape(
        self, question_row, stage1_lookup
    ):
        """The rendered Stage-2 prompt must contain every structural anchor
        of the historical option_matching.txt template (question, reference
        answer, all four static option lines, and the fixed instruction
        text) — i.e. this is textually the same Stage-2 protocol as
        two_prompt's own Stage 2, not a rephrased or reformatted variant.
        """
        backend = MockBackend(responses=["C"])
        result = _make_runner(backend, stage1_lookup).run_one(question_row, sample_index=0)

        prompt = result["prompt"]
        assert "You are given a question, a reference answer, and four options." in prompt
        assert "Question: Which protocol is primarily used to securely browse websites?" in prompt
        assert "Reference answer: HTTPS" in prompt
        assert "A. FTP" in prompt
        assert "B. HTTP" in prompt
        assert "C. HTTPS" in prompt
        assert "D. SMTP" in prompt
        assert "Respond with only the letter." in prompt
