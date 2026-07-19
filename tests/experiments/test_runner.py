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

from experiments.visible_llm_matcher.historical_protocol import build_option_matching_prompt
from experiments.visible_llm_matcher.runner import VisibleLlmMatcherRunner
from experiments.visible_llm_matcher.stage1_sources import KNOWN_3OPTION_ARC_QUESTION_IDS
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


_REPAIRED_ROW_TEMPLATE = {
    "subject": "arc_challenge",
    "question_text": "A toy truck rolls over a smooth surface...",
    "choice_a": "slower",
    "choice_b": "faster",
    "choice_c": "at the same speed",
    "choice_d": math.nan,
    "correct_option": "A",
}

def _repaired_row(question_id: str) -> dict:
    return {"question_id": question_id, **_REPAIRED_ROW_TEMPLATE}


def _repaired_lookup(question_id: str, free_text: str = "The truck will most likely roll slower.") -> dict:
    return {
        question_id: {
            "free_text_response": free_text,
            "_source_repo": "corrected_replacement",
            "_source_path": "<corrected replacement, supplied later>",
        }
    }


class TestContaminatedRowsAreRejected:
    """Production-path (defense in depth) counterpart to
    stage1_sources.py's own fail-closed contamination tests: even if a
    missing-4th-option row somehow reached the runner without going through
    load_and_validate_stage1() first, the runner itself must still refuse
    rather than silently render a 4-slot prompt for an unaudited question.
    """

    def test_missing_option_d_outside_the_audited_repair_set_raises(self, stage1_lookup):
        row = {
            "question_id": "some_other_arc_question_not_in_repair_set",
            "subject": "arc_challenge",
            "question_text": "Some other question with a missing option.",
            "choice_a": "a",
            "choice_b": "b",
            "choice_c": "c",
            "choice_d": math.nan,
            "correct_option": "A",
        }
        lookup = {
            row["question_id"]: {
                "free_text_response": "a",
                "_source_repo": "two-stage-prompting",
                "_source_path": "x",
            }
        }
        backend = MockBackend(responses=["A"])

        with pytest.raises(ValueError, match="not in KNOWN_3OPTION_ARC_QUESTION_IDS"):
            _make_runner(backend, lookup).run_one(row, sample_index=0)

        assert len(backend.requests_received) == 0

    def test_known_repair_id_with_empty_string_option_d_also_takes_repaired_path(self, stage1_lookup):
        """choice_d may arrive as an empty string rather than NaN depending
        on the CSV round-trip — both must be treated as "missing", not just
        NaN.
        """
        row = _repaired_row("79e8c959bbeb74a0")
        row["choice_d"] = ""
        lookup = _repaired_lookup("79e8c959bbeb74a0")
        backend = MockBackend(responses=["A"])

        result = _make_runner(backend, lookup).run_one(row, sample_index=0)

        assert "D." not in result["prompt"]


class TestRepairedThreeOptionRows:
    """Corrected replacement rows for the 3 audited-repair ARC questions
    must produce an A/B/C-only Stage-2 prompt — no D line, no "nan", no
    empty 4th option. This is the production-path behavior; the OLD
    D.nan-producing historical behavior is retained only as provenance
    evidence in test_historical_protocol.py and is no longer reachable from
    this runner for these question_ids.
    """

    @pytest.mark.parametrize("question_id", sorted(KNOWN_3OPTION_ARC_QUESTION_IDS))
    def test_produces_abc_only_prompt_for_every_known_repair_id(self, question_id):
        row = _repaired_row(question_id)
        lookup = _repaired_lookup(question_id)
        backend = MockBackend(responses=["A"])

        result = _make_runner(backend, lookup).run_one(row, sample_index=0)

        prompt = result["prompt"]
        assert "nan" not in prompt.lower()
        assert "D." not in prompt
        assert "D " not in prompt
        assert "A. slower" in prompt
        assert "B. faster" in prompt
        assert "C. at the same speed" in prompt
        assert "and three options" in prompt

    def test_repaired_prompt_uses_the_reused_free_text(self):
        row = _repaired_row("79e8c959bbeb74a0")
        lookup = _repaired_lookup("79e8c959bbeb74a0", free_text="corrected answer text")
        backend = MockBackend(responses=["A"])

        result = _make_runner(backend, lookup).run_one(row, sample_index=0)

        assert "Reference answer: corrected answer text" in result["prompt"]

    def test_repaired_row_still_makes_exactly_one_backend_call(self):
        row = _repaired_row("79e8c959bbeb74a0")
        lookup = _repaired_lookup("79e8c959bbeb74a0")
        backend = MockBackend(responses=["A"])

        _make_runner(backend, lookup).run_one(row, sample_index=0)

        assert len(backend.requests_received) == 1

    def test_repaired_row_is_still_scorable(self):
        row = _repaired_row("79e8c959bbeb74a0")
        lookup = _repaired_lookup("79e8c959bbeb74a0")
        backend = MockBackend(responses=["A"])

        result = _make_runner(backend, lookup).run_one(row, sample_index=0)

        assert result["parsed_choice"] == "A"
        assert result["is_correct"] is True


class TestOrdinaryFourOptionRowsStayByteFaithful:
    def test_matches_historical_protocol_output_directly(self, question_row, stage1_lookup):
        """Byte-compare the runner's rendered Stage-2 prompt against calling
        historical_protocol.build_option_matching_prompt directly with the
        same inputs — proves the ordinary (4-option) path was not touched
        by the variable-option repair.
        """
        backend = MockBackend(responses=["C"])
        result = _make_runner(backend, stage1_lookup).run_one(question_row, sample_index=0)

        expected = build_option_matching_prompt(
            template=open(_PROMPTS_DIR / "v1" / "option_matching.txt", encoding="utf-8").read(),
            question=question_row["question_text"],
            free_text="HTTPS",
            option_a="FTP",
            option_b="HTTP",
            option_c="HTTPS",
            option_d="SMTP",
        )
        assert result["prompt"] == expected

    def test_parser_does_not_restrict_to_real_option_letters(self, question_row, stage1_lookup):
        """Faithful reproduction of TSP's parser for an ordinary 4-option
        row (unaffected by the variable-option repair, which only changes
        Stage-2 *serialization* for the 3 known 3-option questions, not
        parsing behavior anywhere — see instruction not to modernize the
        historical parser).
        """
        backend = MockBackend(responses=["D"])
        result = _make_runner(backend, stage1_lookup).run_one(question_row, sample_index=0)

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
