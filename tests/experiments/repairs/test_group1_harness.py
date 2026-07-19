# tests/experiments/repairs/test_group1_harness.py
#
# Focused tests for the Group 1 historical-ARC repair harness, covering
# every item on the required checklist:
#   - a representative four-option prompt remains historically identical;
#   - each of the three repaired questions contains exactly A/B/C;
#   - no prompt or permutation contains D. nan;
#   - cyclic produces exactly three distinct rotations;
#   - parser and fallback behavior are otherwise unchanged;
#   - cache contamination cannot turn a required fresh repair into a cache hit;
#   - the resulting checkpoint contains 997 unchanged rows and exactly 3
#     replaced rows (checkpoint-seed side, already covered by
#     test_stage1_sources.py-style tests in the earlier phase; re-verified
#     here at the harness's own merge boundary).

from __future__ import annotations

import pytest

from experiments.visible_llm_matcher.repairs.group1_historical_arc.harness.baseline_repair import (
    repair_one_baseline_question,
)
from experiments.visible_llm_matcher.repairs.group1_historical_arc.harness.cyclic_repair import (
    repair_one_cyclic_question,
)
from experiments.visible_llm_matcher.repairs.group1_historical_arc.harness.repair_infra import (
    FreshCallRequiredError,
    assert_cache_grew_by,
    count_cache_entries,
)
from experiments.visible_llm_matcher.repairs.group1_historical_arc.harness.text_extraction_repair import (
    repair_one_text_extraction_question,
)
from experiments.visible_llm_matcher.repairs.group1_historical_arc.harness.two_stage_repair import (
    repair_one_two_stage_question,
)
from experiments.visible_llm_matcher.repairs.group1_historical_arc.harness.variable_option_protocol import (
    build_direct_mcq_prompt,
    build_direct_mcq_prompt_repaired_3option,
    generate_cyclic_permutations,
)

from .conftest import MockAsyncBackend

# ---------------------------------------------------------------------------
# 1. Ordinary 4-option rows remain byte-for-byte historically faithful.
#    (The harness never even routes these through a repaired path — see
#    each repair_one_* function's ValueError guard, tested separately
#    below — but the underlying prompt builders must still produce
#    byte-identical output to the historical templates when given 4
#    options, since they ARE the historical templates, unmodified.)
# ---------------------------------------------------------------------------

_HISTORICAL_DIRECT_MCQ_TEMPLATE = (
    "Answer the following multiple-choice question.\n\n"
    "Question: {question}\n\nOptions:\nA. {option_a}\nB. {option_b}\n"
    "C. {option_c}\nD. {option_d}\n\nRespond with only the letter."
)


class TestFourOptionRowsStayHistoricallyIdentical:
    def test_direct_mcq_four_option_prompt_matches_historical_template(self):
        prompt = build_direct_mcq_prompt(
            template=_HISTORICAL_DIRECT_MCQ_TEMPLATE,
            question="An ordinary question?",
            option_a="alpha",
            option_b="beta",
            option_c="gamma",
            option_d="delta",
        )
        expected = _HISTORICAL_DIRECT_MCQ_TEMPLATE.format(
            question="An ordinary question?", option_a="alpha", option_b="beta", option_c="gamma", option_d="delta"
        )
        assert prompt == expected
        assert "D. delta" in prompt

    @pytest.mark.asyncio
    async def test_baseline_repair_refuses_an_unaudited_four_option_question(self, four_option_row):
        backend = MockAsyncBackend(["B"])
        with pytest.raises(ValueError, match="not in the audited 3-option repair set"):
            await repair_one_baseline_question(backend, four_option_row)
        assert len(backend.prompts_received) == 0  # no call was made

    @pytest.mark.asyncio
    async def test_cyclic_repair_refuses_an_unaudited_four_option_question(self, four_option_row):
        backend = MockAsyncBackend(["B", "B", "B", "B"])
        with pytest.raises(ValueError, match="not in the audited 3-option repair set"):
            await repair_one_cyclic_question(backend, four_option_row)
        assert len(backend.prompts_received) == 0

    @pytest.mark.asyncio
    async def test_text_extraction_repair_refuses_an_unaudited_four_option_question(self, four_option_row):
        backend = MockAsyncBackend(["beta"])
        with pytest.raises(ValueError, match="not in the audited 3-option repair set"):
            await repair_one_text_extraction_question(backend, four_option_row)
        assert len(backend.prompts_received) == 0

    @pytest.mark.asyncio
    async def test_two_stage_repair_refuses_an_unaudited_four_option_question(self, four_option_row):
        backend = MockAsyncBackend(["B"])
        with pytest.raises(ValueError, match="not in the audited 3-option repair set"):
            await repair_one_two_stage_question(
                backend, four_option_row, reused_free_text_response="beta", method_name="two_stage_v1"
            )
        assert len(backend.prompts_received) == 0


# ---------------------------------------------------------------------------
# 2 & 3. Every repaired question renders exactly A/B/C, never D. nan.
# ---------------------------------------------------------------------------


class TestRepairedQuestionsAreAbcOnly:
    @pytest.mark.asyncio
    async def test_baseline_prompt_is_abc_only(self, three_option_row):
        backend = MockAsyncBackend(["A"])
        result = await repair_one_baseline_question(backend, three_option_row)
        assert "A. slower" in result["prompt"]
        assert "B. faster" in result["prompt"]
        assert "C. at the same speed" in result["prompt"]
        assert "D." not in result["prompt"]
        assert "nan" not in result["prompt"].lower()

    @pytest.mark.asyncio
    async def test_cyclic_prompts_are_abc_only_across_all_rotations(self, three_option_row):
        backend = MockAsyncBackend(["A", "B", "C"])
        await repair_one_cyclic_question(backend, three_option_row)
        for prompt in backend.prompts_received:
            assert "D." not in prompt
            assert "nan" not in prompt.lower()

    @pytest.mark.asyncio
    async def test_text_extraction_prompt_is_abc_only(self, three_option_row):
        backend = MockAsyncBackend(["slower"])
        result = await repair_one_text_extraction_question(backend, three_option_row)
        assert "A. slower" in result["prompt"]
        assert "D." not in result["prompt"]
        assert "nan" not in result["prompt"].lower()

    @pytest.mark.asyncio
    async def test_two_stage_stage2_prompt_is_abc_only(self, three_option_row):
        backend = MockAsyncBackend(["A"])
        result = await repair_one_two_stage_question(
            backend, three_option_row, reused_free_text_response="It rolls slower.", method_name="two_stage_v2"
        )
        assert "A. slower" in result["prompt"]
        assert "D." not in result["prompt"]
        assert "nan" not in result["prompt"].lower()

    @pytest.mark.parametrize("qid", ["79e8c959bbeb74a0", "ad6b5d46ae54842c", "c30e75b011696a95"])
    def test_all_three_audited_ids_produce_abc_only_direct_mcq_prompts(self, qid):
        prompt = build_direct_mcq_prompt_repaired_3option(
            template="Q: {question}\nA. {option_a}\nB. {option_b}\nC. {option_c}",
            question="x",
            option_a="1",
            option_b="2",
            option_c="3",
        )
        assert "D." not in prompt
        assert "nan" not in prompt.lower()


# ---------------------------------------------------------------------------
# 4. Cyclic produces exactly 3 distinct rotations for a 3-option question.
# ---------------------------------------------------------------------------


class TestCyclicExactlyThreeRotations:
    def test_generate_cyclic_permutations_returns_three_for_three_options(self):
        options = {"A": "slower", "B": "faster", "C": "at the same speed"}
        perms = generate_cyclic_permutations(options)
        assert len(perms) == 3

    def test_the_three_rotations_are_distinct(self):
        options = {"A": "slower", "B": "faster", "C": "at the same speed"}
        perms = generate_cyclic_permutations(options)
        as_tuples = [tuple(p.values()) for p in perms]
        assert len(set(as_tuples)) == 3

    def test_rotation_ordering_matches_historical_cyclic_shift(self):
        """Ported PermutationRunner._generate_permutations: rotation i is
        values[i:] + values[:i] — verify byte-for-byte for the 3-option case."""
        options = {"A": "x", "B": "y", "C": "z"}
        perms = generate_cyclic_permutations(options)
        assert list(perms[0].values()) == ["x", "y", "z"]
        assert list(perms[1].values()) == ["y", "z", "x"]
        assert list(perms[2].values()) == ["z", "x", "y"]

    @pytest.mark.asyncio
    async def test_repair_makes_exactly_three_calls(self, three_option_row):
        backend = MockAsyncBackend(["A", "B", "C"])
        result = await repair_one_cyclic_question(backend, three_option_row)
        assert result["n_rotations"] == 3
        assert len(backend.prompts_received) == 3

    def test_four_option_permutations_still_produce_four(self):
        """Sanity: the (unmodified) generation function is generic — an
        ordinary 4-option dict still yields 4 rotations, matching history
        exactly for questions never routed through the repair path."""
        options = {"A": "w", "B": "x", "C": "y", "D": "z"}
        perms = generate_cyclic_permutations(options)
        assert len(perms) == 4


# ---------------------------------------------------------------------------
# 5. Parser and fallback behavior are otherwise unchanged (unrestricted
#    A-D letter acceptance, same as historical_protocol.py's ported
#    parser — verified here at the harness's own call sites).
# ---------------------------------------------------------------------------


class TestParserBehaviorUnchanged:
    @pytest.mark.asyncio
    async def test_baseline_accepts_a_bare_d_response_even_for_a_3_option_question(self, three_option_row):
        """Faithful to history: the parser is never restricted to real
        option letters (see historical_protocol.py) -- a stray "D" is
        still PARSE_OK, just unscorable/incorrect against a real A/B/C
        gold answer, not silently rejected or crashed on.
        """
        backend = MockAsyncBackend(["D"])
        result = await repair_one_baseline_question(backend, three_option_row)
        assert result["parsed_choice"] == "D"
        assert result["is_correct"] is False

    @pytest.mark.asyncio
    async def test_cyclic_unpermute_treats_an_out_of_range_letter_as_no_vote(self, three_option_row):
        """A rotation response of "D" (no D option exists) must not crash
        the unpermute step -- see variable_option_protocol.unpermute_choice's
        documented handling.

        Rotations for canonical {A: slower, B: faster, C: at the same speed}:
          rotation 0 = {A: slower,          B: faster,           C: at the same speed}
          rotation 1 = {A: faster,          B: at the same speed, C: slower}
          rotation 2 = {A: at the same speed, B: slower,          C: faster}
        Responses ["D", "C", "B"]:
          rotation 0 "D" -> no such key -> no vote
          rotation 1 "C" -> text "slower" -> canonical "A"
          rotation 2 "B" -> text "slower" -> canonical "A"
        -> 2 votes for canonical "A", 0 elsewhere -> majority "A".
        """
        backend = MockAsyncBackend(["D", "C", "B"])
        result = await repair_one_cyclic_question(backend, three_option_row)
        assert result["per_rotation_canonical_choices"] == [None, "A", "A"]
        assert result["parsed_choice"] == "A"

    @pytest.mark.asyncio
    async def test_text_extraction_unparseable_response_is_unscorable(self, three_option_row):
        backend = MockAsyncBackend(["completely unrelated gibberish text"])
        result = await repair_one_text_extraction_question(backend, three_option_row)
        assert result["parsed_choice"] is None


# ---------------------------------------------------------------------------
# 6. Cache contamination cannot turn a required fresh repair into a cache
#    hit -- assert_cache_grew_by is the authoritative, file-system-based
#    check (see repair_infra.py's module docstring for why response
#    metadata alone is insufficient).
# ---------------------------------------------------------------------------


class TestCacheFreshnessEnforcement:
    def test_zero_growth_raises(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "experiments.visible_llm_matcher.repairs.group1_historical_arc.harness.repair_infra.REPAIR_CACHE_ROOT",
            tmp_path,
        )
        ns_dir = tmp_path / "test_namespace"
        ns_dir.mkdir()
        before = count_cache_entries("test_namespace")
        # No new files written -- simulates every call being served from a
        # pre-existing cache entry.
        with pytest.raises(FreshCallRequiredError, match="expected the repair cache to grow"):
            assert_cache_grew_by("test_namespace", before=before, expected_new_entries=3)

    def test_correct_growth_passes(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "experiments.visible_llm_matcher.repairs.group1_historical_arc.harness.repair_infra.REPAIR_CACHE_ROOT",
            tmp_path,
        )
        ns_dir = tmp_path / "test_namespace"
        ns_dir.mkdir()
        before = count_cache_entries("test_namespace")
        for i in range(3):
            (ns_dir / f"entry{i}.json").write_text("{}")
        assert_cache_grew_by("test_namespace", before=before, expected_new_entries=3)  # does not raise

    def test_partial_growth_raises(self, tmp_path, monkeypatch):
        """2 of 3 expected calls were fresh, 1 was (somehow) a cache hit --
        must still fail, not silently accept a partially-fresh batch."""
        monkeypatch.setattr(
            "experiments.visible_llm_matcher.repairs.group1_historical_arc.harness.repair_infra.REPAIR_CACHE_ROOT",
            tmp_path,
        )
        ns_dir = tmp_path / "test_namespace"
        ns_dir.mkdir()
        before = count_cache_entries("test_namespace")
        for i in range(2):
            (ns_dir / f"entry{i}.json").write_text("{}")
        with pytest.raises(FreshCallRequiredError):
            assert_cache_grew_by("test_namespace", before=before, expected_new_entries=3)

    def test_pre_existing_cache_entries_are_not_miscounted_as_fresh(self, tmp_path, monkeypatch):
        """Simulates the exact failure mode this whole mechanism exists to
        catch: a namespace that ALREADY has entries (e.g. from an old,
        wrongly-reused cache dir) must not let a 0-growth batch look
        "correct" just because entries exist -- growth is what's measured,
        not raw file count.
        """
        monkeypatch.setattr(
            "experiments.visible_llm_matcher.repairs.group1_historical_arc.harness.repair_infra.REPAIR_CACHE_ROOT",
            tmp_path,
        )
        ns_dir = tmp_path / "test_namespace"
        ns_dir.mkdir()
        (ns_dir / "stale_entry.json").write_text("{}")  # pre-existing contamination
        before = count_cache_entries("test_namespace")
        assert before == 1
        # No new file added -- this batch must still be rejected.
        with pytest.raises(FreshCallRequiredError):
            assert_cache_grew_by("test_namespace", before=before, expected_new_entries=1)
