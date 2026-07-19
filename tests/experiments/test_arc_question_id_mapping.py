# tests/experiments/test_arc_question_id_mapping.py

import pytest

from experiments.visible_llm_matcher.arc_question_id_mapping import (
    RESOLVED_HASH_ONLY_MISMATCHES,
    UNRESOLVED_SCIENTIFIC_MISMATCHES,
    MismatchBlockedError,
    translate_tsp_arc_question_id,
)
from experiments.visible_llm_matcher.stage1_sources import KNOWN_3OPTION_ARC_QUESTION_IDS


class TestResolvedHashOnlyMismatches:
    def test_exactly_the_three_known_3option_ids(self):
        """The 3 hash-only mismatches are exactly the 3 questions already
        tracked as genuinely 3-option — same root cause (TSP's hash content
        string includes an empty trailing slot for a missing option;
        ChoiceBench's omits it entirely), so the two sets must coincide.
        """
        assert set(RESOLVED_HASH_ONLY_MISMATCHES) == KNOWN_3OPTION_ARC_QUESTION_IDS

    @pytest.mark.parametrize("tsp_id", sorted(RESOLVED_HASH_ONLY_MISMATCHES))
    def test_translates_to_the_mapped_choicebench_id(self, tsp_id):
        expected = RESOLVED_HASH_ONLY_MISMATCHES[tsp_id]
        assert translate_tsp_arc_question_id(tsp_id) == expected

    def test_mapped_ids_are_16_hex_chars(self):
        """Sanity check that both sides look like the sha256[:16]-hex ids
        both repos actually produce, not placeholder/typo values.
        """
        import re

        for tsp_id, cb_id in RESOLVED_HASH_ONLY_MISMATCHES.items():
            assert re.fullmatch(r"[0-9a-f]{16}", tsp_id)
            assert re.fullmatch(r"[0-9a-f]{16}", cb_id)
            assert tsp_id != cb_id  # the whole point: they differ


class TestUnresolvedScientificMismatches:
    def test_exactly_three_blocked_ids(self):
        assert len(UNRESOLVED_SCIENTIFIC_MISMATCHES) == 3

    @pytest.mark.parametrize("tsp_id", sorted(UNRESOLVED_SCIENTIFIC_MISMATCHES))
    def test_raises_mismatch_blocked_error(self, tsp_id):
        with pytest.raises(MismatchBlockedError, match="scientific content difference"):
            translate_tsp_arc_question_id(tsp_id)

    def test_error_names_the_missing_option(self):
        with pytest.raises(MismatchBlockedError, match="missing real option E"):
            translate_tsp_arc_question_id("f87cb129d9aa26c0")

    def test_disjoint_from_resolved_set(self):
        assert set(UNRESOLVED_SCIENTIFIC_MISMATCHES).isdisjoint(RESOLVED_HASH_ONLY_MISMATCHES)

    def test_disjoint_from_known_3option_set(self):
        """The blocked set is a different failure mode from the 3-option
        repair set — these 3 questions have 5 real options (one dropped by
        TSP), not 3.
        """
        assert set(UNRESOLVED_SCIENTIFIC_MISMATCHES).isdisjoint(KNOWN_3OPTION_ARC_QUESTION_IDS)


class TestIdentityForAlreadyMatchingIds:
    def test_unlisted_id_passes_through_unchanged(self):
        """The other 994/1000 ARC questions already hash identically in
        both repos — translate_tsp_arc_question_id must be a no-op for
        them, not force every id through a lookup table that would need to
        enumerate all 1000.
        """
        ordinary_id = "f87cb129d9aa26c1"  # not in either mapping
        assert translate_tsp_arc_question_id(ordinary_id) == ordinary_id
