# tests/scoring/test_text_matcher.py

"""Tests for the shared exact->containment->cosine text-matching cascade
used by semantic_matching_v1, text_extraction, and visible_llm_matcher.

The cosine stage's embedder is injectable so these tests run fully offline
and fast -- no real sentence-transformers model load in the unit suite.
"""

import numpy as np
import pytest

from choicebench.scoring.text_matcher import match_text_to_options


def _fake_embedder_factory(vectors: dict[str, list[float]]):
    """Returns an embed_fn(texts: list[str]) -> np.ndarray stub keyed by exact text."""
    def embed_fn(texts: list[str]) -> np.ndarray:
        return np.array([vectors[t] for t in texts], dtype=np.float64)
    return embed_fn


OPTIONS = {"A": "Paris", "B": "London", "C": "Berlin", "D": "Madrid"}
LABEL_TO_ID = {"A": 10, "B": 20, "C": 30, "D": 40}
TIEBREAK_KWARGS = dict(
    label_to_source_index=LABEL_TO_ID, seed=42, benchmark_id="mmlu",
    question_id="q1", method_name="semantic_matching_v1",
)


class TestExactMatch:
    def test_exact_match_wins_without_needing_an_embedder(self):
        result = match_text_to_options("Paris", OPTIONS, embed_fn=None, **TIEBREAK_KWARGS)
        assert result == "A"

    def test_exact_match_is_case_and_whitespace_insensitive(self):
        result = match_text_to_options("  paris  ", OPTIONS, embed_fn=None, **TIEBREAK_KWARGS)
        assert result == "A"


class TestContainmentMatch:
    def test_unique_containment_wins(self):
        result = match_text_to_options(
            "The answer is Paris, the capital of France.", OPTIONS, embed_fn=None, **TIEBREAK_KWARGS
        )
        assert result == "A"

    def test_containment_below_min_length_is_ignored(self):
        # "options" here are shorter than the min containment length (4);
        # a spurious short substring hit must not win.
        options = {"A": "Ab", "B": "Cd"}
        label_to_id = {"A": 1, "B": 2}
        result = match_text_to_options(
            "Ab is not a real containment match given the length floor",
            options, embed_fn=None,
            label_to_source_index=label_to_id, seed=1, benchmark_id="mmlu",
            question_id="q2", method_name="text_extraction",
        )
        # Falls through to cosine (embed_fn=None) -> None, since containment
        # was correctly suppressed rather than incorrectly firing on "Ab".
        assert result is None

    def test_non_unique_containment_falls_through_to_cosine(self):
        # Both "Paris" and "Pari" (contrived) would match -- containment
        # must not fire on an ambiguous multi-option hit.
        options = {"A": "Paris", "B": "Paris-adjacent"}
        label_to_id = {"A": 1, "B": 2}
        embed_fn = _fake_embedder_factory({
            "the city near paris-adjacent region": [1.0, 0.0],
            "paris": [0.0, 1.0],
            "paris-adjacent": [1.0, 0.0],
        })
        result = match_text_to_options(
            "the city near paris-adjacent region", options, embed_fn=embed_fn,
            label_to_source_index=label_to_id, seed=1, benchmark_id="mmlu",
            question_id="q3", method_name="text_extraction",
        )
        assert result == "B"


class TestCosineMatch:
    def test_cosine_argmax_above_threshold_wins(self):
        embed_fn = _fake_embedder_factory({
            "the city of light": [0.9, 0.1, 0.0, 0.0],
            "paris": [0.95, 0.05, 0.0, 0.0],
            "london": [0.0, 0.9, 0.1, 0.0],
            "berlin": [0.0, 0.0, 0.9, 0.1],
            "madrid": [0.0, 0.0, 0.0, 0.9],
        })
        result = match_text_to_options(
            "the city of light", OPTIONS, embed_fn=embed_fn, **TIEBREAK_KWARGS
        )
        assert result == "A"

    def test_below_threshold_is_unscorable(self):
        embed_fn = _fake_embedder_factory({
            "completely unrelated free text": [1.0, 0.0, 0.0, 0.0],
            "paris": [0.0, 1.0, 0.0, 0.0],
            "london": [0.0, 0.0, 1.0, 0.0],
            "berlin": [0.0, 0.0, 0.0, 1.0],
            "madrid": [-1.0, 0.0, 0.0, 0.0],
        })
        result = match_text_to_options(
            "completely unrelated free text", OPTIONS, embed_fn=embed_fn, **TIEBREAK_KWARGS
        )
        assert result is None

    def test_cosine_tie_routes_through_shared_tiebreak_utility(self):
        from choicebench.scoring.tiebreak import resolve_tie

        embed_fn = _fake_embedder_factory({
            "ambiguous free text": [1.0, 0.0],
            "paris": [1.0, 0.0],
            "london": [1.0, 0.0],
            "berlin": [0.0, 1.0],
            "madrid": [0.0, -1.0],
        })
        result = match_text_to_options(
            "ambiguous free text", OPTIONS, embed_fn=embed_fn, **TIEBREAK_KWARGS
        )
        expected_id = resolve_tie(
            seed=42, benchmark_id="mmlu", question_id="q1", method_name="semantic_matching_v1",
            tied_canonical_ids=[LABEL_TO_ID["A"], LABEL_TO_ID["B"]],
        )
        expected = {v: k for k, v in LABEL_TO_ID.items()}[expected_id]
        assert result == expected


class TestNoFreeText:
    def test_empty_free_text_is_unscorable(self):
        result = match_text_to_options("", OPTIONS, embed_fn=None, **TIEBREAK_KWARGS)
        assert result is None

    def test_none_free_text_is_unscorable(self):
        result = match_text_to_options(None, OPTIONS, embed_fn=None, **TIEBREAK_KWARGS)
        assert result is None
