"""Tests for the deterministic semantic-rematch engine (methodology lock #6).

Ported from two-stage-prompting@b8e784f3eb5d2a727a97eb675140b383a34584fa's
src/twoprompt/parsing/text_matcher.py::match_text_to_options. All fixtures
here are synthetic/disposable -- per the session's scope boundary, the real
authoritative semantic-rematch manifest/bundle content is never read or
executed by these tests.
"""

from __future__ import annotations

from final_paper_analysis.semantic_rematch import (
    THREE_OPTION_QUESTION_IDS,
    match_text_to_options,
    normalize_text,
    options_for_question,
)

_FOUR_OPTIONS = {
    "A": "Photosynthesis",
    "B": "Respiration",
    "C": "Fermentation",
    "D": "Osmosis",
}


def test_normalize_text_lowercases_strips_punctuation_and_collapses_whitespace():
    assert normalize_text("  The Answer, Is: B!  ") == "the answer is b"


def test_normalize_text_handles_none_and_nan():
    assert normalize_text(None) == ""
    assert normalize_text(float("nan")) == ""


def test_exact_match_wins_without_touching_embedding_model(monkeypatch):
    def _boom():
        raise AssertionError("embedding model must not be loaded for an exact match")

    monkeypatch.setattr(
        "final_paper_analysis.semantic_rematch._get_embed_model", _boom
    )
    result = match_text_to_options("respiration", _FOUR_OPTIONS)
    assert result.label == "B"
    assert result.rule == "exact"
    assert result.score == 1.0
    assert result.scores_by_option == {"A": 0.0, "B": 1.0, "C": 0.0, "D": 0.0}


def test_unambiguous_containment_match(monkeypatch):
    def _boom():
        raise AssertionError("embedding model must not be loaded for a containment match")

    monkeypatch.setattr(
        "final_paper_analysis.semantic_rematch._get_embed_model", _boom
    )
    result = match_text_to_options("The answer is Fermentation, definitely.", _FOUR_OPTIONS)
    assert result.label == "C"
    assert result.rule == "containment"
    assert result.score == 0.95


def test_containment_below_minimum_length_defers_to_embedding(monkeypatch):
    # "the"/"a"/"an" (all <4 chars) would containment-match if contained in
    # the answer, but the >=4 char guard must forbid all three and fall
    # through to embedding. None of these options exact-match the answer.
    options = {"A": "the", "B": "a", "C": "an", "D": "some option d text"}
    calls = {"count": 0}

    class _StubModel:
        def encode(self, texts, normalize_embeddings, show_progress_bar):
            calls["count"] += 1
            import numpy as np
            # answer matches nothing well; scores below threshold for all.
            return np.array([[1.0, 0.0]] + [[0.0, 1.0]] * len(options))

    monkeypatch.setattr(
        "final_paper_analysis.semantic_rematch._get_embed_model", lambda: _StubModel()
    )
    result = match_text_to_options("it is the one true answer", options)
    assert calls["count"] == 1, "must have fallen through to the embedding stage"
    assert result.rule == "unmatched"


def test_ambiguous_containment_two_hits_defers_to_embedding(monkeypatch):
    # Both "cats" and "category" (each >=4 chars) are contained in the
    # answer -- ambiguous containment (>1 hit) must defer to embedding.
    options = {"A": "cats", "B": "category", "C": "dog", "D": "canine"}
    calls = {"count": 0}

    class _StubModel:
        def encode(self, texts, normalize_embeddings, show_progress_bar):
            calls["count"] += 1
            import numpy as np
            return np.array([[1.0, 0.0]] + [[0.0, 1.0]] * len(options))

    monkeypatch.setattr(
        "final_paper_analysis.semantic_rematch._get_embed_model", lambda: _StubModel()
    )
    result = match_text_to_options("cats or category, hard to say", options)
    assert calls["count"] == 1
    assert result.rule == "unmatched"


def test_embedding_threshold_boundary_at_exactly_030_matches(monkeypatch):
    import numpy as np

    class _StubModel:
        def encode(self, texts, normalize_embeddings, show_progress_bar):
            # answer_emb . option_emb dot products: A=0.30 exactly, others lower.
            return np.array([
                [1.0, 0.0, 0.0],  # answer
                [0.30, 0.0, 0.0],  # A -> dot = 0.30
                [0.10, 0.0, 0.0],  # B
                [0.05, 0.0, 0.0],  # C
                [0.0, 0.0, 0.0],  # D
            ])

    monkeypatch.setattr(
        "final_paper_analysis.semantic_rematch._get_embed_model", lambda: _StubModel()
    )
    result = match_text_to_options("some unrelated ramble about nothing", _FOUR_OPTIONS)
    assert result.rule == "embedding"
    assert result.label == "A"
    assert result.score == 0.30


def test_embedding_just_below_threshold_is_unmatched(monkeypatch):
    import numpy as np

    class _StubModel:
        def encode(self, texts, normalize_embeddings, show_progress_bar):
            return np.array([
                [1.0, 0.0],
                [0.29, 0.0],
                [0.10, 0.0],
                [0.05, 0.0],
                [0.0, 0.0],
            ])

    monkeypatch.setattr(
        "final_paper_analysis.semantic_rematch._get_embed_model", lambda: _StubModel()
    )
    result = match_text_to_options("some unrelated ramble", _FOUR_OPTIONS)
    assert result.rule == "unmatched"
    assert result.label is None
    assert result.score == 0.29


def test_embedding_tie_resolves_to_earliest_lettered_option(monkeypatch):
    import numpy as np

    class _StubModel:
        def encode(self, texts, normalize_embeddings, show_progress_bar):
            # B and C tie at the max score (0.5); A and D are lower.
            return np.array([
                [1.0, 0.0],
                [0.2, 0.0],  # A
                [0.5, 0.0],  # B
                [0.5, 0.0],  # C
                [0.1, 0.0],  # D
            ])

    monkeypatch.setattr(
        "final_paper_analysis.semantic_rematch._get_embed_model", lambda: _StubModel()
    )
    result = match_text_to_options("ambiguous ramble", _FOUR_OPTIONS)
    assert result.rule == "embedding"
    assert result.label == "B", "ties must resolve to the earliest lettered option"


def test_empty_or_missing_answer_is_unmatched_without_loading_model(monkeypatch):
    def _boom():
        raise AssertionError("embedding model must not be loaded for empty input")

    monkeypatch.setattr(
        "final_paper_analysis.semantic_rematch._get_embed_model", _boom
    )
    for missing in (None, "", "   ", float("nan")):
        result = match_text_to_options(missing, _FOUR_OPTIONS)
        assert result.label is None
        assert result.rule == "unmatched"
        assert result.score == 0.0


def test_three_audited_arc_ids_are_recorded_exactly():
    assert THREE_OPTION_QUESTION_IDS == frozenset(
        {"79e8c959bbeb74a0", "ad6b5d46ae54842c", "c30e75b011696a95"}
    )


def test_options_for_question_drops_option_d_for_audited_three_option_ids():
    for qid in THREE_OPTION_QUESTION_IDS:
        restricted = options_for_question(qid, _FOUR_OPTIONS)
        assert set(restricted) == {"A", "B", "C"}
        assert restricted == {"A": "Photosynthesis", "B": "Respiration", "C": "Fermentation"}


def test_options_for_question_keeps_all_four_for_other_questions():
    restricted = options_for_question("some_other_question_id", _FOUR_OPTIONS)
    assert set(restricted) == {"A", "B", "C", "D"}


def test_real_embedding_model_matches_a_genuine_paraphrase():
    """Real end-to-end check (no stub): a paraphrase that shares no exact
    words with any option must still resolve via the real all-MiniLM-L6-v2
    model's cosine similarity, not a mock."""
    options = {
        "A": "Photosynthesis",
        "B": "Cellular respiration",
        "C": "Fermentation",
        "D": "Osmosis",
    }
    result = match_text_to_options(
        "Plants make food from sunlight through a green pigment in their "
        "leaves called chlorophyll.",
        options,
    )
    assert result.rule == "embedding"
    assert result.label == "A"
    assert result.score >= 0.30


def test_three_option_restriction_prevents_a_spurious_option_d_match(monkeypatch):
    def _boom():
        raise AssertionError("must resolve by exact match on C, never reach embedding")

    monkeypatch.setattr(
        "final_paper_analysis.semantic_rematch._get_embed_model", _boom
    )
    qid = next(iter(THREE_OPTION_QUESTION_IDS))
    options = options_for_question(qid, {"A": "one", "B": "two", "C": "three", "D": "three"})
    # Without the restriction, "three" would exact-match both C and D (dict
    # iteration would return C first anyway since it comes first) -- but the
    # point of the restriction is D is never even a candidate for these rows.
    result = match_text_to_options("three", options)
    assert result.label == "C"
    assert "D" not in result.scores_by_option
