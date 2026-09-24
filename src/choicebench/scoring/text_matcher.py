# src/choicebench/scoring/text_matcher.py

"""Shared exact -> containment -> cosine-similarity text-matching cascade.

Matches a free-text response (from a hidden-options stage-1 generation, or
a visible-options text_extraction call) back to one of a question's
canonical options. Used by semantic_matching_v1, text_extraction, and
visible_llm_matcher -- one cascade, one tie policy, not three ad-hoc
reimplementations.

Cascade:
  1. Exact match (normalized: lowercased, whitespace-collapsed).
  2. Unique substring containment (option text contained in the free text,
     or vice versa), only when the contained string has length >= 4 and
     exactly one option qualifies -- a shorter or ambiguous multi-option
     hit falls through to cosine rather than guessing.
  3. Cosine similarity between the free text and every option's text
     (via an embedding function), argmax over the threshold. Below
     threshold is unscorable (None), not a forced guess.

Any tie at any stage is broken via the shared canonical tie-break utility
(choicebench.scoring.tiebreak.resolve_tie) over each option's stable
source_index -- never display letter, never first-encountered order.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from choicebench.scoring.tiebreak import resolve_tie

MIN_CONTAINMENT_LEN = 4
COSINE_THRESHOLD = 0.30

EmbedFn = Callable[[list[str]], np.ndarray]


def _normalize(text: str) -> str:
    return " ".join(text.strip().lower().split())


def _resolve(label_to_source_index: dict[str, int], tied_letters: list[str], *,
             seed: int, benchmark_id: str, question_id: str, method_name: str) -> str:
    if len(tied_letters) == 1:
        return tied_letters[0]
    tied_ids = [label_to_source_index[letter] for letter in tied_letters]
    winning_id = resolve_tie(
        seed=seed, benchmark_id=benchmark_id, question_id=question_id,
        method_name=method_name, tied_canonical_ids=tied_ids,
    )
    id_to_label = {v: k for k, v in label_to_source_index.items()}
    return id_to_label[winning_id]


def _exact_match(free_norm: str, options_norm: dict[str, str]) -> list[str]:
    return [letter for letter, text in options_norm.items() if text == free_norm]


def _containment_match(free_norm: str, options_norm: dict[str, str]) -> list[str]:
    hits = []
    for letter, text in options_norm.items():
        if len(text) < MIN_CONTAINMENT_LEN:
            continue
        if text in free_norm or free_norm in text:
            if len(free_norm) >= MIN_CONTAINMENT_LEN or text in free_norm:
                hits.append(letter)
    return hits


def _cosine_match(
        free_norm: str,
        options_norm: dict[str, str],
        embed_fn: EmbedFn,
) -> list[str]:
    labels = list(options_norm.keys())
    texts = [free_norm] + [options_norm[letter] for letter in labels]
    vectors = np.asarray(embed_fn(texts), dtype=np.float64)
    free_vec, option_vecs = vectors[0], vectors[1:]

    free_norm = np.linalg.norm(free_vec)
    option_norms = np.linalg.norm(option_vecs, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        sims = (option_vecs @ free_vec) / (option_norms * free_norm)
    sims = np.nan_to_num(sims, nan=-1.0)

    best = float(np.max(sims))
    if best < COSINE_THRESHOLD:
        return []
    return [labels[i] for i, s in enumerate(sims) if s == best]


_DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"
_default_model = None  # lazy-loaded singleton, avoids importing sentence-transformers at module load


def default_embed_fn(texts: list[str]) -> np.ndarray:
    """Real embedder: sentence-transformers all-MiniLM-L6-v2.

    Requires the optional 'matching' extra (pip install -e ".[matching]").
    Imported lazily so this module stays importable -- and match_text_to_options
    stays fully unit-testable with a fake embed_fn -- without the dependency
    installed.
    """
    global _default_model
    if _default_model is None:
        from sentence_transformers import SentenceTransformer
        _default_model = SentenceTransformer(_DEFAULT_MODEL_NAME)
    return np.asarray(_default_model.encode(texts))


def match_text_to_options(
        free_text: str | None,
        options: dict[str, str],
        *,
        embed_fn: EmbedFn | None,
        label_to_source_index: dict[str, int],
        seed: int,
        benchmark_id: str,
        question_id: str,
        method_name: str,
) -> str | None:
    """Match free_text to one of options's canonical letters, or None.

    Args:
        free_text: The free-form response to match.
        options: This question's canonical letter->text mapping.
        embed_fn: Embeds a list of texts to a (N, D) array for the cosine
            stage; pass None to skip cosine entirely (e.g. in tests that
            only exercise exact/containment).
        label_to_source_index, seed, benchmark_id, question_id, method_name:
            Tie-break inputs, forwarded to resolve_tie on any tie.

    Returns:
        The matched canonical letter, or None if nothing matched (exact/
        containment found nothing and cosine's best score was below
        COSINE_THRESHOLD, or embed_fn was None and nothing matched earlier).
    """
    tiebreak_kwargs = dict(seed=seed, benchmark_id=benchmark_id,
                            question_id=question_id, method_name=method_name)

    if not free_text or not free_text.strip():
        return None

    free_norm = _normalize(free_text)
    options_norm = {letter: _normalize(text) for letter, text in options.items()}

    exact_hits = _exact_match(free_norm, options_norm)
    if exact_hits:
        return _resolve(label_to_source_index, exact_hits, **tiebreak_kwargs)

    containment_hits = _containment_match(free_norm, options_norm)
    if len(containment_hits) == 1:
        return _resolve(label_to_source_index, containment_hits, **tiebreak_kwargs)

    if embed_fn is None:
        return None

    cosine_hits = _cosine_match(free_norm, options_norm, embed_fn)
    if not cosine_hits:
        return None
    return _resolve(label_to_source_index, cosine_hits, **tiebreak_kwargs)
