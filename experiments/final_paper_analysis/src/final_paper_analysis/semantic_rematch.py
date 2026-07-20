"""Deterministic port of the historical semantic-rematch matcher.

Ported byte-for-byte in behavior from
two-stage-prompting@b8e784f3eb5d2a727a97eb675140b383a34584fa's
src/twoprompt/parsing/text_matcher.py::match_text_to_options, per
METHODOLOGY_LOCK.md #6. Matching priority is fixed by that historical
implementation and must not be altered:

1. Exact match after normalization.
2. Unambiguous containment (>=4 char guard on the shorter side; exactly one
   hit, otherwise defer to embedding).
3. sentence-transformers/all-MiniLM-L6-v2 cosine similarity, threshold 0.30,
   ties resolved to the first-encountered (earliest lettered) option.

Below threshold (or no answer text at all) yields an unmatched result, which
the caller maps to parse_status=parse_missing / score_status=score_unscorable.
This module makes zero model/API calls except the local embedding model load,
and only for rows that reach rule 3.
"""

from __future__ import annotations

import math
import string
from dataclasses import dataclass
from typing import Mapping

import numpy as np

_SEMANTIC_THRESHOLD = 0.30
_MIN_CONTAINMENT_LEN = 4
_EMBED_MODEL_NAME = "all-MiniLM-L6-v2"

# The 3 genuine three-option ARC-Challenge questions (methodology lock #2/#3):
# choice_d is a structurally empty string for these, never a real 4th option.
THREE_OPTION_QUESTION_IDS = frozenset(
    {"79e8c959bbeb74a0", "ad6b5d46ae54842c", "c30e75b011696a95"}
)

_embed_model = None


def _get_embed_model():
    global _embed_model
    if _embed_model is None:
        from sentence_transformers import SentenceTransformer

        _embed_model = SentenceTransformer(_EMBED_MODEL_NAME)
    return _embed_model


@dataclass(frozen=True)
class SemanticMatchResult:
    label: str | None
    option_text: str | None
    score: float
    scores_by_option: Mapping[str, float]
    rule: str  # "exact" | "containment" | "embedding" | "unmatched"


def normalize_text(text: object) -> str:
    if text is None:
        return ""
    if isinstance(text, float) and math.isnan(text):
        return ""
    if not isinstance(text, str):
        text = str(text)
    text = text.lower().strip()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return " ".join(text.split())


def options_for_question(question_id: str, choices: Mapping[str, str]) -> dict[str, str]:
    """Restrict the option set to A/B/C for the 3 genuine three-option ARC
    questions, per the task's explicit instruction to match only against
    A/B/C for those rows (not merely rely on an empty choice_d never
    winning)."""
    if question_id in THREE_OPTION_QUESTION_IDS:
        return {k: v for k, v in choices.items() if k in ("A", "B", "C")}
    return dict(choices)


def match_text_to_options(
    answer_text: str | None, options: Mapping[str, str]
) -> SemanticMatchResult:
    if not isinstance(answer_text, str) or not answer_text.strip():
        return SemanticMatchResult(
            label=None,
            option_text=None,
            score=0.0,
            scores_by_option={k: 0.0 for k in options},
            rule="unmatched",
        )

    norm_answer = normalize_text(answer_text)

    # 1. Exact match.
    for letter, opt_text in options.items():
        if normalize_text(opt_text) == norm_answer:
            scores = {k: (1.0 if k == letter else 0.0) for k in options}
            return SemanticMatchResult(letter, opt_text, 1.0, scores, "exact")

    # 2. Unambiguous containment.
    containment_hits = []
    for letter, opt_text in options.items():
        norm_opt = normalize_text(opt_text)
        shorter = norm_answer if len(norm_answer) <= len(norm_opt) else norm_opt
        if len(shorter) >= _MIN_CONTAINMENT_LEN and (
            norm_opt in norm_answer or norm_answer in norm_opt
        ):
            containment_hits.append(letter)

    if len(containment_hits) == 1:
        letter = containment_hits[0]
        scores = {k: (0.95 if k == letter else 0.0) for k in options}
        return SemanticMatchResult(letter, options[letter], 0.95, scores, "containment")

    # 3. Semantic similarity.
    model = _get_embed_model()
    letters = list(options.keys())
    texts = [answer_text] + [
        options[letter] if isinstance(options[letter], str) else "" for letter in letters
    ]
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    answer_emb = embeddings[0]
    option_embs = embeddings[1:]
    scores_by_option = {
        letter: float(np.dot(answer_emb, emb)) for letter, emb in zip(letters, option_embs)
    }
    best_letter = max(scores_by_option, key=lambda k: scores_by_option[k])
    best_score = scores_by_option[best_letter]

    if best_score >= _SEMANTIC_THRESHOLD:
        return SemanticMatchResult(
            best_letter, options[best_letter], best_score, scores_by_option, "embedding"
        )
    return SemanticMatchResult(None, None, best_score, scores_by_option, "unmatched")
