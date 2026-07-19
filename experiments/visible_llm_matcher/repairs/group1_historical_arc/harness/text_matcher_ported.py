# experiments/visible_llm_matcher/repairs/group1_historical_arc/harness/text_matcher_ported.py
#
# Verbatim port of two-stage-prompting's embedding-based text matcher
# (used by the historical `text_extraction` method's own offline scoring
# pass — NOT the same as the fourth cell's Stage-2 LLM matcher, which uses
# historical_protocol.py's option_matching-based parser instead).
#
# Source: two-stage-prompting @ b8e784f3eb5d2a727a97eb675140b383a34584fa
#   src/twoprompt/parsing/text_matcher.py
#
# NOT modified in any way. match_text_to_options()/parse_result_from_text_match()
# are already generic over whatever `options` dict is passed in — they never
# hardcode 4 keys anywhere. The one permitted correction for the 3 repaired
# questions is achieved entirely by the CALLER passing a 3-key options dict
# (real A/B/C only) instead of the historical always-4-key
# ExperimentRunner._build_options() result (which always includes a "D" key,
# NaN or not) — see text_extraction_repair.py. This file itself needed zero
# changes to support that.

from __future__ import annotations

import math
import string
from typing import NamedTuple

import numpy as np

from choicebench.parsing.types import PARSE_MISSING, PARSE_OK, ParseResult

_SEMANTIC_THRESHOLD = 0.30
_MIN_CONTAINMENT_LEN = 4

_EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
_embed_model = None


def _get_embed_model():
    global _embed_model
    if _embed_model is None:
        from sentence_transformers import SentenceTransformer
        _embed_model = SentenceTransformer(_EMBED_MODEL_NAME)
    return _embed_model


class TextMatchResult(NamedTuple):
    label: str | None
    option_text: str | None
    score: float
    scores_by_option: dict[str, float]


def _normalize(text) -> str:
    if text is None:
        return ""
    if isinstance(text, float) and math.isnan(text):
        return ""
    if not isinstance(text, str):
        text = str(text)
    text = text.lower().strip()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return " ".join(text.split())


def match_text_to_options(
    answer_text: str | None,
    options: dict[str, str],
) -> TextMatchResult:
    """Ported verbatim from twoprompt.parsing.text_matcher.match_text_to_options."""
    if not isinstance(answer_text, str) or not answer_text.strip():
        return TextMatchResult(
            label=None,
            option_text=None,
            score=0.0,
            scores_by_option={k: 0.0 for k in options},
        )

    norm_answer = _normalize(answer_text)

    # 1. Exact match
    for letter, opt_text in options.items():
        if _normalize(opt_text) == norm_answer:
            scores = {k: (1.0 if k == letter else 0.0) for k in options}
            return TextMatchResult(
                label=letter,
                option_text=opt_text,
                score=1.0,
                scores_by_option=scores,
            )

    # 2. Unambiguous containment
    containment_hits = []
    for letter, opt_text in options.items():
        norm_opt = _normalize(opt_text)
        shorter = norm_answer if len(norm_answer) <= len(norm_opt) else norm_opt
        if len(shorter) >= _MIN_CONTAINMENT_LEN:
            if norm_opt in norm_answer or norm_answer in norm_opt:
                containment_hits.append(letter)

    if len(containment_hits) == 1:
        letter = containment_hits[0]
        scores = {k: (0.95 if k == letter else 0.0) for k in options}
        return TextMatchResult(
            label=letter,
            option_text=options[letter],
            score=0.95,
            scores_by_option=scores,
        )

    # 3. Semantic similarity
    model = _get_embed_model()
    letters = list(options.keys())
    texts = [answer_text] + [options[l] if isinstance(options[l], str) else "" for l in letters]
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)

    answer_emb = embeddings[0]
    option_embs = embeddings[1:]

    scores_by_option: dict[str, float] = {
        letter: float(np.dot(answer_emb, opt_emb))
        for letter, opt_emb in zip(letters, option_embs)
    }

    best_letter = max(scores_by_option, key=lambda k: scores_by_option[k])
    best_score = scores_by_option[best_letter]

    if best_score >= _SEMANTIC_THRESHOLD:
        return TextMatchResult(
            label=best_letter,
            option_text=options[best_letter],
            score=best_score,
            scores_by_option=scores_by_option,
        )

    return TextMatchResult(
        label=None,
        option_text=None,
        score=best_score,
        scores_by_option=scores_by_option,
    )


def parse_result_from_text_match(
    answer_text: str | None,
    options: dict[str, str],
) -> ParseResult:
    """Ported verbatim from twoprompt.parsing.text_matcher.parse_result_from_text_match."""
    match = match_text_to_options(answer_text, options)

    if match.label is not None:
        return ParseResult(
            final_choice=match.label,
            status=PARSE_OK,
            raw_text=answer_text,
            normalized_text=_normalize(answer_text or ""),
            reason=f"Text matched to option {match.label} (cosine={match.score:.3f})",
        )

    return ParseResult(
        final_choice=None,
        status=PARSE_MISSING,
        raw_text=answer_text,
        normalized_text=_normalize(answer_text or ""),
        reason=f"No option matched above threshold (best cosine={match.score:.3f})",
    )
