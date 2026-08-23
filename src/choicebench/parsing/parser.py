# src/choicebench/parsing/parser.py

from __future__ import annotations

import re
from collections.abc import Collection, Mapping

from choicebench.constants import LEGACY_OPTION_LETTERS
from choicebench.parsing.types import (
    PARSE_AMBIGUOUS,
    PARSE_MISSING,
    PARSE_OK,
    ParseResult,
)


DEFAULT_VALID_CHOICES: tuple[str, ...] = tuple(LEGACY_OPTION_LETTERS)

# Evidence-class policy (ratified Batch-2 semantics):
#   Explicit answer-selection evidence outranks mere reference/discussion;
#   the MOST RECENT explicit selection construction determines answerhood;
#   a selection committing to multiple candidates is AMBIGUOUS; mere
#   occurrence of an option's TEXT is never selection evidence.
_SELECTION_HEAD_NOUNS = {"answer", "choice", "selection", "pick", "vote"}
# Present-tense selection verbs only: past tense ("I picked Madrid because...")
# is narration, not commitment (FP guard, ratified staging).
_SELECTION_VERBS = {"choose", "pick", "select"}
_CONCLUDING_WORDS = {"therefore", "thus"}
_REFERENCE_NOUNS = {"option", "letter"}
_HEDGES = {"probably", "likely", "maybe", "perhaps", "possibly",
           "certainly", "definitely", "clearly", "apparently", "arguably"}
_CONNECTORS = {",", "or", "and"}
_STRIP_CHARS = '()[]{}<>".,:;!?' + "'"
_STRIP_CHARS_TYPO = _STRIP_CHARS + "\u201c\u201d\u2018\u2019"

_ANSWER_TAGS = frozenset({"answer", "final", "choice"})


def _sentence_ends(word: str) -> bool:
    """True when the ORIGINAL (pre-strip) token terminated a sentence."""
    return word.endswith((".", "!", "?"))


def _match_selection_construction(
    stripped: list[str], i: int, wl: str, valid_choices: Collection[str]
) -> tuple[int, str] | None:
    """Explicit selection/conclusion language at index i.

    Returns (candidate_index, letter). Exactly one hedge adverb may sit
    between the copula/colon and the candidate ("my answer is probably B").
    Direct candidate without copula is restricted to 'answer' and 'pick' so
    bare discussion nouns ('choice B', 'selection B') gain no strength.
    """
    def cand(idx: int):
        if idx < len(stripped) and stripped[idx] and stripped[idx].upper() in valid_choices:
            return idx, stripped[idx].upper()
        if idx < len(stripped) and stripped[idx].lower() in _HEDGES:
            idx2 = idx + 1
            if idx2 < len(stripped) and stripped[idx2].upper() in valid_choices:
                return idx2, stripped[idx2].upper()
        return None

    # S1: "final answer [:|is|was] X"
    if wl == "final" and i + 2 < len(stripped) and stripped[i + 1].lower() == "answer":
        c = cand(i + 2)
        if c is not None:
            return c

    # S2: head noun with copula (+hedge), or direct X for answer/pick only
    if wl in _SELECTION_HEAD_NOUNS:
        j = i + 1
        if j < len(stripped) and stripped[j].lower() in {"is", "was"}:
            c = cand(j + 1)
            if c is not None:
                return c
        elif wl in {"answer", "pick"}:
            c = cand(j)
            if c is not None:
                return c

    # S3: present-tense selection verbs (past tense is narration, not selection)
    if wl in _SELECTION_VERBS:
        c = cand(i + 1)
        if c is not None:
            return c

    # S3b: "go(es) with X"
    if wl in {"go", "goes"} and i + 2 < len(stripped) \
            and stripped[i + 1].lower() == "with":
        c = cand(i + 2)
        if c is not None:
            return c

    # S4: concluding adverbs
    if wl in _CONCLUDING_WORDS:
        c = cand(i + 1)
        if c is not None:
            return c

    return None


def _match_reference_construction(
    stripped: list[str], i: int, wl: str, valid_choices: Collection[str]
) -> tuple[int, str] | None:
    """Reference/discussion language (demoted tier): 'option X', 'letter is X'."""
    if wl in _REFERENCE_NOUNS and i + 1 < len(stripped):
        j = i + 1
        if stripped[j].lower() in {"is", "was"}:
            j += 1
        if j < len(stripped) and stripped[j].upper() in valid_choices:
            return j, stripped[j].upper()
    return None


def _extend_coordination_chain(
    stripped: list[str],
    valid_choices: Collection[str],
    sent_end: list[bool],
    cand_idx: int,
) -> tuple[list[str], int]:
    """Absorb ', or'-style coordinated candidates within the same sentence.

    Returns (additional_letters_beyond_the_candidate, last_absorbed_index).
    Chains break on any non-connector token and never cross a sentence end.
    """
    chain: list[str] = []
    prev_idx = cand_idx
    idx = cand_idx + 1
    n = len(stripped)
    while idx < n:
        if sent_end[prev_idx]:
            break
        tok = stripped[idx]
        if not tok or tok not in _CONNECTORS:
            break
        if sent_end[idx]:
            break
        j = idx + 1
        while j < n and stripped[j] in _CONNECTORS:
            j += 1
        if j >= n or not stripped[j]:
            break
        nxt = stripped[j].upper()
        if nxt in valid_choices:
            chain.append(nxt)
            prev_idx = j
            idx = j + 1
            continue
        break
    end = (idx - 1) if chain else cand_idx
    return chain, max(end, cand_idx)


def _resolve_constructions(constructions, normalized_text: str) -> ParseResult:
    """Ratified resolution order.

    1. Any explicit selection construction exists -> the MOST RECENT one
       governs: compound => AMBIGUOUS, clean => OK.
    2. No explicit selection -> last construction compound => AMBIGUOUS;
       otherwise last reference/weak => OK; none => MISSING.
    """
    selections = [c for c in constructions if c["tier"] == "selection"]
    if selections:
        last_sel = selections[-1]
        if last_sel["compound"]:
            return ParseResult(
                final_choice=None, status=PARSE_AMBIGUOUS, raw_text=None,
                normalized_text=normalized_text,
                reason="Compound answer construction")
        return ParseResult(
            final_choice=last_sel["head"], status=PARSE_OK, raw_text=None,
            normalized_text=normalized_text,
            reason="Answer successfully parsed from explicit selection language")
    last = constructions[-1] if constructions else None
    if last is not None and last["compound"]:
        return ParseResult(
            final_choice=None, status=PARSE_AMBIGUOUS, raw_text=None,
            normalized_text=normalized_text,
            reason="Compound answer construction")
    if last is not None:
        reason = ("Answer successfully parsed from option reference"
                  if last["tier"] == "reference"
                  else "Answer successfully parsed from last standalone letter")
        return ParseResult(
            final_choice=last["head"], status=PARSE_OK, raw_text=None,
            normalized_text=normalized_text, reason=reason)
    return ParseResult(
        final_choice=None, status=PARSE_MISSING, raw_text=None,
        normalized_text=normalized_text, reason="No direct answer letter found")


def normalize_output_text(raw_text: str | None) -> str:
    """
    Normalize raw model output before parsing.

    Returns an empty string for None input. Otherwise strips leading and
    trailing whitespace and collapses repeated internal whitespace into
    single spaces, preserving content for both letter extraction and text
    matching.

    Args:
        raw_text: Raw model output text from a provider response.

    Returns:
        Normalized text string suitable for downstream parsing.
    """
    if raw_text is None or not isinstance(raw_text, str):
        return ""
    raw_text = raw_text.strip()
    if raw_text == "":
        return ""
    words = raw_text.split()
    normalized_string = " ".join(words)
    return normalized_string


def extract_choice_letter(
    normalized_text: str,
    valid_choices: Collection[str] = DEFAULT_VALID_CHOICES,
) -> ParseResult:
    """
    Attempt to extract a direct answer letter from normalized output.

    Ratified evidence-class semantics (Batch 2):

    1. Standalone single-token response.
    2. Constructions, detected left→right and resolved by class:
       - SELECTION (explicit answer language): "final answer [:|is] X",
         "{answer|choice|selection|pick|vote} [is|was] X" (direct X only for
         answer/pick), present-tense {choose,pick,select} X, "go(es) with X",
         "therefore/thus X"; one hedge adverb may precede the candidate.
       - REFERENCE (mere mention): "{option|letter} [is] X".
       - WEAK: any other standalone valid letter.
       Candidates extend through ", or"-style coordination within the same
       sentence; a construction absorbing >=2 distinct letters is COMPOUND.
       Resolution: the MOST RECENT selection governs (compound => AMBIGUOUS);
       with no selection, a compound last construction is AMBIGUOUS, else the
       last reference/weak wins. Discussion never overrides selection.

    Args:
        normalized_text: Pre-normalized model output text.
        valid_choices: Allowed answer letters.

    Returns:
        ParseResult describing the letter-extraction attempt.
    """
    if normalized_text == "":
        return ParseResult(
            final_choice=None,
            status=PARSE_MISSING,
            raw_text=None,
            normalized_text=normalized_text,
            reason="No output to parse",
        )

    words = normalized_text.split()
    stripped = [w.strip(_STRIP_CHARS_TYPO) for w in words]
    sent_end = [_sentence_ends(w) for w in words]

    # Priority 1: single-token response
    if len(stripped) == 1 and stripped[0].upper() in valid_choices:
        return ParseResult(
            final_choice=stripped[0].upper(),
            status=PARSE_OK,
            raw_text=None,
            normalized_text=normalized_text,
            reason="Answer successfully parsed",
        )

    constructions: list[dict] = []
    consumed = [False] * len(stripped)

    i = 0
    n = len(stripped)
    while i < n:
        w = stripped[i]
        if not w or consumed[i]:
            i += 1
            continue
        wl = w.lower()

        sel = _match_selection_construction(stripped, i, wl, valid_choices)
        if sel is not None:
            cand_idx, head = sel
            chain, end_idx = _extend_coordination_chain(
                stripped, valid_choices, sent_end, cand_idx)
            letters = {head, *chain}
            constructions.append({
                "tier": "selection", "head": head, "letters": letters,
                "compound": len(letters) > 1,
                "start": i, "end": max(end_idx, cand_idx),
            })
            for k in range(i, constructions[-1]["end"] + 1):
                consumed[k] = True
            i = constructions[-1]["end"] + 1
            continue

        ref = _match_reference_construction(stripped, i, wl, valid_choices)
        if ref is not None:
            cand_idx, head = ref
            chain, end_idx = _extend_coordination_chain(
                stripped, valid_choices, sent_end, cand_idx)
            letters = {head, *chain}
            constructions.append({
                "tier": "reference", "head": head, "letters": letters,
                "compound": len(letters) > 1,
                "start": i, "end": max(end_idx, cand_idx),
            })
            for k in range(i, constructions[-1]["end"] + 1):
                consumed[k] = True
            i = constructions[-1]["end"] + 1
            continue

        if w.upper() in valid_choices:
            cand_idx = i
            chain, end_idx = _extend_coordination_chain(
                stripped, valid_choices, sent_end, cand_idx)
            letters = {w.upper(), *chain}
            constructions.append({
                "tier": "weak", "head": w.upper(), "letters": letters,
                "compound": len(letters) > 1,
                "start": i, "end": max(end_idx, i),
            })
            if chain:
                for k in range(i, constructions[-1]["end"] + 1):
                    consumed[k] = True
                i = constructions[-1]["end"] + 1
                continue
        i += 1

    return _resolve_constructions(constructions, normalized_text)


def _option_matches_as_token(normalized_option: str, normalized_text: str) -> bool:
    """
    Return True if the option text appears in the model output as a distinct
    token sequence rather than as an incidental substring.

    Matching is bounded by non-alphanumeric characters (or the string edges),
    so option ``"4"`` matches the standalone ``4`` in ``"the answer is 4"`` but
    not the ``4`` inside ``"14"``. Comparison is case-insensitive.
    """
    if normalized_option == "":
        return False
    pattern = (
        r"(?<![A-Za-z0-9])"
        + re.escape(normalized_option)
        + r"(?![A-Za-z0-9])"
    )
    return re.search(pattern, normalized_text, flags=re.IGNORECASE) is not None


def extract_choice_text_match(
    normalized_text: str,
    options: Mapping[str, str],
) -> ParseResult:
    """
    Fallback matching against the option texts themselves, for outputs where
    the model writes out the answer text instead of a letter.

    Each option's normalized text is matched against the normalized model
    output on token boundaries (see _option_matches_as_token): the option must
    appear as a distinct token sequence, not merely as a raw substring, so a
    short/numeric option like ``"4"`` does not spuriously match inside ``"14"``.
    Returns PARSE_OK if exactly one option matches, PARSE_AMBIGUOUS if more than
    one option matches, and PARSE_MISSING if none match.

    Args:
        normalized_text: Pre-normalized model output text.
        options: Mapping from answer letter to answer text.

    Returns:
        ParseResult describing the text-matching attempt.
    """
    candidates = []
    for letter, option_text in options.items():
        normalized_option = normalize_output_text(option_text)
        if _option_matches_as_token(normalized_option, normalized_text):
            candidates.append(letter)

    candidates = set(candidates)

    if len(candidates) == 1:
        return ParseResult(
            final_choice=next(iter(candidates)),
            status=PARSE_OK,
            raw_text=None,
            normalized_text=normalized_text,
            reason="Answer successfully parsed from option text match"
        )
    elif len(candidates) > 1:
        return ParseResult(
            final_choice=None,
            status=PARSE_AMBIGUOUS,
            raw_text=None,
            normalized_text=normalized_text,
            reason="Multiple conflicting option text matches found"
        )
    else:
        return ParseResult(
            final_choice=None,
            status=PARSE_MISSING,
            raw_text=None,
            normalized_text=normalized_text,
            reason="Text present but no option text match found"
        )


_UNWRAP_PAIRS = (("**", "**"), ("__", "__"), ("`", "`"),
                 ("\u201c", "\u201d"), ('"', '"'),
                 ("\u2018", "\u2019"), ("'", "'"))
_TAG_WRAP_RE = re.compile(r"^<([A-Za-z][A-Za-z0-9_]*)>(.*)</\1>$", re.DOTALL)

# G5 (tier-4): anchored present-tense option-text selection. Past tense is
# narration, not commitment; arbitrary prose containing a verb cannot match
# because the pattern is anchored to the whole payload.
_SELECTION_OPT_VERB_RE = re.compile(
    r"^(?:i\s+|we\s+|you\s+)?(?:(?:will|shall|'ll|\u2019ll)\s+)?"
    r"(?:choose|pick|select)\s+(?:the\s+)?(?P<opt>.+?)\s*[.!]?$",
    re.IGNORECASE,
)
_GO_WITH_RE = re.compile(
    r"^(?:i\s+|we\s+|you\s+|he\s+|she\s+|it\s+|they\s+)?"
    r"go(?:es)?\s+with\s+(?:the\s+)?(?P<opt>.+?)\s*[.!]?$",
    re.IGNORECASE,
)
_G2_CUE_HEAD_RE = re.compile(
    r"^(?:(?:my|the|final)\s+)*(?:answer|choice|selection|pick|vote)"
    r"\b[\s:.\-]*(?:is|was)?\s*(?:the\s+)?",
    re.IGNORECASE,
)
_G3_COPULA_RE = re.compile(
    r"^.+?\b(?:is|was|are|were|could\s+be|may\s+be|might\s+be|"
    r"would\s+be|should\s+be|must\s+be)\s+(?:the\s+)?",
    re.IGNORECASE,
)
_G4_APPOSITIVE_RE_TAIL = r"(?:\s*[,;:\-\u2013\u2014]\s*\S+){1,2}\s*[.!]?"


def _unwrap_simple_wrappers(text: str) -> str:
    """Strip well-formed wrapper pairs around the WHOLE payload, repeatedly.

    Restricted per ratified policy: quote/markup pairs from _UNWRAP_PAIRS and
    a tag allowlist (_ANSWER_TAGS: answer/final/choice). Arbitrary XML/HTML
    tags are NOT unwrapped, and mid-prose markup is untouched.
    """
    changed = True
    while changed:
        changed = False
        t = text.strip()
        for open_w, close_w in _UNWRAP_PAIRS:
            if len(t) >= len(open_w) + len(close_w) \
                    and t.startswith(open_w) and t.endswith(close_w):
                inner = t[len(open_w):len(t) - len(close_w)].strip()
                if inner:
                    text = inner
                    changed = True
                    break
        if not changed:
            m = _TAG_WRAP_RE.match(text.strip())
            if m and m.group(1).lower() in _ANSWER_TAGS and m.group(2).strip():
                text = m.group(2).strip()
                changed = True
    return text


def _answer_like_option_match(
    normalized_text: str,
    options: Mapping[str, str],
) -> ParseResult:
    """P6-1 gate: option text counts as an answer ONLY in answer-like structure.

    Anchored grammars over the (wrapper-normalized) payload — no length
    heuristics:
      G1 exact payload;
      G2 cued head ("Answer:", "My answer is", "final pick:") + exact text;
      G3 copula immediately before the text in the FINAL sentence;
      G4 appositive head ("Paris, France");
      G5 explicit present-tense selection verb + exact text.
    Multiple distinct options matched by these anchored forms => AMBIGUOUS;
    incidental mentions elsewhere in the payload never match.
    """
    T = normalized_text
    if not T:
        return ParseResult(
            final_choice=None, status=PARSE_MISSING, raw_text=None,
            normalized_text=T, reason="No output to parse")

    sentences = [s for s in re.split(r"(?<=[.!?)])\s+", T) if s.strip()]
    last_sentence = (sentences[-1] if sentences else T).rstrip(".!?").strip()

    hits: set[str] = set()
    for letter, opt_text in options.items():
        opt = normalize_output_text(opt_text)
        if not opt:
            continue
        esc = re.escape(opt)
        if T == opt:                                                    # G1
            hits.add(letter)
            continue
        if re.match(_G2_CUE_HEAD_RE.pattern + esc + r"\s*[.!]?$", T, re.IGNORECASE):
            hits.add(letter)                                            # G2
            continue
        if last_sentence and re.match(
                _G3_COPULA_RE.pattern + rf"(?:the\s+)?{esc}$",
                last_sentence, re.IGNORECASE):                          # G3
            hits.add(letter)
            continue
        if re.match(rf"^{esc}{_G4_APPOSITIVE_RE_TAIL}$", T):            # G4
            hits.add(letter)
            continue
        m = _SELECTION_OPT_VERB_RE.match(T) or _GO_WITH_RE.match(T)     # G5
        if m and m.group("opt").strip().lower() == opt.lower():
            hits.add(letter)

    if len(hits) == 1:
        return ParseResult(
            final_choice=next(iter(hits)), status=PARSE_OK, raw_text=None,
            normalized_text=T,
            reason="Answer successfully parsed from structured option text")
    if len(hits) > 1:
        return ParseResult(
            final_choice=None, status=PARSE_AMBIGUOUS, raw_text=None,
            normalized_text=T,
            reason="Multiple conflicting option text matches found")
    # Ambiguity-recall preservation: the gated grammars above never accept a
    # bare mention, but a payload containing MULTIPLE option texts inside
    # hedged/uncued prose ("It looks like X or Y") is still noncommittal and
    # must stay AMBIGUOUS rather than degrade to MISSING. A SINGLE blanket
    # hit is deliberately ignored here (P6-1: mention != selection) and can
    # only ever produce MISSING from this path.
    legacy = extract_choice_text_match(T, options)
    if legacy.status == PARSE_AMBIGUOUS:
        return ParseResult(
            final_choice=None, status=PARSE_AMBIGUOUS, raw_text=None,
            normalized_text=T, reason=legacy.reason)
    return ParseResult(
        final_choice=None, status=PARSE_MISSING, raw_text=None,
        normalized_text=T,
        reason="Text present but not structured as an answer")


def parse_model_answer(
    raw_text: str | None,
    options: Mapping[str, str],
) -> ParseResult:
    """
    Parse a model's MCQ answer into a structured result.

    Normalizes the raw text, strips simple whole-payload answer wrappers
    (allowlisted), tries direct letter extraction first; if that returns
    PARSE_MISSING, falls back to STRUCTURALLY answer-like option-text
    matching. Never raises on junk, empty, or unexpected output.

    Args:
        raw_text: Raw text returned by the model.
        options: Mapping from answer letter to answer text.

    Returns:
        Final ParseResult for downstream scoring.
    """
    normalized_text = normalize_output_text(raw_text)
    unwrapped = _unwrap_simple_wrappers(normalized_text)
    temp_result = extract_choice_letter(unwrapped, valid_choices=tuple(options.keys()))
    if temp_result.status == PARSE_MISSING:
        temp_result = _answer_like_option_match(unwrapped, options)
    return ParseResult(
        final_choice=temp_result.final_choice,
        status=temp_result.status,
        raw_text=raw_text,
        normalized_text=temp_result.normalized_text,
        reason=temp_result.reason
    )
