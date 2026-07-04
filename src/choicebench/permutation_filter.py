"""Permutation-safety filter for MCQ benchmarks.

Reference: Zheng et al., ICLR 2024, "Large Language Models Are Not Robust
Multiple Choice Selectors" (arXiv:2309.03882) — Table 7 footnote: "We excluded
a few MMLU samples (about 3.2% in total), where the options refer to each
other like 'A and B', 'none of the above', in all the experiments."

Permuting a question's option order changes the identity of "A", "B", etc. —
so an option whose *entire* meaning is a reference to another option's label
(e.g. an option that reads "A and B", or "None of the above") becomes false or
nonsensical after shuffling, silently corrupting any position-shuffling method
(cyclic permutation, PriDe, answer-moving attacks). This is an opt-in data-prep
step (see --filter-permutation-unsafe in scripts/prepare_data.py), never
applied by default.

Patterns match the *entire* (whitespace-normalized, case-insensitive) option
text, not a substring search over arbitrary prose: a sentence that happens to
contain "A and B" as ordinary content (e.g. "Iron and B12 deficiency") must
not be excluded — only an option whose full text IS one of these referential
forms should be. This deliberately biases toward false negatives (an unusual
referential option worded differently slips through unfiltered) over false
positives (a valid, content-bearing option gets silently dropped).
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from choicebench.pipeline.options import build_choices

_NORMALIZED_SUFFIX = "_normalized.csv"
_FILTER_SUFFIX = "_permutation_filter.json"

# Option labels never exceed J (10-option max, MMLU-Pro's ceiling); this
# benchmark set is 4-option MMLU, but the pattern set is written generally.
_LETTER = r"[A-J]"

_PATTERNS: dict[str, re.Pattern[str]] = {
    "all_of_the_above": re.compile(r"^all of the above\.?$", re.IGNORECASE),
    "none_of_the_above": re.compile(r"^none of the above\.?$", re.IGNORECASE),
    "both_and": re.compile(rf"^both {_LETTER} and {_LETTER}\.?$", re.IGNORECASE),
    "letter_and_letter": re.compile(rf"^{_LETTER} and {_LETTER}\.?$", re.IGNORECASE),
    "letter_list_and_letter": re.compile(
        rf"^{_LETTER}(, {_LETTER})+,? and {_LETTER}\.?$", re.IGNORECASE
    ),
    "neither_nor": re.compile(rf"^neither {_LETTER} nor {_LETTER}\.?$", re.IGNORECASE),
}


@dataclass
class PermutationFilterExclusion:
    question_id: str
    matched_pattern: str
    offending_option_text: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def match_permutation_unsafe_pattern(option_text: str) -> str | None:
    """Return the matched pattern name, or None if option_text is permutation-safe."""
    text = " ".join(option_text.strip().split())
    for name, pattern in _PATTERNS.items():
        if pattern.match(text):
            return name
    return None


def filter_permutation_unsafe(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, list[PermutationFilterExclusion]]:
    """Exclude questions with meta-referential option text.

    Returns (filtered_df, exclusions) — filtered_df keeps only questions where
    every option's text is permutation-safe; exclusions records one entry per
    excluded question (its first-matched offending option only).
    """
    exclusions: list[PermutationFilterExclusion] = []
    keep_mask: list[bool] = []
    for _, row in df.iterrows():
        row_dict = row.to_dict()
        choices = build_choices(row_dict)
        matched_pattern: str | None = None
        offending_text: str | None = None
        for choice in choices:
            pattern_name = match_permutation_unsafe_pattern(choice["text"])
            if pattern_name is not None:
                matched_pattern = pattern_name
                offending_text = choice["text"]
                break
        if matched_pattern is not None:
            exclusions.append(
                PermutationFilterExclusion(
                    question_id=row_dict["question_id"],
                    matched_pattern=matched_pattern,
                    offending_option_text=offending_text,
                )
            )
            keep_mask.append(False)
        else:
            keep_mask.append(True)

    filtered = df[pd.Series(keep_mask, index=df.index)].reset_index(drop=True)
    return filtered, exclusions


def permutation_filter_path_for(normalized_csv: Path) -> Path:
    """Sidecar path for a normalized CSV (X_normalized.csv -> X_permutation_filter.json)."""
    normalized_csv = Path(normalized_csv)
    name = normalized_csv.name
    if name.endswith(_NORMALIZED_SUFFIX):
        stem = name[: -len(_NORMALIZED_SUFFIX)]
    else:
        stem = normalized_csv.stem
    return normalized_csv.with_name(f"{stem}{_FILTER_SUFFIX}")


def write_permutation_filter_report(
    exclusions: list[PermutationFilterExclusion],
    n_total: int,
    path: Path,
) -> Path:
    """Write the exclusion accounting to a JSON sidecar."""
    report = {
        "n_total": n_total,
        "n_excluded": len(exclusions),
        "n_kept": n_total - len(exclusions),
        "exclusions": [e.as_dict() for e in exclusions],
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2))
    return path
