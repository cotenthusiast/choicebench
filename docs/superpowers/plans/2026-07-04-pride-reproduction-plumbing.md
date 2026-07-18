# PriDe Single-Cell Reproduction Plumbing — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the reproduction plumbing for the PriDe paper's (Zheng et al., ICLR 2024, arXiv:2309.03882) single MMLU/llama-13B/0-shot cell — a permutation-safety data filter, the paper's exact prompt template with BOS control, a runnable HPC config for `direct_logprob` + `cyclic_logprob` only (PriDe itself is recomputed offline later, out of scope here), and an official-numbers fixture for later comparison.

**Architecture:** Four independent, additive slices onto the existing pipeline: (1) a new `permutation_filter.py` module wired into `scripts/prepare_data.py` behind an opt-in flag; (2) a new `prompts/pride_repro/` template version plus a `subject` parameter threaded through `build_direct_mcq_prompt()` and its two call sites that matter here (`direct_logprob.py`, `permutation.py._build_permuted_prompt` — the latter shared by `cyclic_logprob`); (3) a new `add_bos_token` field on `ModelConfig`, threaded into `HuggingFaceBackend`'s two tokenizer call sites; (4) a runnable YAML + sbatch + a static JSON targets fixture. No GPU code path is exercised in this repo — everything is verified via the dummy backend.

**Tech Stack:** Python 3.14, pandas, pytest (+ pytest-asyncio), PyYAML, HuggingFace transformers/torch (import-lazy, not required for this plan's own test suite).

## Global Constraints

- PriDe is explicitly OUT of scope for the run config in this plan — only `direct_logprob` and `cyclic_logprob` are configured. A separate offline script recomputes PriDe from persisted distributions later; do not add a `pride:` method entry anywhere in this plan's YAML.
- `huggyllama/llama-13b` is LLaMA-1, not Llama-2 — every docstring/comment referencing the model must say "LLaMA-1" explicitly to prevent future confusion.
- The permutation-safety filter is opt-in (`--filter-permutation-unsafe`), never applied by default — must not change the byte content of any existing `*_normalized.csv` when the flag is absent.
- Every prompt `.txt` file is used verbatim as a `str.format()` template — no comments, no extra whitespace beyond what the paper's template specifies. The rendered prompt must end in the literal string `Answer:` with **no trailing newline or space** (score_options reads the logit at the last token position).
- Paper citation format used elsewhere in this repo: `Zheng et al., ICLR 2024, "Large Language Models Are Not Robust Multiple Choice Selectors" (arXiv:2309.03882)` — match it exactly in new docstrings.
- Full test suite must stay green after every task (`python -m pytest -q`; 634 passed at plan-writing time).

---

## Primary-source facts gathered before writing this plan

Verified directly against the arXiv:2309.03882 PDF (`pdftotext -layout`, not a summary) so the plan is not building on a hallucinated template:

- **Prompt template (paper's Figure 6, "Input formats for open-source models (e.g., llama(-2) and vicuna)", 0-shot)**:
  ```
  The following are multiple choice questions about {subject}. You should directly answer the question by choosing the correct option.

  Question: {question}
  Options:
  {options}
  Answer:
  ```
  (the `[in-context examples]` line in the figure is 5-shot-only; 0-shot omits it, which is exactly the blank-line-then-Question layout above).
- **BOS**: Figure 6's caption states verbatim: *"Note that for all the open-source models, we do not prepend the input text with a bos token."*
- **Subject formatting**: cross-checked against `chujiezheng/LLM-MCQ-Bias/code/eval_clm_utils.py` — subject is substituted via `subject.replace('_', ' ')` (e.g. MMLU's raw `electrical_engineering` → `electrical engineering`, matching the figure's example). Every benchmark normalizer in this repo already populates a `subject` column (MMLU: real subject; ARC/TruthfulQA/HellaSwag: a constant/activity-label placeholder) — the same formatting call handles all of them, so no new fallback column is needed, only documentation of what a non-MMLU benchmark will render there.
- **Permutation-safety exclusion (paper §Table 7 footnote)**: *"We excluded a few MMLU samples (about 3.2% in total), where the options refer to each other like 'A and B', 'none of the above', in all the experiments."* Table 7 gives the exact counts: MMLU test 14,042 → 13,592 evaluated (i.e. 450 excluded, 3.2%) — matches the target the task brief already names.
- **Table 3 (0-shot), MMLU block, llama-13B row** (the row backing Part D's fixture) reads, column-by-column (Default RStd/Acc, Removing-IDs RStd/Acc, Cyclic-Perm RStd/Acc, PriDe-5% RStd/Acc, PriDe-40% RStd/Acc, PriDe-80% RStd/Acc):
  `17.4  34.6   2.6  29.1   2.9  47.7   5.7  36.4   3.9  40.4   2.6  45.3`
  This matches the task brief's numbers exactly — **no discrepancy, no fixture correction needed.** (The optional GitHub cross-check turned up no released per-cell numbers in `eval_clm_utils.py`'s scope; the PDF is the primary source anyway per the brief's own tie-breaking rule, so this satisfies the "optional cross-check" without further repo spelunking.)
- **Benchmark CSV path resolution quirk (load-bearing for Part C)**: `scripts/run_experiment.py::load_benchmark()` resolves a **registered** benchmark name (e.g. `name: mmlu`) to a *hardcoded* path `PROCESSED_DIR / "mmlu_normalized.csv"` via `get_benchmark_path(name)` — it does **not** consult `output_name`. Only the generic `name: huggingface` path honors `output_name` (via `benchmark_normalized_stem()`). Since the permutation-filtered CSV must not overwrite the canonical unfiltered `mmlu_normalized.csv` (that would silently corrupt every other experiment that runs plain MMLU), Part C's config **must** use `name: huggingface` + `hf_path: cais/mmlu` + `hf_subset: all` + `output_name: mmlu_filtered`, not `name: mmlu`.

---

### Task 1: Permutation-safety filter module

**Files:**
- Create: `src/choicebench/permutation_filter.py`
- Test: `tests/test_permutation_filter.py`

**Interfaces:**
- Consumes: `choicebench.pipeline.options.build_choices(question_row: Mapping) -> list[dict]` (each dict has `label`, `text`, `source_index`) — already exists, unchanged.
- Produces (used by Task 2):
  - `PermutationFilterExclusion` dataclass with fields `question_id: str`, `matched_pattern: str`, `offending_option_text: str`, method `.as_dict() -> dict`.
  - `match_permutation_unsafe_pattern(option_text: str) -> str | None` — returns the matched pattern name or `None`.
  - `filter_permutation_unsafe(df: pd.DataFrame) -> tuple[pd.DataFrame, list[PermutationFilterExclusion]]`.
  - `permutation_filter_path_for(normalized_csv: Path) -> Path` — sidecar path, mirrors `stats.py::stats_path_for` (`X_normalized.csv` → `X_permutation_filter.json`).
  - `write_permutation_filter_report(exclusions: list[PermutationFilterExclusion], n_total: int, path: Path) -> Path`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_permutation_filter.py

import json
from pathlib import Path

import pandas as pd
import pytest

from choicebench.permutation_filter import (
    PermutationFilterExclusion,
    filter_permutation_unsafe,
    match_permutation_unsafe_pattern,
    permutation_filter_path_for,
    write_permutation_filter_report,
)


def _row(question_id: str, options: list[str], correct_index: int = 0) -> dict:
    choices_json = json.dumps(
        [{"text": opt, "source_index": i} for i, opt in enumerate(options)]
    )
    return {
        "question_id": question_id,
        "subject": "abstract_algebra",
        "question_text": "irrelevant",
        "choices_json": choices_json,
        "correct_index": correct_index,
        "correct_option": ["A", "B", "C", "D"][correct_index],
        "n_choices": len(options),
    }


class TestMatchPermutationUnsafePattern:
    """Positive fixtures: each must match, since permuting them changes meaning."""

    @pytest.mark.parametrize(
        "text",
        [
            "All of the above",
            "all of the above.",
            "None of the above",
            "none of the above.",
            "Both A and C",
            "both a and c",
            "A and B",
            "b and d",
            "Neither A nor C",
            "neither b nor d.",
        ],
    )
    def test_matches_meta_referential_option(self, text):
        assert match_permutation_unsafe_pattern(text) is not None

    """Negative fixtures: ordinary option content that happens to contain
    conjunction-like substrings must NOT match — the filter matches the
    *entire* option text, not a substring buried in normal prose, to bias
    toward false negatives over dropping valid items."""

    @pytest.mark.parametrize(
        "text",
        [
            "Iron and B12 deficiency",
            "Mitochondria and ribosomes",
            "Vitamin A and vitamin B are both required",
            "The mixture of A and B produces a precipitate",
            "42",
            "A rapid increase in temperature",
        ],
    )
    def test_does_not_match_ordinary_option_text(self, text):
        assert match_permutation_unsafe_pattern(text) is None


class TestFilterPermutationUnsafe:
    def test_excludes_question_with_unsafe_option(self):
        df = pd.DataFrame(
            [
                _row("q1", ["4", "5", "6", "7"]),
                _row("q2", ["oxygen", "nitrogen", "A and B", "carbon"]),
            ]
        )

        filtered, exclusions = filter_permutation_unsafe(df)

        assert filtered["question_id"].tolist() == ["q1"]
        assert len(exclusions) == 1
        assert exclusions[0].question_id == "q2"
        assert exclusions[0].matched_pattern == "letter_and_letter"
        assert exclusions[0].offending_option_text == "A and B"

    def test_keeps_all_rows_when_none_are_unsafe(self):
        df = pd.DataFrame(
            [
                _row("q1", ["4", "5", "6", "7"]),
                _row("q2", ["oxygen", "nitrogen", "hydrogen", "carbon"]),
            ]
        )

        filtered, exclusions = filter_permutation_unsafe(df)

        assert len(filtered) == 2
        assert exclusions == []

    def test_exclusion_is_dataclass_with_as_dict(self):
        df = pd.DataFrame([_row("q1", ["none of the above", "5", "6", "7"])])

        _, exclusions = filter_permutation_unsafe(df)

        assert isinstance(exclusions[0], PermutationFilterExclusion)
        assert exclusions[0].as_dict() == {
            "question_id": "q1",
            "matched_pattern": "none_of_the_above",
            "offending_option_text": "none of the above",
        }


class TestSidecarIO:
    def test_permutation_filter_path_for_replaces_normalized_suffix(self):
        path = permutation_filter_path_for(Path("/data/mmlu_filtered_normalized.csv"))
        assert path == Path("/data/mmlu_filtered_permutation_filter.json")

    def test_write_permutation_filter_report_round_trips(self, tmp_path):
        exclusions = [
            PermutationFilterExclusion("q2", "letter_and_letter", "A and B"),
        ]
        path = write_permutation_filter_report(exclusions, n_total=2, path=tmp_path / "x_permutation_filter.json")

        report = json.loads(path.read_text())
        assert report["n_total"] == 2
        assert report["n_excluded"] == 1
        assert report["n_kept"] == 1
        assert report["exclusions"] == [
            {"question_id": "q2", "matched_pattern": "letter_and_letter", "offending_option_text": "A and B"}
        ]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_permutation_filter.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'choicebench.permutation_filter'`

- [ ] **Step 3: Write the implementation**

```python
# src/choicebench/permutation_filter.py

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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_permutation_filter.py -v`
Expected: PASS (all tests green)

- [ ] **Step 5: Commit**

```bash
git add src/choicebench/permutation_filter.py tests/test_permutation_filter.py
git commit -m "feat(data): add permutation-safety filter module"
```

---

### Task 2: Wire `--filter-permutation-unsafe` into `scripts/prepare_data.py`

**Files:**
- Modify: `scripts/prepare_data.py`
- Test: `tests/scripts/test_prepare_data.py` (append)

**Interfaces:**
- Consumes: Task 1's `filter_permutation_unsafe`, `permutation_filter_path_for`, `write_permutation_filter_report`.
- Produces: `main()` now filters+reports when `--filter-permutation-unsafe` is passed; `_resolve_output_stem` behavior unchanged (still tested by the existing three tests) but `main()`'s stem gets an `_filtered` suffix appended when the flag is set and `--output-name` was **not** explicitly given (so a bare rerun of the flag never collides with the canonical unfiltered CSV, and an explicit `--output-name` still wins verbatim per the existing precedence rule).

- [ ] **Step 1: Write the failing test**

```python
# Append to tests/scripts/test_prepare_data.py

import json

import pandas as pd
import pytest


def test_filter_flag_appends_filtered_suffix_when_no_output_name(monkeypatch, tmp_path):
    mod = _load_prepare_data()
    monkeypatch.setattr(mod, "PROCESSED_DIR", tmp_path)

    raw_df = pd.DataFrame(
        [
            {"subject": "x", "question": "q1", "choices": "['4','5','6','7']", "answer": 0},
            {"subject": "x", "question": "q2", "choices": "['A and B','5','6','7']", "answer": 1},
        ]
    )
    monkeypatch.setattr(mod, "download_from_huggingface", lambda *a, **k: raw_df)
    monkeypatch.setattr(
        mod,
        "parse_args",
        lambda: __import__("argparse").Namespace(
            hf_path="cais/mmlu",
            hf_subset="all",
            split="test",
            output_name=None,
            filter_permutation_unsafe=True,
        ),
    )

    mod.main()

    out_csv = tmp_path / "mmlu_filtered_normalized.csv"
    assert out_csv.exists()
    df = pd.read_csv(out_csv)
    assert len(df) == 1  # q2 excluded

    sidecar = tmp_path / "mmlu_filtered_permutation_filter.json"
    report = json.loads(sidecar.read_text())
    assert report["n_total"] == 2
    assert report["n_excluded"] == 1
    assert report["exclusions"][0]["matched_pattern"] == "letter_and_letter"


def test_filter_flag_respects_explicit_output_name(monkeypatch, tmp_path):
    mod = _load_prepare_data()
    monkeypatch.setattr(mod, "PROCESSED_DIR", tmp_path)

    raw_df = pd.DataFrame(
        [{"subject": "x", "question": "q1", "choices": "['4','5','6','7']", "answer": 0}]
    )
    monkeypatch.setattr(mod, "download_from_huggingface", lambda *a, **k: raw_df)
    monkeypatch.setattr(
        mod,
        "parse_args",
        lambda: __import__("argparse").Namespace(
            hf_path="cais/mmlu",
            hf_subset="all",
            split="test",
            output_name="custom_stem",
            filter_permutation_unsafe=True,
        ),
    )

    mod.main()

    assert (tmp_path / "custom_stem_normalized.csv").exists()
    assert not (tmp_path / "custom_stem_filtered_normalized.csv").exists()


def test_no_filter_flag_leaves_output_unfiltered(monkeypatch, tmp_path):
    mod = _load_prepare_data()
    monkeypatch.setattr(mod, "PROCESSED_DIR", tmp_path)

    raw_df = pd.DataFrame(
        [{"subject": "x", "question": "q1", "choices": "['A and B','5','6','7']", "answer": 0}]
    )
    monkeypatch.setattr(mod, "download_from_huggingface", lambda *a, **k: raw_df)
    monkeypatch.setattr(
        mod,
        "parse_args",
        lambda: __import__("argparse").Namespace(
            hf_path="cais/mmlu",
            hf_subset="all",
            split="test",
            output_name=None,
            filter_permutation_unsafe=False,
        ),
    )

    mod.main()

    df = pd.read_csv(tmp_path / "mmlu_normalized.csv")
    assert len(df) == 1  # kept — filter never ran
    assert not (tmp_path / "mmlu_permutation_filter.json").exists()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/scripts/test_prepare_data.py -v`
Expected: FAIL — `parse_args()` Namespace has no `filter_permutation_unsafe` attribute error is avoided by the monkeypatch itself, but `main()` doesn't yet reference it or produce the `_filtered` stem, so the new assertions on file existence/content fail (`out_csv` not found / wrong row count).

- [ ] **Step 3: Implement**

Edit `scripts/prepare_data.py`:

```python
from choicebench.benchmarks.registry import get_by_hf_path
from choicebench.config.paths import PROCESSED_DIR, ensure_dirs
from choicebench.permutation_filter import (
    filter_permutation_unsafe,
    permutation_filter_path_for,
    write_permutation_filter_report,
)
from choicebench.stats import compute_benchmark_stats, stats_path_for, write_stats
```

```python
    parser.add_argument(
        "--output-name", default=None,
        help="CSV filename stem; derived from --hf-path if omitted",
    )
    parser.add_argument(
        "--filter-permutation-unsafe", action="store_true",
        help=(
            "Exclude questions whose option text is meta-referential (e.g. "
            "'A and B', 'none of the above') — see choicebench.permutation_filter. "
            "Opt-in; never applied unless this flag is passed. Unless "
            "--output-name is also given, '_filtered' is appended to the "
            "resolved stem so this never overwrites the canonical unfiltered "
            "normalized CSV that other experiments read."
        ),
    )
    return parser.parse_args()
```

Replace `main()`:

```python
def main() -> None:
    ensure_dirs()
    args = parse_args()

    stem = _resolve_output_stem(args.output_name, args.hf_path, args.hf_subset)
    if args.filter_permutation_unsafe and not args.output_name:
        stem = f"{stem}_filtered"
    output_path = PROCESSED_DIR / f"{stem}_normalized.csv"

    if output_path.exists():
        logger.info("Normalized file already exists: %s — skipping download.", output_path)
        # Backfill the stats sidecar if it is missing (e.g. a CSV prepared
        # before stats existed), so the modal-k gate has data to read.
        if not stats_path_for(output_path).exists():
            logger.info("Stats sidecar missing — computing from existing CSV.")
            _write_stats(pd.read_csv(output_path), output_path, stem)
        return

    df_raw = download_from_huggingface(args.hf_path, args.hf_subset, args.split)
    logger.info("Downloaded %d rows.", len(df_raw))

    df_normalized = normalize_to_schema(df_raw, args.hf_path, args.hf_subset)
    logger.info("Normalized to %d rows.", len(df_normalized))

    if args.filter_permutation_unsafe:
        n_before = len(df_normalized)
        df_normalized, exclusions = filter_permutation_unsafe(df_normalized)
        logger.info(
            "Permutation-safety filter: excluded %d/%d question(s) (%.1f%%) with "
            "meta-referential options.",
            len(exclusions), n_before,
            (len(exclusions) / n_before * 100) if n_before else 0.0,
        )
        write_permutation_filter_report(
            exclusions, n_before, permutation_filter_path_for(output_path)
        )

    df_normalized.to_csv(output_path, index=False)
    logger.info("Saved → %s", output_path)
    _write_stats(df_normalized, output_path, stem)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/scripts/test_prepare_data.py -v`
Expected: PASS (all tests, including the three pre-existing ones, green)

- [ ] **Step 5: Commit**

```bash
git add scripts/prepare_data.py tests/scripts/test_prepare_data.py
git commit -m "feat(data): wire --filter-permutation-unsafe into prepare_data.py"
```

- [ ] **Step 6: Sanity-check the exclusion rate against the real MMLU dataset**

This is CPU-only (no GPU, no HPC) — `datasets` is already a core dependency,
so this runs anywhere with internet access. It is a sanity target, not a
constraint: the paper reports 13,592 of 14,042 MMLU test items surviving
(3.2% excluded); do not tune the regex patterns in
`src/choicebench/permutation_filter.py` to force this exact number — just run
it, record what this repo's conservative pattern set actually gets, and
document the delta.

Run:
```bash
python scripts/prepare_data.py --hf-path cais/mmlu --hf-subset all \
    --filter-permutation-unsafe --output-name mmlu_filtered
```
Expected: downloads MMLU test (14,042 rows), logs a line like
`Permutation-safety filter: excluded N/14042 question(s) (X.X%) with
meta-referential options.`, and writes
`data/processed/mmlu_filtered_normalized.csv` +
`data/processed/mmlu_filtered_permutation_filter.json`.

Record the actual `n_excluded`/`n_total` this run produced (from the sidecar
JSON's `n_excluded`/`n_total` fields) in this plan file's Task 6 YAML comment
block (the "Prerequisite" comment already references the paper's ~13,592
figure — append a one-line note there with the actual observed count once
this step has been run) and in the PR/commit description when this task is
integrated. Do not assert this exact count in any unit test — it depends on
live network access to the HuggingFace Hub and is not deterministic CI
input; the unit tests in Task 1 cover the pattern-matching logic in
isolation instead.

---

### Task 3: `pride_repro` prompt templates

**Files:**
- Create: `prompts/pride_repro/direct_mcq.txt`
- Create: `prompts/pride_repro/free_text.txt`
- Create: `prompts/pride_repro/option_matching.txt`

**Interfaces:**
- Consumes: nothing new — `load_prompt_templates("pride_repro", PROMPTS_DIR)` (existing function) requires all three filenames to exist for **any** prompt version, even though `direct_logprob`/`cyclic_logprob` only ever read `templates["direct_mcq"]`.
- Produces: the exact template string Task 4's `build_direct_mcq_prompt(subject=...)` renders against.

- [ ] **Step 1: Create `prompts/pride_repro/direct_mcq.txt`**

Write exactly this content (no trailing newline after `Answer:` — the file must end immediately after the colon, since `score_options()` reads the logit at the prompt's last token):

```
The following are multiple choice questions about {subject}. You should directly answer the question by choosing the correct option.

Question: {question}
Options:
{options}
Answer:
```

- [ ] **Step 2: Verify no trailing newline**

Run: `python3 -c "print(repr(open('prompts/pride_repro/direct_mcq.txt').read()[-10:]))"`
Expected: ends in `...\nAnswer:` with no `\n` after the final colon — output should be `'\nAnswer:'` (10 chars, no trailing `\n`).

- [ ] **Step 3: Create `prompts/pride_repro/free_text.txt` and `prompts/pride_repro/option_matching.txt`**

These are unused by `direct_logprob`/`cyclic_logprob` (both are single-stage logprob methods) but must exist because `load_prompt_templates()` always loads all three names for any prompt version. Copy the v1 files verbatim (byte-identical):

```bash
cp prompts/v1/free_text.txt prompts/pride_repro/free_text.txt
cp prompts/v1/option_matching.txt prompts/pride_repro/option_matching.txt
```

- [ ] **Step 4: Confirm the loader accepts the new version**

Run: `python3 -c "
from pathlib import Path
from choicebench.pipeline.prompt_builder import load_prompt_templates
t = load_prompt_templates('pride_repro', Path('prompts'))
assert set(t) == {'direct_mcq', 'free_text', 'option_matching'}
assert t['direct_mcq'].endswith('Answer:')
print('OK')
"`
Expected: prints `OK`

- [ ] **Step 5: Commit**

```bash
git add prompts/pride_repro/
git commit -m "feat(prompts): add pride_repro prompt version (paper's Figure 6 template)"
```

---

### Task 4: Subject-aware `build_direct_mcq_prompt()` + wire into `direct_logprob` and `cyclic_logprob`

**Files:**
- Modify: `src/choicebench/pipeline/prompt_builder.py`
- Modify: `src/choicebench/methods/library/direct_logprob.py`
- Modify: `src/choicebench/methods/library/permutation.py`
- Test: `tests/pipeline/test_prompt_builder.py` (append)
- Test: `tests/runners/test_direct_logprob.py` (append)
- Test: `tests/runners/test_cyclic_logprob.py` (append)

**Interfaces:**
- Consumes: `question_row["subject"]` (already present on every normalized row, in every benchmark).
- Produces: `build_direct_mcq_prompt(template, question, options, subject=None) -> str` — new optional 4th parameter. Formats `subject` as `subject.replace("_", " ")` before substitution (matches the paper's reference implementation convention). If `subject` is `None`, it is omitted from the `.format()` kwargs entirely (so a template without `{subject}` — e.g. v1's — is unaffected; a template *with* `{subject}` and no subject passed raises `KeyError` from `str.format`, a fail-fast signal rather than literally rendering the word "None").

- [ ] **Step 1: Write the failing tests**

```python
# Append to tests/pipeline/test_prompt_builder.py

class TestBuildDirectMcqPromptSubject:
    """Tests for the optional subject parameter (pride_repro template)."""

    _PRIDE_REPRO_TEMPLATES = load_prompt_templates("pride_repro", _PROMPTS_DIR)

    def test_subject_underscores_become_spaces(self):
        prompt = build_direct_mcq_prompt(
            self._PRIDE_REPRO_TEMPLATES["direct_mcq"],
            question="Q?",
            options={"A": "one", "B": "two"},
            subject="abstract_algebra",
        )
        assert "about abstract algebra." in prompt
        assert "abstract_algebra" not in prompt

    def test_prompt_ends_exactly_at_answer_colon(self):
        prompt = build_direct_mcq_prompt(
            self._PRIDE_REPRO_TEMPLATES["direct_mcq"],
            question="Q?",
            options={"A": "one", "B": "two"},
            subject="anatomy",
        )
        assert prompt.endswith("Answer:")

    def test_v1_template_ignores_absent_subject(self):
        # v1's direct_mcq.txt has no {subject} placeholder; omitting subject
        # (the default) must not raise and must not alter existing behavior.
        prompt = build_direct_mcq_prompt(
            _TEMPLATES["direct_mcq"], "Q?", {"A": "one", "B": "two"}
        )
        assert "Q?" in prompt

    def test_pride_repro_template_without_subject_raises_keyerror(self):
        with pytest.raises(KeyError):
            build_direct_mcq_prompt(
                self._PRIDE_REPRO_TEMPLATES["direct_mcq"], "Q?", {"A": "one", "B": "two"}
            )
```

Add the missing import at the top of the file:

```python
import pytest
```

```python
# Append to tests/runners/test_direct_logprob.py

class TestDirectLogprobPrideReproPrompt:
    def _make_pride_repro_runner(self, backend):
        return DirectLogprobRunner(
            backend=backend,
            method_name="direct_logprob",
            split_name="test",
            prompt_version="pride_repro",
            prompts_dir=_PROMPTS_DIR,
            run_id="test_run",
        )

    def test_prompt_uses_formatted_subject_and_ends_at_answer(self, runner_question_row):
        from tests.runners.conftest import MockBackend

        backend = MockBackend(score_responses=[[-0.1, -2.0, -3.0, -4.0]], supports_logprobs=True)
        row = self._make_pride_repro_runner(backend).run_one(runner_question_row, sample_index=0)

        # runner_question_row["subject"] == "computer_security"
        assert "about computer security." in row["prompt"]
        assert row["prompt"].endswith("Answer:")
```

```python
# Append to tests/runners/test_cyclic_logprob.py

class TestCyclicLogprobPrideReproPrompt:
    def test_prompt_uses_formatted_subject_and_ends_at_answer(self, runner_question_row):
        backend = MockBackend(
            score_responses=[[-0.1, -2.0, -3.0, -4.0]] * 4,
            supports_logprobs=True,
        )
        runner = _make_runner(backend, prompt_version="pride_repro")
        row = runner.run_one(runner_question_row, sample_index=0)

        # runner_question_row["subject"] == "computer_security"
        assert "about computer security." in row["prompt"]
        assert row["prompt"].endswith("Answer:")
```

`_make_runner(backend, **kw)` (defined at the top of this file) already forwards
`**kw` into `CyclicLogprobRunner(...)`, overriding its default
`prompt_version="v1"` — no changes to the helper itself are needed.

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/pipeline/test_prompt_builder.py tests/runners/test_direct_logprob.py tests/runners/test_cyclic_logprob.py -v`
Expected: FAIL — `build_direct_mcq_prompt() got an unexpected keyword argument 'subject'`

- [ ] **Step 3: Implement `build_direct_mcq_prompt`**

Edit `src/choicebench/pipeline/prompt_builder.py`:

```python
def build_direct_mcq_prompt(
    template: str,
    question: str,
    options: dict[str, str],
    subject: str | None = None,
) -> str:
    """Format the direct MCQ template with question, option text, and subject.

    Args:
        template: Raw template string from load_prompt_templates.
        question: Question stem to present to the model.
        options: Mapping from answer label to option text.
        subject: Optional subject/category label (e.g. MMLU's per-question
            subject). Underscores are replaced with spaces before
            substitution, matching the paper's reference implementation
            convention (e.g. "abstract_algebra" -> "abstract algebra"). Only
            templates with a {subject} placeholder need this (v1's does not);
            omitted entirely from .format()'s kwargs when None, so a template
            that does reference {subject} without one being supplied raises
            KeyError rather than silently rendering the literal word "None".

    Returns:
        Fully formatted prompt string.
    """
    format_kwargs = {"question": question, "options": _build_options_block(options)}
    if subject is not None:
        format_kwargs["subject"] = subject.replace("_", " ")
    return template.format(**format_kwargs)
```

- [ ] **Step 4: Wire `subject` into `direct_logprob.py`**

Edit `src/choicebench/methods/library/direct_logprob.py`, `_build_prompt`:

```python
    def _build_prompt(self, question_row: Any) -> str:
        return build_direct_mcq_prompt(
            template=self._prompts["direct_mcq"],
            question=question_row["question_text"],
            options=self._build_options(question_row),
            subject=question_row["subject"],
        )
```

- [ ] **Step 5: Wire `subject` into `permutation.py` (covers both `cyclic_permutation` and `cyclic_logprob`)**

Edit `src/choicebench/methods/library/permutation.py`, `_build_permuted_prompt`:

```python
    @staticmethod
    def _build_permuted_prompt(
            question_row: Any,
            permuted_options: dict[str, str],
            template: str,
    ) -> str:
        """Build a direct MCQ prompt using a permuted option ordering.

        Args:
            question_row: Normalized question record.
            permuted_options: Permuted letter-to-text mapping.
            template: Raw direct_mcq template string.

        Returns:
            Fully formatted prompt string with permuted options.
        """
        return build_direct_mcq_prompt(
            template=template,
            question=question_row["question_text"],
            options=permuted_options,
            subject=question_row["subject"],
        )
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `python -m pytest tests/pipeline/test_prompt_builder.py tests/runners/test_direct_logprob.py tests/runners/test_cyclic_logprob.py tests/runners/test_permutation.py -v`
Expected: PASS (all green, including pre-existing permutation/cyclic tests using v1 templates — subject is passed through but v1's template ignores it)

- [ ] **Step 7: Run the full suite (this task touches two shared call sites used by four methods)**

Run: `python -m pytest -q`
Expected: PASS, no regressions

- [ ] **Step 8: Commit**

```bash
git add src/choicebench/pipeline/prompt_builder.py src/choicebench/methods/library/direct_logprob.py src/choicebench/methods/library/permutation.py tests/pipeline/test_prompt_builder.py tests/runners/test_direct_logprob.py tests/runners/test_cyclic_logprob.py
git commit -m "feat(prompts): thread per-question subject through direct_logprob/cyclic_logprob prompts"
```

---

### Task 5: `add_bos_token` model config field, threaded into `HuggingFaceBackend`

**Files:**
- Modify: `src/choicebench/config/schema.py`
- Modify: `src/choicebench/backends/hf_backend.py`
- Modify: `scripts/run_experiment.py`
- Test: `tests/config/test_schema.py` (append)
- Test: `tests/backends/test_hf_backend.py` (append + update existing fake)
- Test: `tests/scripts/test_build_backend.py` (update one existing assertion)

**Interfaces:**
- Produces: `ModelConfig.add_bos_token: bool = True` (default preserves current behavior — BOS is currently always added by the tokenizer's default `add_special_tokens=True`). `HuggingFaceBackend.__init__(model_name_or_path, device="cuda", add_bos_token: bool = True, **generation_kwargs)`; both `generate()` and `score_options()` now call `self._tokenizer(prompt, return_tensors="pt", add_special_tokens=self._add_bos_token)`.

- [ ] **Step 1: Write the failing tests**

```python
# Append to tests/config/test_schema.py
# Uses this file's existing _valid_config()/_write_config(tmp_path, data)
# helpers (see the top of the file) — same pattern as test_generation_kwargs_parsed.

def test_add_bos_token_defaults_true(tmp_path):
    cfg = load_config(_write_config(tmp_path, _valid_config()))
    assert cfg.models[0].add_bos_token is True


def test_add_bos_token_can_be_set_false(tmp_path):
    data = _valid_config()
    data["models"][0]["add_bos_token"] = False
    cfg = load_config(_write_config(tmp_path, data))
    assert cfg.models[0].add_bos_token is False
```

```python
# Append to tests/backends/test_hf_backend.py

def test_score_options_passes_add_special_tokens_true_by_default():
    backend = HuggingFaceBackend("fake-model", "cpu")
    tokenizer = _FakeTokenizer()
    backend._loaded = True
    backend._tokenizer = tokenizer
    backend._model = _FakeScoringModel()
    backend._torch = _FakeTorch()
    backend._F = _FakeF()

    backend.score_options("prompt", ["A", "B"])

    assert tokenizer.add_special_tokens == True  # noqa: E712


def test_score_options_passes_add_special_tokens_false_when_bos_disabled():
    backend = HuggingFaceBackend("fake-model", "cpu", add_bos_token=False)
    tokenizer = _FakeTokenizer()
    backend._loaded = True
    backend._tokenizer = tokenizer
    backend._model = _FakeScoringModel()
    backend._torch = _FakeTorch()
    backend._F = _FakeF()

    backend.score_options("prompt", ["A", "B"])

    assert tokenizer.add_special_tokens == False  # noqa: E712


def test_generate_passes_add_special_tokens_false_when_bos_disabled():
    backend = HuggingFaceBackend("fake-model", "cpu", add_bos_token=False)
    tokenizer = _FakeTokenizer()
    model = _FakeModel()
    torch = _FakeTorch()
    backend._loaded = True
    backend._tokenizer = tokenizer
    backend._model = model
    backend._torch = torch

    backend.generate("prompt")

    assert tokenizer.add_special_tokens == False  # noqa: E712
```

`HuggingFaceBackend` will now call
`self._tokenizer(prompt, return_tensors="pt", add_special_tokens=...)` in both
`generate()` and `score_options()`, and `score_options()` also calls
`self._tokenizer.encode(opt, add_special_tokens=False)` and indexes
`outputs.logits[0, -1, :]` (a single `__getitem__((0, -1, slice(None, None, None)))`
call, not three chained lookups — Python packs a comma-separated subscript
into one tuple argument). Extend the top of the file with these exact fakes
(the existing `_FakeTensor`, `_FakeTokenizer`, `_FakeModel`, `_NoGrad`,
`_FakeTorch` classes stay as they are; `_FakeTokenizer` gains the two new
pieces shown):

```python
class _FakeTokenizer:
    eos_token_id = 0

    def __call__(self, prompt, return_tensors, add_special_tokens=True):
        self.prompt = prompt
        self.return_tensors = return_tensors
        self.add_special_tokens = add_special_tokens
        return {"input_ids": _FakeTensor()}

    def encode(self, text, add_special_tokens=False):
        return [ord(text)]  # options are single-char labels, e.g. "A" -> 65

    def decode(self, generated_ids, skip_special_tokens):
        self.generated_ids = generated_ids
        self.skip_special_tokens = skip_special_tokens
        return "decoded"


class _FakeScalar:
    """Wraps a plain float so `.item()` (as real torch tensors expose) works."""

    def __init__(self, value: float):
        self._value = value

    def item(self) -> float:
        return self._value


class _FakeLogitsRow(dict):
    """token_id -> logit, wrapping each lookup in a _FakeScalar for `.item()`."""

    def __getitem__(self, key):
        return _FakeScalar(dict.__getitem__(self, key))


class _FakeLogitsTensor:
    """Stands in for outputs.logits; supports the exact outputs.logits[0, -1, :] call."""

    def __init__(self, row: _FakeLogitsRow):
        self._row = row

    def __getitem__(self, index):
        assert index == (0, -1, slice(None, None, None))
        return self._row


class _FakeScoreOutputs:
    def __init__(self, row: _FakeLogitsRow):
        self.logits = _FakeLogitsTensor(row)


class _FakeScoringModel:
    def __init__(self):
        self._row = _FakeLogitsRow({ord("A"): -0.1, ord("B"): -1.5})

    def __call__(self, **kwargs):
        return _FakeScoreOutputs(self._row)


class _FakeF:
    @staticmethod
    def log_softmax(x, dim):
        return x  # identity — these tests only assert on the tokenizer kwarg
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/config/test_schema.py tests/backends/test_hf_backend.py -v`
Expected: FAIL — `ConfigError`/`AttributeError` (no `add_bos_token` field) and `TypeError: __init__() got an unexpected keyword argument 'add_bos_token'` / `TypeError: __call__() got an unexpected keyword argument 'add_special_tokens'`

- [ ] **Step 3: Implement schema.py changes**

Edit `src/choicebench/config/schema.py`:

```python
@dataclass
class ModelConfig:
    backend: str
    model_name_or_path: str
    provider: str | None = None  # Only required for API backends
    device: str = "cuda"
    generation_kwargs: GenerationKwargsConfig = field(default_factory=GenerationKwargsConfig)
    # Max in-flight requests for this model (API backends only). None means
    # "inherit run.concurrency_limit", resolved when the backend is built.
    concurrency_limit: int | None = None
    base_url: str | None = None  # vLLM server address; ignored for all other providers
    # HuggingFaceBackend only: whether the tokenizer prepends a BOS token
    # (transformers' add_special_tokens, applied to both generate() and
    # score_options()). Default True preserves prior behavior (most
    # tokenizers' own default). Zheng et al., ICLR 2024 (arXiv:2309.03882)
    # do not prepend BOS for open-source models — set False to match.
    # Ignored by api/dummy backends.
    add_bos_token: bool = True
```

```python
def _build_models(raw: dict) -> list[ModelConfig]:
    models = []
    for i, entry in enumerate(raw.get("models", [])):
        backend = _require(entry, "backend", f"models[{i}]")
        if backend not in _VALID_BACKENDS:
            raise ConfigError(
                f"model.backend must be one of {sorted(_VALID_BACKENDS)}; got {backend!r}."
            )
        device = entry.get("device", "cuda")
        if device not in _VALID_DEVICES:
            raise ConfigError(
                f"model.device must be one of {sorted(_VALID_DEVICES)}; got {device!r}."
            )
        if backend == "api" and "provider" not in entry:
            raise ConfigError("model.provider is required for API backends.")
        models.append(
            ModelConfig(
                backend=backend,
                model_name_or_path=_require(entry, "model_name_or_path", f"models[{i}]"),
                provider=entry.get("provider"),
                device=device,
                generation_kwargs=_build_generation_kwargs(entry.get("generation_kwargs")),
                concurrency_limit=(
                    int(entry["concurrency_limit"])
                    if entry.get("concurrency_limit") is not None
                    else None
                ),
                base_url=entry.get("base_url"),
                add_bos_token=bool(entry.get("add_bos_token", True)),
            )
        )
    return models
```

- [ ] **Step 4: Implement hf_backend.py changes**

Edit `src/choicebench/backends/hf_backend.py`:

```python
    def __init__(
        self,
        model_name_or_path: str,
        device: str = "cuda",
        add_bos_token: bool = True,
        **generation_kwargs,
    ) -> None:
        """
        Args:
            model_name_or_path: HuggingFace hub ID or local model directory path.
            device: "cuda", "cpu", "mps", or "auto" (device_map=auto for
                multi-GPU). Defaults to "cuda".
            add_bos_token: Whether to let the tokenizer prepend its BOS token
                (transformers' add_special_tokens), applied identically to
                generate() and score_options(). Default True matches most
                tokenizers' own default and prior behavior of this class.
                Zheng et al., ICLR 2024 (arXiv:2309.03882) do not prepend BOS
                for open-source models — pass False to reproduce that setup.
            **generation_kwargs: Default overrides for generate(), e.g.
                max_new_tokens=256, temperature=0.7, do_sample=True. Any of
                these can also be passed per-call to generate().
        """
        self._model_path = model_name_or_path
        self._device = device
        self._add_bos_token = add_bos_token
        self._default_generation_kwargs = generation_kwargs
        self._model = None
        self._tokenizer = None
        self._torch = None
        self._F = None
        self._loaded = False
```

```python
        inputs = self._tokenizer(
            prompt, return_tensors="pt", add_special_tokens=self._add_bos_token
        )
        prompt_len = inputs["input_ids"].shape[-1]
        input_device = self._get_input_device()
        inputs = {k: v.to(input_device) for k, v in inputs.items()}

        generate_kwargs: dict = {
```

(that edit is inside `generate()`, replacing its existing `inputs = self._tokenizer(prompt, return_tensors="pt")` line)

```python
        inputs = self._tokenizer(
            prompt, return_tensors="pt", add_special_tokens=self._add_bos_token
        )
        input_device = self._get_input_device()
        inputs = {k: v.to(input_device) for k, v in inputs.items()}

        with self._torch.no_grad():
            # Single forward pass; logits shape is (1, seq_len, vocab_size).
            outputs = self._model(**inputs)
```

(that edit is inside `score_options()`, replacing its existing `inputs = self._tokenizer(prompt, return_tensors="pt")` line)

- [ ] **Step 5: Implement run_experiment.py wiring**

Edit `scripts/run_experiment.py`'s `build_backend()`:

```python
    elif backend_type == "huggingface":
        backend = HuggingFaceBackend(
            model_config.model_name_or_path,
            model_config.device,
            add_bos_token=model_config.add_bos_token,
            max_new_tokens=model_config.generation_kwargs.max_new_tokens,
            temperature=model_config.generation_kwargs.temperature,
            do_sample=model_config.generation_kwargs.do_sample,
        )
        backend.load()  # load tokenizer + weights before any generate()/score_options() call
        return backend
```

- [ ] **Step 6: Fix the one existing assertion this change touches**

In `tests/scripts/test_build_backend.py`, `test_huggingface_backend_receives_generation_kwargs` currently asserts:

```python
    assert backend.generation_kwargs == {
        "max_new_tokens": 17,
        "temperature": 0.25,
        "do_sample": True,
    }
```

Since `_FakeHF.__init__(self, model_name_or_path, device, **generation_kwargs)` now also receives `add_bos_token=True` (the model default) via `**generation_kwargs`, update the expected dict:

```python
    assert backend.generation_kwargs == {
        "add_bos_token": True,
        "max_new_tokens": 17,
        "temperature": 0.25,
        "do_sample": True,
    }
```

- [ ] **Step 7: Run tests to verify they pass**

Run: `python -m pytest tests/config/test_schema.py tests/backends/test_hf_backend.py tests/scripts/test_build_backend.py -v`
Expected: PASS (all green)

- [ ] **Step 8: Run the full suite**

Run: `python -m pytest -q`
Expected: PASS, no regressions

- [ ] **Step 9: Commit**

```bash
git add src/choicebench/config/schema.py src/choicebench/backends/hf_backend.py scripts/run_experiment.py tests/config/test_schema.py tests/backends/test_hf_backend.py tests/scripts/test_build_backend.py
git commit -m "feat(config): add add_bos_token model config field for HuggingFaceBackend"
```

---

### Task 6: `examples/pride_reproduction.yaml` + companion sbatch

**Files:**
- Create: `examples/pride_reproduction.yaml`
- Create: `examples/hpc/run_pride_reproduction.sbatch`

**Interfaces:**
- Consumes: Tasks 1–5 (permutation filter, `pride_repro` prompts, `add_bos_token`, `direct_logprob`/`cyclic_logprob` methods, `recall_rstd`/`mad`/`accuracy` metrics — all already exist).
- No new code interfaces — this is a config artifact, verified in Task 8 by loading it with the dummy backend swapped in.

- [ ] **Step 1: Write `examples/pride_reproduction.yaml`**

```yaml
# examples/pride_reproduction.yaml
#
# Single-cell reproduction of Zheng et al., ICLR 2024 (arXiv:2309.03882),
# Table 3, MMLU block, llama-13B row, 0-shot. See
# examples/pride_reproduction_targets.json for the official numbers this
# run's metrics should land near.
#
# huggyllama/llama-13b is LLaMA-1 (NOT Llama-2) — the paper's "llama-13B".
#
# PriDe itself is deliberately NOT a method below. PriDe (every alpha/seed)
# is recomputed offline from the option_distributions_json this run persists
# for direct_logprob/cyclic_logprob — see the (separate) offline recompute
# script. Running PriDe here would waste GPU time re-deriving what the
# offline script already gets from these two methods' logprobs.
#
# Prerequisite — prepare the permutation-filtered MMLU CSV first:
#   python scripts/prepare_data.py --hf-path cais/mmlu --hf-subset all \
#       --filter-permutation-unsafe --output-name mmlu_filtered
# This writes data/processed/mmlu_filtered_normalized.csv (~13,592 of 14,042
# rows survive per the paper's ~3.2% exclusion) plus a
# mmlu_filtered_permutation_filter.json accounting sidecar.

experiment:
  name: pride_reproduction_mmlu_llama13b_0shot

models:
  - backend: huggingface
    model_name_or_path: huggyllama/llama-13b
    # revision: <PIN_BEFORE_RUN>  # pin the exact HF commit SHA before submitting the sbatch job
    device: cuda

    # Zheng et al. do not prepend BOS for open-source models (Figure 6
    # caption, arXiv:2309.03882) — the HuggingFaceBackend default is True
    # (add BOS), so this must be set explicitly for a faithful reproduction.
    add_bos_token: false

    # direct_logprob/cyclic_logprob never call generate() — they only use
    # score_options() (single forward pass, no sampling) — so
    # generation_kwargs below is unused by this run; left at safe defaults.
    generation_kwargs:
      max_new_tokens: 1
      temperature: 0.0
      do_sample: false

# fp16 is HuggingFaceBackend's fixed load() dtype (torch_dtype=torch.float16),
# not a config knob — see examples/hpc/run_pride_reproduction.sbatch for the
# VRAM sizing that assumes it.

benchmarks:
  # NOTE: name: huggingface (not name: mmlu) — a registered benchmark name's
  # CSV path is hardcoded to "<name>_normalized.csv" and ignores output_name,
  # which would silently point this run at the *unfiltered* mmlu_normalized.csv.
  # The generic huggingface path is the only one that honors output_name.
  - name: huggingface
    hf_path: cais/mmlu
    hf_subset: all
    split: test
    output_name: mmlu_filtered
    n_samples: null
    subject_filter: null

methods:
  - name: direct_logprob
    requires_logprobs: true
  - name: cyclic_logprob
    requires_logprobs: true

metrics:
  - accuracy
  - recall_rstd
  - mad

run:
  seed: 42
  resume: true
  dry_run: false
  checkpoint_every_n: 50
  prompt_version: "pride_repro"
  concurrency_limit: 10
```

- [ ] **Step 2: Validate the YAML loads (dummy-backend substitution) before writing the sbatch**

Run:
```bash
python3 -c "
import yaml
from choicebench.config.schema import load_config
raw = yaml.safe_load(open('examples/pride_reproduction.yaml'))
raw['models'][0]['backend'] = 'dummy'
import tempfile, pathlib
p = pathlib.Path(tempfile.mktemp(suffix='.yaml'))
p.write_text(yaml.dump(raw))
cfg = load_config(str(p))
assert [m.name for m in cfg.methods] == ['direct_logprob', 'cyclic_logprob']
assert cfg.run.prompt_version == 'pride_repro'
print('OK')
"
```
Expected: prints `OK` (proves the YAML is syntactically valid and schema-passes once the backend is swapped to dummy — the real `huggingface` backend requires torch/transformers, deferred to Task 8's fuller dry run)

- [ ] **Step 3: Write `examples/hpc/run_pride_reproduction.sbatch`**

```bash
#!/usr/bin/env bash
# Slurm template for the PriDe single-cell reproduction (MMLU / llama-13B /
# 0-shot / direct_logprob + cyclic_logprob only — see
# examples/pride_reproduction.yaml for the full design rationale).
#
# VRAM sizing: huggyllama/llama-13b in fp16 is ~26GB of weights alone;
# activations for a single forward pass (no generation — both methods here
# are score_options()-only) add a modest amount on top. A single 40GB or
# 80GB A100 is sufficient; this does not need multi-GPU device_map=auto.
#
# Cost shape: direct_logprob is 1 forward pass/question; cyclic_logprob is
# N forward passes/question (N=4 for MMLU's 4 options). Over the ~13,592
# permutation-filtered MMLU test questions that is ~5x13,592 ≈ 68,000 total
# forward passes — the --time budget below is a generous upper bound for a
# single A100, not a tuned estimate.

#SBATCH --job-name=pride_repro
#SBATCH --output=logs/pride_repro_%j.out
#SBATCH --error=logs/pride_repro_%j.err
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
##SBATCH --partition=gpu

set -euo pipefail

# BASH_SOURCE breaks here: sbatch copies the submitted script to a spool
# directory before executing it, so BASH_SOURCE resolves to the spool path,
# not this file's real location. SLURM_SUBMIT_DIR is the directory sbatch
# was invoked from and stays stable across that copy.
REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
SCRIPT_DIR="$REPO_ROOT/examples/hpc"

mkdir -p "$REPO_ROOT/logs"

# shellcheck source=examples/hpc/env_hpc.sh
source "$SCRIPT_DIR/env_hpc.sh"

CONFIG="${CHOICEBENCH_CONFIG:-examples/pride_reproduction.yaml}"
RUN_ID="${CHOICEBENCH_RUN_ID:-pride_reproduction_${SLURM_JOB_ID:-manual}}"

echo "Slurm job: ${SLURM_JOB_ID:-manual}"
echo "Node: ${SLURMD_NODENAME:-$(hostname)}"
echo "Config: $CONFIG"
echo "Run ID: $RUN_ID"

# Prerequisite: the permutation-filtered MMLU CSV must already exist (see the
# comment atop examples/pride_reproduction.yaml for the prepare_data.py
# invocation) — this job does not download/filter data itself.
if [[ ! -f "$REPO_ROOT/data/processed/mmlu_filtered_normalized.csv" ]]; then
    echo "ERROR: data/processed/mmlu_filtered_normalized.csv not found." >&2
    echo "Run: python scripts/prepare_data.py --hf-path cais/mmlu --hf-subset all --filter-permutation-unsafe --output-name mmlu_filtered" >&2
    exit 1
fi

python scripts/run_experiment.py \
    --config "$CONFIG" \
    --run-id "$RUN_ID" \
    --yes

python scripts/evaluate_run.py \
    --run-id "$RUN_ID"
```

- [ ] **Step 4: Make the sbatch executable**

Run: `chmod +x examples/hpc/run_pride_reproduction.sbatch`

- [ ] **Step 5: Shellcheck the sbatch (matches this repo's existing HPC scripts' quality bar)**

Run: `shellcheck examples/hpc/run_pride_reproduction.sbatch || true`
Expected: no new categories of warnings beyond what `shellcheck examples/hpc/run_choicebench.sbatch` already reports (SLURM directives look like comments to shellcheck and are expected to be flagged/ignored the same way in both files)

- [ ] **Step 6: Commit**

```bash
git add examples/pride_reproduction.yaml examples/hpc/run_pride_reproduction.sbatch
git commit -m "feat(examples): add PriDe reproduction run config and sbatch"
```

---

### Task 7: Official targets fixture

**Files:**
- Create: `examples/pride_reproduction_targets.json`

**Interfaces:**
- None — static data fixture, consumed by the (separate, future) offline PriDe recompute script's comparison step.

- [ ] **Step 1: Write the fixture with exactly this content**

```json
{
  "provenance": "Zheng et al., ICLR 2024 (arXiv:2309.03882), Table 3, MMLU block, llama-13B row, 0-shot. PriDe cells are means over 5 calibration-subset runs per the paper; no per-cell std reported.",
  "model": "llama-13B (LLaMA-1; huggyllama/llama-13b mirror)",
  "benchmark": "MMLU test, 4-option, post permutation-safety filter (paper N=13592)",
  "setting": "0-shot",
  "metrics_scale": "RStd and Acc on 0-100 scale",
  "targets": {
    "default":     {"rstd": 17.4, "acc": 34.6},
    "cyclic_perm": {"rstd": 2.9,  "acc": 47.7},
    "pride_5":     {"rstd": 5.7,  "acc": 36.4},
    "pride_40":    {"rstd": 3.9,  "acc": 40.4},
    "pride_80":    {"rstd": 2.6,  "acc": 45.3}
  }
}
```

- [ ] **Step 2: Validate it's well-formed JSON**

Run: `python3 -c "import json; json.load(open('examples/pride_reproduction_targets.json')); print('OK')"`
Expected: prints `OK`

- [ ] **Step 3: Commit**

```bash
git add examples/pride_reproduction_targets.json
git commit -m "feat(examples): add PriDe reproduction official-numbers fixture"

Note in the commit body (not the file — the file's content is fixed verbatim
per the task brief): these numbers were independently cross-checked against a
direct pdftotext extraction of arXiv:2309.03882's Table 3 (not a summary) and
match exactly; no discrepancy to record.
```

---

### Task 8: Dummy-backend end-to-end dry run, CHANGELOG, final verification

**Files:**
- Create (throwaway, not committed): a temporary copy of `examples/pride_reproduction.yaml` with `backend: dummy`
- Test: `tests/scripts/test_prepare_data.py` or a new `tests/test_pride_reproduction_wiring.py` — pick based on Step 1's outcome
- Modify: `CHANGELOG.md`

**Interfaces:**
- Consumes: everything from Tasks 1–7.
- Produces: nothing new — this task is pure verification plus the changelog entry.

- [ ] **Step 1: Write an end-to-end wiring test that proves filter → pride_repro prompts → both methods → all three metrics executes without a GPU**

```python
# tests/test_pride_reproduction_wiring.py

"""End-to-end proof that the PriDe reproduction config's full pipeline —
permutation filter -> pride_repro prompts -> direct_logprob/cyclic_logprob ->
accuracy/recall_rstd/mad -- executes with the dummy backend, before any GPU
time is spent on the real huggyllama/llama-13b run.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from choicebench.config.schema import load_config
from choicebench.metrics import BUILTIN_METRICS
from choicebench.methods.library.cyclic_logprob import CyclicLogprobRunner
from choicebench.methods.library.direct_logprob import DirectLogprobRunner
from choicebench.permutation_filter import filter_permutation_unsafe

_REPO_ROOT = Path(__file__).resolve().parents[1]
_PROMPTS_DIR = _REPO_ROOT / "prompts"


def _mmlu_style_rows() -> pd.DataFrame:
    """A handful of MMLU-shaped rows, one of which is permutation-unsafe."""
    from choicebench.benchmarks.base import make_normalized_row

    rows = [
        make_normalized_row("abstract_algebra", "1 + 1 = ?", ["1", "2", "3", "4"], 1),
        make_normalized_row("anatomy", "Which is a bone?", ["Femur", "Liver", "A and B", "Skin"], 0),
        make_normalized_row("astronomy", "Closest planet to the sun?", ["Venus", "Mercury", "Earth", "Mars"], 1),
    ]
    return pd.DataFrame(rows)


def test_config_loads_with_dummy_backend(tmp_path):
    raw = yaml.safe_load((_REPO_ROOT / "examples" / "pride_reproduction.yaml").read_text())
    raw["models"][0]["backend"] = "dummy"
    cfg_path = tmp_path / "dummy_pride_repro.yaml"
    cfg_path.write_text(yaml.dump(raw))

    cfg = load_config(str(cfg_path))

    assert [m.name for m in cfg.methods] == ["direct_logprob", "cyclic_logprob"]
    assert cfg.run.prompt_version == "pride_repro"
    assert cfg.metrics == ["accuracy", "recall_rstd", "mad"]


def test_full_pipeline_dummy_backend_smoke():
    from choicebench.backends.dummy_backend import DummyBackend

    df = _mmlu_style_rows()
    filtered, exclusions = filter_permutation_unsafe(df)
    assert len(exclusions) == 1
    assert len(filtered) == 2

    backend = DummyBackend(fixed_scores={"A": -0.1, "B": -1.5, "C": -2.0, "D": -2.5})

    direct_runner = DirectLogprobRunner(
        backend=backend, method_name="direct_logprob", split_name="test",
        prompt_version="pride_repro", prompts_dir=_PROMPTS_DIR, run_id="dry_run",
    )
    cyclic_runner = CyclicLogprobRunner(
        backend=backend, method_name="cyclic_logprob", split_name="test",
        prompt_version="pride_repro", prompts_dir=_PROMPTS_DIR, run_id="dry_run",
    )

    direct_rows = direct_runner.run_many(filtered)
    cyclic_rows = cyclic_runner.run_many(filtered)

    assert len(direct_rows) == len(filtered)
    assert len(cyclic_rows) == len(filtered)
    for row in direct_rows + cyclic_rows:
        assert "about " in row["prompt"]  # pride_repro subject sentence rendered
        assert row["prompt"].rstrip("\n").endswith("Answer:")
        assert "_" not in row["prompt"].split("about ", 1)[1].split(".", 1)[0]

    results_df = pd.DataFrame(direct_rows)
    for metric_name in ("accuracy", "recall_rstd", "mad"):
        metric = BUILTIN_METRICS[metric_name]()
        computed = metric.compute(results_df)
        assert isinstance(computed, dict)
        assert computed  # non-empty — every metric produced *something*
```

- [ ] **Step 2: Run the new test to verify it fails first (TDD sanity check), then passes**

Run: `python -m pytest tests/test_pride_reproduction_wiring.py -v`
Expected first (before Tasks 1–7 land, or if run standalone against a clean checkout): FAIL. Since this task runs last, after Tasks 1–7 are already committed, this should PASS immediately — if it does not, that is a real integration bug between the pieces (e.g. a subject-formatting edge case) to fix before moving on, not a test to weaken.

Run again to confirm: `python -m pytest tests/test_pride_reproduction_wiring.py -v`
Expected: PASS

- [ ] **Step 3: Add the CHANGELOG entry**

Edit `CHANGELOG.md`, prepending to the `## Unreleased` section (above the existing `direct_logprob` bullet, newest-first):

```markdown
- Added the PriDe (Zheng et al., ICLR 2024, arXiv:2309.03882) single-cell
  reproduction plumbing for MMLU / llama-13B (LLaMA-1) / 0-shot:
  `--filter-permutation-unsafe` on `scripts/prepare_data.py` (excludes
  meta-referential MMLU options like "A and B", "none of the above" —
  `choicebench.permutation_filter`), a new `prompts/pride_repro/` template
  version matching the paper's exact Figure 6 layout with per-question
  `subject` support in `build_direct_mcq_prompt()`, a new `add_bos_token`
  model config field (`HuggingFaceBackend` defaults to adding BOS; the paper
  does not for open-source models), `examples/pride_reproduction.yaml` +
  `examples/hpc/run_pride_reproduction.sbatch` (runs `direct_logprob` +
  `cyclic_logprob` only — PriDe itself is recomputed offline from their
  persisted distributions in a separate script), and
  `examples/pride_reproduction_targets.json` with the paper's official
  Table 3 numbers for this cell.
```

- [ ] **Step 4: Run the full suite**

Run: `python -m pytest -q`
Expected: PASS — should read `635 passed` (634 baseline + new files' tests) or higher once all of Tasks 1–8's new test files are counted; confirm the count is baseline + (new tests added across all tasks) with zero failures/errors.

- [ ] **Step 5: Commit**

```bash
git add tests/test_pride_reproduction_wiring.py CHANGELOG.md
git commit -m "test: add PriDe reproduction dummy-backend E2E wiring test; changelog"
```

---

## Post-plan note (out of scope, for context only)

The offline script that recomputes PriDe at every alpha/seed from `direct_logprob`/`cyclic_logprob`'s persisted `option_distributions_json` — and the script that ultimately compares its output against `examples/pride_reproduction_targets.json` — is explicitly the *next* task per the brief, not part of this plan. Nothing here should be extended to attempt that.
