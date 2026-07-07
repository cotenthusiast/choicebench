"""Reconciliation of the MMLU-duplicates finding against the raw upstream data.

Loads the RAW cais/mmlu `all` test split (no ChoiceBench normalization, no
permutation filter, no dedup guard) directly via `datasets.load_dataset`, and
measures duplicate rows under a ladder of strict -> loose key definitions.

Usage:
    python scripts/mmlu_duplicate_reconciliation.py
"""

from __future__ import annotations

import json
import re
from itertools import combinations
from pathlib import Path

import pandas as pd

EXPECTED_RAW_ROWS = 14_042
FILTERED_ROWS = 13_616

OUTPUT_JSON = Path(__file__).parent / "mmlu_strict_dupes.json"


def ws_norm(s: str) -> str:
    """Whitespace-normalize: strip and collapse internal runs of whitespace."""
    return re.sub(r"\s+", " ", s.strip())


def ws_case_norm(s: str) -> str:
    return ws_norm(s).lower()


def load_raw_mmlu() -> pd.DataFrame:
    """Load the raw upstream MMLU test split, pre-filtering, pre-dedup."""
    from datasets import load_dataset

    ds = load_dataset("cais/mmlu", "all", split="test")
    df = ds.to_pandas()

    rows = []
    for row in df.itertuples(index=False):
        choices = list(row.choices)
        if hasattr(choices, "tolist"):
            choices = choices.tolist()
        rows.append(
            {
                "stem": str(row.question),
                "choices": [str(c) for c in choices],
                "answer": int(row.answer),
                "subject": str(row.subject),
            }
        )
    return pd.DataFrame(rows)


def make_key_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Precompute the normalized fields the ladder's keys are built from."""
    df = df.copy()
    df["stem_ws"] = df["stem"].map(ws_norm)
    df["stem_ws_case"] = df["stem"].map(ws_case_norm)
    df["choices_ws"] = df["choices"].map(lambda cs: tuple(ws_norm(c) for c in cs))
    df["choices_ws_case"] = df["choices"].map(lambda cs: tuple(ws_case_norm(c) for c in cs))
    return df


def rung_stats(df: pd.DataFrame, key_cols: list[str], label: str) -> dict:
    total = len(df)
    grouped = df.groupby(key_cols, sort=False).size()
    dup_sizes = grouped[grouped >= 2]

    groups = len(dup_sizes)
    pairs = int(sum(size * (size - 1) // 2 for size in dup_sizes))
    redundant_rows = int(sum(size - 1 for size in dup_sizes))
    rows_in_dup_groups = int(sum(dup_sizes))

    return {
        "rung": label,
        "groups": groups,
        "pairs": pairs,
        "redundant_rows": redundant_rows,
        "redundant_rows_pct": round(redundant_rows / total * 100, 3) if total else 0.0,
        "rows_in_dup_groups": rows_in_dup_groups,
        "rows_in_dup_groups_pct": round(rows_in_dup_groups / total * 100, 3) if total else 0.0,
    }


def export_strict_groups(df: pd.DataFrame, key_cols: list[str], path: Path) -> int:
    """Export rung-1 (strict) duplicate groups to JSON. Returns group count."""
    grouped = df.groupby(key_cols, sort=False).indices
    groups = []
    for key, indices in grouped.items():
        if len(indices) < 2:
            continue
        idx = sorted(int(i) for i in indices)
        subject = df.iloc[idx[0]]["subject"]
        answer = int(df.iloc[idx[0]]["answer"])
        stem_preview = df.iloc[idx[0]]["stem_ws"][:80]
        groups.append(
            {
                "row_indices": idx,
                "size": len(idx),
                "subject": subject,
                "answer": answer,
                "stem_preview": stem_preview,
            }
        )
    groups.sort(key=lambda g: g["row_indices"][0])
    path.write_text(json.dumps(groups, indent=2))
    return len(groups)


def main() -> None:
    df = load_raw_mmlu()
    n = len(df)
    print(f"Loaded {n} raw MMLU test-split rows from cais/mmlu (subset=all).")

    if n == EXPECTED_RAW_ROWS:
        print(f"OK: row count matches expected raw split size ({EXPECTED_RAW_ROWS}).")
    elif n == FILTERED_ROWS:
        raise SystemExit(
            f"STOP: loaded {n} rows, which matches the POST-FILTER count "
            f"({FILTERED_ROWS}), not the raw split ({EXPECTED_RAW_ROWS}). "
            "This means the loader picked up the filtered set — the "
            "reconciliation below would be invalid. Aborting."
        )
    else:
        raise SystemExit(
            f"STOP: loaded {n} rows, which matches neither the expected raw "
            f"split ({EXPECTED_RAW_ROWS}) nor the known filtered count "
            f"({FILTERED_ROWS}). Aborting — investigate before trusting any "
            "reconciliation numbers."
        )

    df = make_key_columns(df)

    ladder = [
        ("1. stem+choices+subject+answer (strict, ws)", ["stem_ws", "choices_ws", "subject", "answer"]),
        ("2. stem+choices+answer (drop subject)", ["stem_ws", "choices_ws", "answer"]),
        ("3. stem+choices (drop answer)", ["stem_ws", "choices_ws"]),
        ("4. stem+choices (ws+case)", ["stem_ws_case", "choices_ws_case"]),
        ("5. stem+answer (ignore choices)", ["stem_ws", "answer"]),
        ("6. stem only (ws)", ["stem_ws"]),
        ("7. stem only (ws+case)", ["stem_ws_case"]),
    ]

    results = [rung_stats(df, key_cols, label) for label, key_cols in ladder]

    header = (
        f"{'Rung':<45} {'groups':>8} {'pairs':>8} {'redund':>8} {'redund%':>8} "
        f"{'in-grp':>8} {'in-grp%':>8}"
    )
    print()
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r['rung']:<45} {r['groups']:>8} {r['pairs']:>8} "
            f"{r['redundant_rows']:>8} {r['redundant_rows_pct']:>7.3f}% "
            f"{r['rows_in_dup_groups']:>8} {r['rows_in_dup_groups_pct']:>7.3f}%"
        )

    n_strict_groups = export_strict_groups(df, ["stem_ws", "choices_ws", "subject", "answer"], OUTPUT_JSON)

    print()
    print("Reconciliation note:")
    print(
        f"- Total raw rows: {n} (cais/mmlu, subset=all, split=test — pre-filter, pre-dedup).\n"
        f"- Rung 1 (strict: stem+choices+subject+answer, ws-normalized) is the exact "
        f"criterion used by the repo's own permutation-safety dedup guard.\n"
        f"- 'groups' counts duplicate clusters; 'pairs' counts C(size,2) pairwise "
        f"collisions within each cluster; 'redundant_rows' counts rows beyond the "
        f"first per cluster (what dedup would actually remove); 'rows_in_dup_groups' "
        f"counts every row that participates in a cluster (numerator differs from "
        f"redundant_rows whenever any cluster has size > 2).\n"
        f"- Looser rungs (2-7) progressively drop fields from the key and are "
        f"expected to inflate all four counts monotonically; they characterize how "
        f"much of the 'duplicate' signal is coincidental stem/choice overlap versus "
        f"true verbatim duplication."
    )
    print(f"- Rung-1 strict duplicate groups exported: {n_strict_groups} groups -> {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
