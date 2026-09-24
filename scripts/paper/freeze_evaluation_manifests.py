# scripts/paper/freeze_evaluation_manifests.py
#
# Paper-specific (eacl-2026-revision): generates the frozen MMLU and ARC
# evaluation-set manifests specified in the frozen scientific specification.
#
# MMLU: exactly 57 subjects x 20 seeded questions/subject = 1,140, drawn via
# the same per-group sampling mechanism (pandas groupby().sample(),
# random_state=seed) that choicebench's sampling: {strategy: per_group}
# config uses -- this script exists to freeze the resulting IDs to disk
# once, not to re-derive them at run time on every launch.
#
# ARC-Challenge: the complete 1,172-question test split, no sampling.
#
# Run with: python scripts/paper/freeze_evaluation_manifests.py

from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_DIR = REPO_ROOT / "data" / "manifests"

MMLU_SOURCE = REPO_ROOT / "data/processed/mmlu/test/norm-v2/src_705fa6e8972d/normalized.csv"
ARC_SOURCE = REPO_ROOT / "data/processed/arc_challenge_normalized.csv"

SEED = 42
N_PER_SUBJECT = 20


def freeze_mmlu() -> pd.DataFrame:
    df = pd.read_csv(MMLU_SOURCE)
    sampled = df.groupby("subject", group_keys=False).sample(
        n=N_PER_SUBJECT, random_state=SEED
    )
    n_subjects = sampled["subject"].nunique()
    assert n_subjects == 57, f"expected 57 MMLU subjects, got {n_subjects}"
    assert len(sampled) == 57 * N_PER_SUBJECT, f"expected 1140 rows, got {len(sampled)}"
    assert sampled["question_id"].is_unique, "duplicate question_id in MMLU sample"
    return sampled[["question_id", "subject"]].sort_values("question_id").reset_index(drop=True)


def freeze_arc() -> pd.DataFrame:
    df = pd.read_csv(ARC_SOURCE)
    assert len(df) == 1172, f"expected 1172 ARC-Challenge rows, got {len(df)}"
    assert df["question_id"].is_unique, "duplicate question_id in ARC-Challenge split"
    return df[["question_id"]].sort_values("question_id").reset_index(drop=True)


def _write_manifest(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, quoting=csv.QUOTE_MINIMAL)
    print(f"Wrote {len(df)} rows -> {path}")


def main() -> None:
    _write_manifest(freeze_mmlu(), MANIFEST_DIR / "mmlu_eval_v2.csv")
    _write_manifest(freeze_arc(), MANIFEST_DIR / "arc_challenge_eval_v2.csv")


if __name__ == "__main__":
    main()
