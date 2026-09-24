# scripts/paper/freeze_stochasticity_manifests.py
#
# Paper-specific (eacl-2026-revision): freezes the 100-MMLU / 100-ARC
# fixed question-ID subsets used for the API stochasticity experiment,
# per the frozen protocol:
#
#   for each question_id in the candidate population:
#       key = f"stochasticity-subset-v1|{seed}|{benchmark_id}|{question_id}"
#       digest = blake2b(key.encode("utf-8"), digest_size=8).digest()
#   sort question_ids by digest (ascending, unsigned big-endian)
#   take the first 100
#
# Candidate population is the FINAL frozen evaluation manifest for each
# benchmark (mmlu_eval_v2.csv's 1,140 questions; arc_challenge_eval_v2.csv's
# 1,172), not the raw source data -- the stochasticity subset must be a
# subset of what's actually evaluated.
#
# Run with: python scripts/paper/freeze_stochasticity_manifests.py

from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_DIR = REPO_ROOT / "data" / "manifests"

SEED = 42
N_STOCHASTICITY = 100


def _digest(benchmark_id: str, question_id: str) -> bytes:
    key = f"stochasticity-subset-v1|{SEED}|{benchmark_id}|{question_id}"
    return hashlib.blake2b(key.encode("utf-8"), digest_size=8).digest()


def select_stochasticity_subset(question_ids: list[str], benchmark_id: str) -> list[str]:
    ranked = sorted(question_ids, key=lambda qid: _digest(benchmark_id, qid))
    return ranked[:N_STOCHASTICITY]


def main() -> None:
    mmlu = pd.read_csv(MANIFEST_DIR / "mmlu_eval_v2.csv")
    arc = pd.read_csv(MANIFEST_DIR / "arc_challenge_eval_v2.csv")

    mmlu_subset_ids = select_stochasticity_subset(mmlu["question_id"].tolist(), "mmlu")
    arc_subset_ids = select_stochasticity_subset(arc["question_id"].tolist(), "arc_challenge")

    assert len(mmlu_subset_ids) == N_STOCHASTICITY
    assert len(set(mmlu_subset_ids)) == N_STOCHASTICITY
    assert len(arc_subset_ids) == N_STOCHASTICITY
    assert len(set(arc_subset_ids)) == N_STOCHASTICITY

    mmlu_subset = mmlu[mmlu["question_id"].isin(mmlu_subset_ids)].sort_values("question_id")
    arc_subset = pd.DataFrame({"question_id": sorted(arc_subset_ids)})

    for df, name in [(mmlu_subset, "mmlu_stochasticity_v2.csv"),
                      (arc_subset, "arc_challenge_stochasticity_v2.csv")]:
        path = MANIFEST_DIR / name
        df.to_csv(path, index=False, quoting=csv.QUOTE_MINIMAL)
        print(f"Wrote {len(df)} rows -> {path}")

    print("\nMMLU stochasticity subset subject distribution:")
    print(mmlu_subset["subject"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
