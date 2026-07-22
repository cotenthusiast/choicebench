"""Derive 'baseline' per-permutation flip-rate traces from the existing
cyclic_generation_majority traces.

No new model calls. `baseline` (DirectMCQRunner) and `cyclic`
(PermutationRunner) issue the IDENTICAL single-call direct-MCQ prompt per
permutation, with identical settings (temperature=0.0, max_tokens=500,
seed=42, prompt_version=v1 -- verified against config/default.yaml and the
historical run's own stored columns). A cyclic permutation's raw response IS
what baseline would have produced for that same permutation. This script
copies the existing trace rows verbatim except for cell_id/method labeling,
so the two labeled outputs (baseline, cyclic) are traceably identical data,
not a re-derivation.
"""
from __future__ import annotations

import glob
from pathlib import Path

import pandas as pd

SRC_DIR = Path("/home/cotenthusiast/Projects/final_paper_run_bundle_EXPANDED_20260720T170604Z/flip_rate_traces")
OUT_DIR = Path(__file__).resolve().parent / "traces"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    for sub in ("api", "local"):
        for src_path in sorted(glob.glob(str(SRC_DIR / sub / "*.csv"))):
            df = pd.read_csv(src_path)
            assert df["cell_id"].nunique() == 1
            src_cell_id = df["cell_id"].iloc[0]
            assert src_cell_id.endswith("cyclic_generation_majority_traced_v2")

            baseline_cell_id = src_cell_id.replace(
                "cyclic_generation_majority_traced_v2", "baseline_traced_v2"
            )
            out = df.copy()
            out["cell_id"] = baseline_cell_id
            out["source_note"] = "relabeled_from_cyclic_generation_majority_traces_no_new_calls"

            dest = OUT_DIR / f"baseline_{Path(src_path).name}"
            out.to_csv(dest, index=False)
            print(f"{src_path} ({len(df)} rows) -> {dest}")


if __name__ == "__main__":
    main()
