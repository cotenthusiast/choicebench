# scripts/paper/derive_baseline_from_cyclic.py
#
# Paper-specific (eacl-2026-revision): the frozen protocol requires
# baseline (direct_mcq) to BE cyclic_permutation's own canonical/rotation-0
# observation -- one target-model observation, not a second independent
# generation merely to produce baseline. This script is the paper-side
# wiring that reads a saved cyclic_permutation (or reasoning_cyclic) result
# CSV and derives direct_mcq (or reasoning_mcq) from it, writing the result
# out in the normal result-row CSV schema so downstream metrics tooling
# doesn't need to know it wasn't a live run. See
# choicebench.analysis.derive_baseline_from_cyclic for why this is exactly
# rotation 0's own observation, re-scored, rather than a copy of the
# source row's own (majority-vote) parsed_choice.
#
# direct_mcq no longer appears in config/paper/{mmlu,arc}_core_methods.yaml's
# methods: list -- it is produced by this script from that benchmark's own
# mmlu_core_methods_cyclic_batch.yaml / arc_core_methods_cyclic_batch.yaml
# cyclic_permutation run instead. Same relationship for reasoning_mcq <-
# reasoning_cyclic (pass --method-name reasoning_mcq).
#
# Run with:
#   python scripts/paper/derive_baseline_from_cyclic.py \
#       --source runs/<run_id>/<..._cyclic_permutation_..._<model>_<benchmark>.csv> \
#       --output runs/<run_id>/<..._direct_mcq_..._<model>_<benchmark>.csv>

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from choicebench.analysis.derive_baseline_from_cyclic import derive_baseline_from_cyclic

METHOD_NAME = "direct_mcq"


def derive(source_path: Path, output_path: Path, method_name: str = METHOD_NAME) -> pd.DataFrame:
    source_df = pd.read_csv(source_path)
    derived_df = derive_baseline_from_cyclic(source_df, method_name=method_name)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    derived_df.to_csv(output_path, index=False)
    return derived_df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path,
                         help="Path to a saved cyclic_permutation (or reasoning_cyclic) result CSV.")
    parser.add_argument("--output", required=True, type=Path,
                         help="Path to write the derived direct_mcq (or reasoning_mcq) result CSV.")
    parser.add_argument("--method-name", default=METHOD_NAME,
                         help="Stamped on output rows; pass 'reasoning_mcq' to reuse this "
                              "script for reasoning_mcq's derivation from reasoning_cyclic.")
    args = parser.parse_args()

    derived_df = derive(args.source, args.output, method_name=args.method_name)
    n_scored = derived_df["parsed_choice"].notna().sum()
    print(f"Derived {len(derived_df)} rows -> {args.output} "
          f"({n_scored}/{len(derived_df)} scorable, zero new model calls)")


if __name__ == "__main__":
    main()
