# scripts/paper/derive_semantic_matching.py
#
# Paper-specific (eacl-2026-revision): semantic_matching_v1 is defined as a
# zero-inference derivation from a saved two_stage_v1 run's Stage-1
# free-text output (see choicebench.analysis.derive_from_free_text) --
# this script is the paper-side wiring that reads one, derives the other,
# and writes it out in the normal result-row CSV schema so downstream
# metrics tooling doesn't need to know it wasn't a live run.
#
# Run with:
#   python scripts/paper/derive_semantic_matching.py \
#       --source runs/<run_id>/<..._two_stage_v1_..._<benchmark>.csv> \
#       --output runs/<run_id>/<..._semantic_matching_v1_..._<benchmark>.csv>

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from choicebench.analysis.derive_from_free_text import derive_matched_results

METHOD_NAME = "semantic_matching_v1"


def derive(source_path: Path, output_path: Path) -> pd.DataFrame:
    source_df = pd.read_csv(source_path)
    derived_df = derive_matched_results(source_df, method_name=METHOD_NAME)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    derived_df.to_csv(output_path, index=False)
    return derived_df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path,
                         help="Path to a saved two_stage_v1 result CSV.")
    parser.add_argument("--output", required=True, type=Path,
                         help="Path to write the derived semantic_matching_v1 result CSV.")
    args = parser.parse_args()

    derived_df = derive(args.source, args.output)
    n_scored = derived_df["parsed_choice"].notna().sum()
    print(f"Derived {len(derived_df)} rows -> {args.output} "
          f"({n_scored}/{len(derived_df)} scorable, zero new model calls)")


if __name__ == "__main__":
    main()
