# scripts/paper/derive_text_extraction_corrected.py
#
# Paper-specific (eacl-2026-revision): writes a CORRECTED copy of a saved text_extraction result file, re-parsed
# with the intended matching cascade (see choicebench.analysis.derive_text_extraction). Zero model calls. The
# source file is only ever read; the output is a new file carrying the source path and SHA-256.
#
# Run with:
#   python scripts/paper/derive_text_extraction_corrected.py \
#       --source runs/<run_id>/results/cond_<id>.csv --output runs/<derived_dir>/<bench>_text_extraction_corrected_<model>.csv

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pandas as pd

from choicebench.analysis.derive_text_extraction import reparse_text_extraction


def derive(source: Path, output: Path, *, embed_fn=None) -> pd.DataFrame:
    source, output = Path(source), Path(output)
    if output.resolve() == source.resolve():
        raise ValueError("output must not be the source file")
    if output.exists():
        raise FileExistsError(f"{output} already exists -- refusing to overwrite")
    sha = hashlib.sha256(source.read_bytes()).hexdigest()
    derived = reparse_text_extraction(pd.read_csv(source), embed_fn=embed_fn)
    derived["derived_from_file"] = str(source)
    derived["derived_from_sha256"] = sha
    output.parent.mkdir(parents=True, exist_ok=True)
    derived.to_csv(output, index=False)
    return derived


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    derived = derive(args.source, args.output)
    changed = int((derived["parsed_choice"].fillna("") != derived["original_parsed_choice"].fillna("")).sum())
    print(f"{args.output}: {len(derived)} rows, {changed} parsed answers changed vs the original, zero model calls")


if __name__ == "__main__":
    main()
