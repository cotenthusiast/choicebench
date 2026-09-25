# src/choicebench/infra/resumable_csv.py

"""Shared pre-flight guard for every resumable, incrementally-appended CSV
script in this codebase (the standalone rotation/repeat scripts under
scripts/paper/).

Before any of these scripts appends to or resumes from an existing output
file, it must verify the existing file's schema/identity is actually
compatible with what THIS run is about to write -- otherwise a stale file
(produced by an older script version, or accidentally pointed at the
wrong method/run) gets silently treated as valid completed work, gets
new-format rows appended into the same file, or ends up with rows of
inconsistent column counts that pandas can no longer parse back. Call
this once, before computing the pending-work set, so an incompatibility
is caught immediately rather than corrupting the file mid-run.

Two modes, chosen by which argument the caller supplies:
  - exact_columns: the file's header must match this list exactly, in
    order. For scripts whose every row is schema-uniform (the
    single-stage rotation scripts) -- any difference at all means a
    stale or wrong-version file.
  - required_columns: the file's header must be a SUPERSET of this list
    (order-independent). For scripts whose rows are legitimately
    heterogeneous across observations (e.g. a reused canonical
    observation carrying different provenance columns than a freshly
    computed one) -- only a minimum identity set is enforced.

Either mode can be combined with expected_method_name, which additionally
refuses to resume if the file contains rows for a different method_name
-- the "wrong file entirely" case, distinct from legitimate schema
variation within one correctly-identified file.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def check_resume_compatible(
        output_path: Path,
        *,
        exact_columns: list[str] | None = None,
        required_columns: list[str] | None = None,
        expected_method_name: str | None = None,
        method_name_column: str = "method_name",
) -> None:
    """Raise ValueError if an existing output_path is incompatible with
    the schema/identity this run is about to write. No-op if the file
    doesn't exist yet (a fresh run has nothing to be incompatible with).
    """
    if not output_path.exists():
        return

    existing_header = list(pd.read_csv(output_path, nrows=0).columns)

    if exact_columns is not None and existing_header != exact_columns:
        raise ValueError(
            f"{output_path} has a different schema than this run would write -- "
            f"existing columns {existing_header} != {exact_columns}. "
            "Move or delete the stale file (it was likely produced by a "
            "different script version) before resuming."
        )

    if required_columns is not None:
        missing = [c for c in required_columns if c not in existing_header]
        if missing:
            raise ValueError(
                f"{output_path} has a different schema than this run expects -- "
                f"missing required column(s) {missing} (existing columns: "
                f"{existing_header}). Move or delete the stale file (it was "
                "likely produced by a different script or method) before resuming."
            )

    if expected_method_name is not None and method_name_column in existing_header:
        existing_methods = pd.read_csv(output_path, usecols=[method_name_column])
        mismatched = set(existing_methods[method_name_column].astype(str).unique())
        mismatched.discard(str(expected_method_name))
        if mismatched:
            raise ValueError(
                f"{output_path} contains {method_name_column}={sorted(mismatched)}, "
                f"but this run is producing {method_name_column}={expected_method_name!r}. "
                "This looks like the wrong output file -- refusing to mix runs."
            )
