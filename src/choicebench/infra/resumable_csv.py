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

Also provides detect_conflicting_ids()/load_completed_ids_excluding_
conflicts(): a resumable script's own "which ids are already done" check
must never silently collapse two DISAGREEING rows for the same id into a
single "completed" verdict (a corrupted/concatenated output file
symptom) -- see each function's own docstring.
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


def detect_conflicting_ids(output_path: Path, *, id_column: str = "question_id") -> set[str]:
    """Return the set of id_column values that have more than one row in
    output_path where those rows DON'T all agree (some other field
    differs) -- a corrupted/concatenated output file symptom, distinct
    from a harmless exact-duplicate double-write (e.g. two byte-identical
    rows from a retried write), which is not flagged.

    No-op (empty set) if the file doesn't exist or has no id_column.
    """
    if not output_path.exists():
        return set()
    existing = pd.read_csv(output_path)
    if id_column not in existing.columns:
        return set()
    existing = existing.copy()
    existing[id_column] = existing[id_column].astype(str)
    conflicting: set[str] = set()
    for qid, group in existing.groupby(id_column):
        if len(group) < 2:
            continue
        distinct_rows = group.drop(columns=[id_column]).drop_duplicates()
        if len(distinct_rows) > 1:
            conflicting.add(qid)
    return conflicting


def load_completed_ids_excluding_conflicts(
        output_path: Path, *, id_column: str = "question_id",
) -> set[str]:
    """Safe (never-raises) resume-completeness check: an id_column value
    with MULTIPLE rows that don't all agree is excluded from the returned
    "completed" set entirely, rather than silently collapsed into a single
    verdict for whichever row happened to load first (pandas' own
    behavior when treating a DataFrame column as a set).

    This function's own contract is "what does the file say is safely
    completed" -- it never raises. A caller that wants to fail loudly on
    the same condition (recommended for any resumable run() before it
    starts skipping "completed" work) should separately call
    detect_conflicting_ids() and raise on a non-empty result.
    """
    if not output_path.exists():
        return set()
    existing = pd.read_csv(output_path)
    if id_column not in existing.columns:
        return set()
    ids = set(existing[id_column].astype(str))
    return ids - detect_conflicting_ids(output_path, id_column=id_column)
