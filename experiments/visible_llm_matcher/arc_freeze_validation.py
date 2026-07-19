# experiments/visible_llm_matcher/arc_freeze_validation.py
#
# CORRECTION: an earlier version of this experiment compared reused ARC
# Stage-1 rows against ChoiceBench's own current, independently
# re-normalized data/processed/arc_challenge_normalized.csv and found 6
# apparent question_id mismatches. That comparison used the wrong
# authority. For this paper experiment, all historical cells (two_prompt,
# text_extraction, and now this fourth cell) were and are evaluated against
# the IMMUTABLE STAGE 1 FREEZE in model-generalization's
# paper_data_freeze/raw/local_model_generalization/, not against
# ChoiceBench's current repository state. Re-running the comparison against
# the correct authority (see below) found ZERO mismatches across all 1000
# ARC-Challenge robustness-split questions, for all 6 reused Stage-1
# sources (4 API models + 2 local models) — TSP's own question_id IS the
# freeze's question_id, verified by direct content comparison, not assumed.
#
# Freeze inputs (paths are relative to freeze_root, passed in by the
# caller — never hardcoded to one machine layout beyond the freeze_root
# itself):
#   data/processed/arc_challenge_normalized.csv   (1172 rows; the 1000-row
#     robustness split is a subset, selected by robustness_ids.json)
#   data/splits/arc_challenge/robustness_ids.json (the exact 1000 question_ids)
#
# Both are checksum-verified against paper_data_freeze/checksums/checksums.sha256
# before use (FREEZE_ARC_NORMALIZED_SHA256 / FREEZE_ROBUSTNESS_IDS_SHA256
# below) — verified once by hand (sha256sum) against that file when this
# module was written; validate_against_freeze() re-verifies on every call
# so a changed freeze file is caught immediately rather than silently used.
#
# Known dataset-normalization limitation (provenance note, not a bug to fix
# in this branch): 3 of the 1000 frozen ARC questions
# (KNOWN_UPSTREAM_MISSING_OPTION_E_IDS below) have a real 5th option in the
# upstream allenai/ai2_arc source that the historical normalizer which
# produced this frozen dataset never extracted — it only ever reads labels
# A-D. This affects every historical ARC cell built from this same frozen
# dataset (two_prompt, text_extraction, and now this fourth cell alike), so
# preserving it here keeps the fourth cell comparable to the other three;
# "fixing" it only for this cell would introduce an uncontrolled second
# difference into what is supposed to be a single-axis (matcher-only) 2x2
# comparison. A true A-E correction, if ever undertaken, requires a new
# dataset version and consistent re-evaluation of the entire historical ARC
# matrix — explicitly out of scope here. See KNOWN_3OPTION_ARC_QUESTION_IDS
# in stage1_sources.py for the unrelated (and, unlike this one, in-scope)
# 3-option repair.

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

FREEZE_ARC_NORMALIZED_RELATIVE_PATH = Path("data/processed/arc_challenge_normalized.csv")
FREEZE_ROBUSTNESS_IDS_RELATIVE_PATH = Path("data/splits/arc_challenge/robustness_ids.json")

# sha256sum of each file under
# model-generalization/paper_data_freeze/raw/local_model_generalization/,
# cross-checked against paper_data_freeze/checksums/checksums.sha256 on
# 2026-07-19. Both matched exactly.
FREEZE_ARC_NORMALIZED_SHA256 = "beeef9ea267226dcd083373f552ffe17434f3d2738fcc71f85ffe29bd7d96a8e"
FREEZE_ROBUSTNESS_IDS_SHA256 = "cfe441d825eaced8c64d710c6d4d5a3d54b173c3a385e445bd581390c6406ee5"

EXPECTED_ROBUSTNESS_ROW_COUNT = 1000

_COMPARISON_COLUMNS = (
    "question_text",
    "choice_a",
    "choice_b",
    "choice_c",
    "choice_d",
    "correct_option",
)

# The 3 questions where the historical normalizer dropped a real 5th
# (E) option present in the upstream HuggingFace source. Frozen A-D
# representation is preserved as-is — see module docstring.
KNOWN_UPSTREAM_MISSING_OPTION_E_IDS: frozenset[str] = frozenset(
    {
        "f87cb129d9aa26c0",
        "e970a6b50d905595",
        "8aec8773c1d6b508",
    }
)


class FreezeValidationError(RuntimeError):
    """Raised when reused ARC-Challenge Stage-1 rows disagree with the
    immutable Stage 1 freeze, or when the freeze files themselves don't
    match their recorded checksum."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _normalize_cell(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and value != value:  # NaN
        return ""
    return str(value).strip()


def load_frozen_arc_dataset(freeze_root: Path) -> pd.DataFrame:
    """Load and checksum-verify the frozen ARC-Challenge normalized dataset.

    Args:
        freeze_root: model-generalization/paper_data_freeze/raw/local_model_generalization

    Returns:
        DataFrame indexed by question_id (question_id also kept as a
        column), 1172 rows — the full normalized pool, not yet filtered to
        the 1000-question robustness split.

    Raises:
        FreezeValidationError: file missing, or its sha256 does not match
            FREEZE_ARC_NORMALIZED_SHA256.
    """
    path = freeze_root / FREEZE_ARC_NORMALIZED_RELATIVE_PATH
    if not path.is_file():
        raise FreezeValidationError(f"Frozen ARC dataset not found: {path}")
    digest = _sha256(path)
    if digest != FREEZE_ARC_NORMALIZED_SHA256:
        raise FreezeValidationError(
            f"{path} sha256={digest} does not match the recorded immutable "
            f"freeze checksum {FREEZE_ARC_NORMALIZED_SHA256}. Refusing to "
            "validate against a file that may have changed since the "
            "freeze was taken."
        )
    return pd.read_csv(path).set_index("question_id", drop=False)


def load_frozen_robustness_ids(freeze_root: Path) -> list[str]:
    """Load and checksum-verify the frozen 1000-question robustness split.

    Raises:
        FreezeValidationError: file missing, or its sha256 does not match
            FREEZE_ROBUSTNESS_IDS_SHA256.
    """
    path = freeze_root / FREEZE_ROBUSTNESS_IDS_RELATIVE_PATH
    if not path.is_file():
        raise FreezeValidationError(f"Frozen robustness_ids.json not found: {path}")
    digest = _sha256(path)
    if digest != FREEZE_ROBUSTNESS_IDS_SHA256:
        raise FreezeValidationError(
            f"{path} sha256={digest} does not match the recorded immutable "
            f"freeze checksum {FREEZE_ROBUSTNESS_IDS_SHA256}. Refusing to "
            "validate against a file that may have changed since the "
            "freeze was taken."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def validate_against_freeze(reused_rows: pd.DataFrame, freeze_root: Path) -> None:
    """Validate reused ARC-Challenge Stage-1 rows against the immutable
    Stage 1 freeze — identifiers AND content, not identifiers alone.

    This is the check that makes it impossible for a current-ChoiceBench-
    normalizer question_id to silently stand in for a freeze question_id:
    such an id would not appear in the frozen robustness_ids split (fails
    the first check below) even before content is compared.

    Args:
        reused_rows: DataFrame with at least question_id, question_text,
            choice_a..choice_d, correct_option — e.g. the output of
            stage1_sources.load_and_validate_stage1() for one (model, "arc_challenge")
            pair.
        freeze_root: model-generalization/paper_data_freeze/raw/local_model_generalization

    Raises:
        FreezeValidationError:
            - reused_rows does not have exactly EXPECTED_ROBUSTNESS_ROW_COUNT rows;
            - any question_id is not in the frozen robustness_ids split
              (catches an id from a different normalizer);
            - any question_id is absent from the frozen normalized dataset;
            - question_text/choice_a..choice_d/correct_option differs from
              the frozen row for that question_id;
            - the frozen robustness_ids split is not fully covered by
              reused_rows (a silently dropped question).
    """
    if len(reused_rows) != EXPECTED_ROBUSTNESS_ROW_COUNT:
        raise FreezeValidationError(
            f"Expected exactly {EXPECTED_ROBUSTNESS_ROW_COUNT} reused "
            f"ARC-Challenge rows, got {len(reused_rows)}."
        )

    frozen_df = load_frozen_arc_dataset(freeze_root)
    robustness_ids = set(load_frozen_robustness_ids(freeze_root))

    seen_ids: set[str] = set()
    for _, row in reused_rows.iterrows():
        qid = row["question_id"]
        if qid in seen_ids:
            raise FreezeValidationError(f"Duplicate question_id in reused_rows: {qid!r}.")
        seen_ids.add(qid)

        if qid not in robustness_ids:
            raise FreezeValidationError(
                f"question_id={qid!r} is not in the frozen robustness_ids "
                "split. A current-ChoiceBench-normalizer id (or any id not "
                "sourced from the freeze) must not be substituted for a "
                "freeze id."
            )
        if qid not in frozen_df.index:
            raise FreezeValidationError(
                f"question_id={qid!r} is in robustness_ids.json but absent "
                "from the frozen normalized dataset — investigate before "
                "proceeding."
            )

        frozen_row = frozen_df.loc[qid]
        for col in _COMPARISON_COLUMNS:
            reused_val = _normalize_cell(row[col])
            frozen_val = _normalize_cell(frozen_row[col])
            if reused_val != frozen_val:
                raise FreezeValidationError(
                    f"question_id={qid!r} column {col!r} disagrees with the "
                    f"frozen dataset: reused={reused_val!r} freeze={frozen_val!r}."
                )

    missing_ids = robustness_ids - seen_ids
    if missing_ids:
        raise FreezeValidationError(
            f"{len(missing_ids)} frozen robustness_ids are absent from "
            f"reused_rows, e.g. {sorted(missing_ids)[:5]}."
        )
