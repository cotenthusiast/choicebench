# experiments/flip_rate_traces/data_source.py
#
# Question-identity loader. Uses the immutable Stage-1 paper freeze
# (model-generalization/paper_data_freeze) as dataset and split authority,
# per explicit instruction -- this experiment does not go through
# choicebench's own CLI/METHOD_REGISTRY question-source path, for the same
# reason experiments/visible_llm_matcher's run_fourth_cell.py didn't (see
# ChoiceBench v0.2.0 Audit.md Finding 4): the CLI's own re-hashed
# question_id scheme does not agree with this externally-frozen dataset.
#
# Only question-IDENTITY columns are read from the canonical CSVs
# (question_id, subject, question_text, choice_a..d, correct_option). The
# historical `prompt`/`raw_text`/`parsed_choice`/... columns in those files
# belong to the OLD non-trace-retaining run and are intentionally ignored --
# this experiment performs fresh inference and builds its own prompts via
# historical_protocol.py + choicebench's prompt_builder.

from __future__ import annotations

import csv
import hashlib
import os
from dataclasses import dataclass
from pathlib import Path

# Overridable so Kelvin2 (no laptop filesystem) can point at a synced subset
# of the freeze instead of the full model-generalization checkout. Laptop/API
# runs use the real, full paper_data_freeze by default.
FREEZE_ROOT = Path(os.environ.get(
    "FLIP_RATE_TRACES_FREEZE_ROOT",
    "/home/cotenthusiast/Projects/model-generalization/paper_data_freeze",
))
CANONICAL_DIR = FREEZE_ROOT / "canonical"
MANIFEST_CSV = FREEZE_ROOT / "manifests" / "canonical_results_manifest.csv"

# cell_id -> (canonical relative path, expected sha256), read once at import
# time from the manifest so a stale/altered freeze file fails loudly instead
# of silently feeding different questions into a run.
_CANONICAL_CELLS = {
    "cbp__gpt-4-1-mini__arc_challenge__cyclic_generation_majority": (
        "cbp__gpt-4-1-mini__arc_challenge__cyclic_generation_majority/"
        "20260601_183346_cyclic_gpt-4.1-mini_arc_challenge.csv",
        "d7a1d3244294d8a4486581ba377d62187fe7626cf8e000ef0428b69525200ef9",
    ),
    "cbp__gpt-4-1-mini__mmlu__cyclic_generation_majority": (
        "cbp__gpt-4-1-mini__mmlu__cyclic_generation_majority/"
        "20260601_183346_cyclic_gpt-4.1-mini_mmlu.csv",
        "dba83af2b8b25db4b689bac61a43e3068f776c028a3aa8e2c608b16180113708",
    ),
    "cbp__qwen-qwen2-5-7b-instruct__arc_challenge__cyclic_generation_majority": (
        "cbp__qwen-qwen2-5-7b-instruct__arc_challenge__cyclic_generation_majority/"
        "20260529_145812_cyclic_Qwen_Qwen2.5-7B-Instruct_arc_challenge.csv",
        "ad7de543bf8bda698c9442f07619e591c4eca7708623f182561ebf82ce08d883",
    ),
    "cbp__qwen-qwen2-5-7b-instruct__mmlu__cyclic_generation_majority": (
        "cbp__qwen-qwen2-5-7b-instruct__mmlu__cyclic_generation_majority/"
        "20260529_145812_cyclic_Qwen_Qwen2.5-7B-Instruct_mmlu.csv",
        "570016a9c291240c960a4cd2023fa89b662e81e02e8d0fd3804dc1329b9b0843",
    ),
}


@dataclass(frozen=True)
class FrozenQuestion:
    question_id: str
    subject: str
    question_text: str
    choice_a: str
    choice_b: str
    choice_c: str
    choice_d: str
    correct_option: str


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_frozen_questions(cell_id: str) -> list[FrozenQuestion]:
    """Load the frozen 1000-question identity set for one canonical cell.

    Verifies the source file's sha256 against the value recorded in
    paper_data_freeze/manifests/canonical_results_manifest.csv before
    reading it, so silent drift in the frozen dataset raises instead of
    quietly changing what this experiment scores against.
    """
    if cell_id not in _CANONICAL_CELLS:
        raise ValueError(f"Unknown canonical cell_id: {cell_id!r}")
    rel_path, expected_sha = _CANONICAL_CELLS[cell_id]
    path = CANONICAL_DIR / rel_path
    if not path.exists():
        raise FileNotFoundError(f"Canonical frozen source not found: {path}")

    actual_sha = _sha256(path)
    if actual_sha != expected_sha:
        raise ValueError(
            f"{cell_id}: canonical source sha256 mismatch. "
            f"expected={expected_sha} actual={actual_sha} path={path}. "
            "The frozen dataset appears to have changed; refusing to run "
            "against unverified question data."
        )

    questions: list[FrozenQuestion] = []
    seen_ids: set[str] = set()
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            qid = row["question_id"]
            if qid in seen_ids:
                raise ValueError(f"{cell_id}: duplicate question_id {qid!r} in source.")
            seen_ids.add(qid)
            questions.append(FrozenQuestion(
                question_id=qid,
                subject=row["subject"],
                question_text=row["question_text"],
                choice_a=row["choice_a"],
                choice_b=row["choice_b"],
                choice_c=row["choice_c"],
                choice_d=row["choice_d"],
                correct_option=row["correct_option"],
            ))

    if len(questions) != 1000:
        raise ValueError(
            f"{cell_id}: expected exactly 1000 unique questions, found {len(questions)}."
        )
    return questions
