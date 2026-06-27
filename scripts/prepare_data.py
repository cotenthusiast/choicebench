"""Download MMLU and ARC-Challenge, normalize them, and generate experiment splits."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from mcq_eval.benchmarks.split import build_all_splits, ROBUSTNESS_TRACK_NAME, REVIEW_TRACK_NAME
from mcq_eval.config.paths import (
    ARC_NORMALIZED_PATH,
    ARC_RAW_PATH,
    ARC_SAMPLE_SEED,
    ARC_SAMPLE_SIZE,
    ARC_SPLIT_NAME,
    ARC_SPLITS_DIR,
    MMLU_NORMALIZED_FILENAME,
    MMLU_NORMALIZED_PATH,
    MMLU_RAW_PATH,
    PROCESSED_DIR,
    SPLITS_DIR,
)
from mcq_eval.io.readers import read_normalized_questions
from mcq_eval.io.writers import (
    write_group_splits,
    write_normalized_questions,
    write_raw_questions,
)

# ---------------------------------------------------------------------------
# ARC helpers
# ---------------------------------------------------------------------------

def _download_arc_raw(output_path: Path) -> None:
    import json
    from datasets import load_dataset
    dataset = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
    df = dataset.to_pandas()
    # Serialize choices as JSON so it round-trips cleanly through CSV.
    df["choices"] = df["choices"].apply(
        lambda c: json.dumps({"text": list(c["text"]), "label": list(c["label"])})
    )
    df.to_csv(output_path, index=False)


def _normalize_arc(raw_path: Path, normalized_path: Path) -> None:
    import json
    from mcq_eval.benchmarks.arc import build_normalized_dataframe
    df_raw = pd.read_csv(raw_path)
    df_raw["choices"] = df_raw["choices"].apply(json.loads)
    df_normalized = build_normalized_dataframe(df_raw)
    df_normalized.to_csv(normalized_path, index=False)


def _build_arc_split(normalized_path: Path, splits_dir: Path) -> None:
    """Sample ARC_SAMPLE_SIZE questions and write robustness split artifacts."""
    df = pd.read_csv(normalized_path).drop_duplicates(subset="question_id")
    n_available = len(df)

    sample_size = min(ARC_SAMPLE_SIZE, n_available)
    rng = np.random.default_rng(ARC_SAMPLE_SEED)
    df_sorted = df.sort_values("question_id")
    sampled_ids = df_sorted.sample(
        n=sample_size,
        random_state=rng,
        replace=False,
    )["question_id"].tolist()

    splits_dir.mkdir(parents=True, exist_ok=True)

    ids_path = splits_dir / f"{ARC_SPLIT_NAME}_ids.json"
    with ids_path.open("w", encoding="utf-8") as f:
        json.dump(sampled_ids, f, indent=2)

    metadata = {
        "split_name": ARC_SPLIT_NAME,
        "benchmark": "arc_challenge",
        "subjects": ["arc_challenge"],
        "seed": ARC_SAMPLE_SEED,
        "strategy": "random_sample",
        "requested_size": ARC_SAMPLE_SIZE,
        "actual_size": len(sampled_ids),
        "available_pool_size": n_available,
    }
    meta_path = splits_dir / f"{ARC_SPLIT_NAME}_metadata.json"
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """Download MMLU and ARC-Challenge, produce normalized CSVs, and generate splits."""

    # ── MMLU ──────────────────────────────────────────────────────────────
    if MMLU_RAW_PATH.exists():
        print(f"[skip] MMLU raw questions already exist at {MMLU_RAW_PATH}")
    else:
        print("[1/8] Downloading MMLU test split from HuggingFace...")
        write_raw_questions(MMLU_RAW_PATH)
        print(f"[done] Saved raw MMLU questions to {MMLU_RAW_PATH}")

    if MMLU_NORMALIZED_PATH.exists():
        print(f"[skip] MMLU normalized questions already exist at {MMLU_NORMALIZED_PATH}")
    else:
        print("[2/8] Normalizing MMLU raw questions...")
        write_normalized_questions(MMLU_RAW_PATH, MMLU_NORMALIZED_PATH)
        print(f"[done] Saved normalized MMLU questions to {MMLU_NORMALIZED_PATH}")

    print("[3/8] Reading MMLU normalized questions and deduplicating...")
    df_mmlu = read_normalized_questions(MMLU_NORMALIZED_FILENAME, PROCESSED_DIR)
    original_count = len(df_mmlu)
    df_mmlu = df_mmlu.drop_duplicates(subset="question_id")
    deduped_count = len(df_mmlu)
    print(
        f"[done] {original_count} rows -> {deduped_count} unique questions "
        f"({original_count - deduped_count} duplicates removed)"
    )

    print("[4/8] Building and writing MMLU experiment splits...")
    all_splits = build_all_splits(df_mmlu)

    write_group_splits(
        {ROBUSTNESS_TRACK_NAME: all_splits[ROBUSTNESS_TRACK_NAME]},
        SPLITS_DIR,
        "benchmark",
    )
    write_group_splits(
        {REVIEW_TRACK_NAME: all_splits[REVIEW_TRACK_NAME]},
        SPLITS_DIR,
        "faithfulness",
    )
    write_group_splits(
        {REVIEW_TRACK_NAME: all_splits[REVIEW_TRACK_NAME]},
        SPLITS_DIR,
        "stronger_model",
    )

    robustness_count = len(all_splits[ROBUSTNESS_TRACK_NAME]["ids"])
    review_count = len(all_splits[REVIEW_TRACK_NAME]["ids"])
    print(f"[done] MMLU robustness split: {robustness_count} questions")
    print(
        f"[done] MMLU review split: {review_count} questions "
        "(written to faithfulness + stronger_model)"
    )

    # ── ARC-Challenge ──────────────────────────────────────────────────────
    if ARC_RAW_PATH.exists():
        print(f"[skip] ARC-Challenge raw data already exists at {ARC_RAW_PATH}")
    else:
        print("[5/8] Downloading ARC-Challenge test split from HuggingFace...")
        _download_arc_raw(ARC_RAW_PATH)
        print(f"[done] Saved raw ARC-Challenge data to {ARC_RAW_PATH}")

    if ARC_NORMALIZED_PATH.exists():
        print(f"[skip] ARC-Challenge normalized data already exists at {ARC_NORMALIZED_PATH}")
    else:
        print("[6/8] Normalizing ARC-Challenge raw data...")
        _normalize_arc(ARC_RAW_PATH, ARC_NORMALIZED_PATH)
        print(f"[done] Saved normalized ARC-Challenge data to {ARC_NORMALIZED_PATH}")

    arc_split_ids_path = ARC_SPLITS_DIR / f"{ARC_SPLIT_NAME}_ids.json"
    if arc_split_ids_path.exists():
        print(f"[skip] ARC-Challenge split already exists at {ARC_SPLITS_DIR}")
    else:
        print("[7/8] Building ARC-Challenge robustness split...")
        _build_arc_split(ARC_NORMALIZED_PATH, ARC_SPLITS_DIR)
        with arc_split_ids_path.open() as f:
            arc_ids = json.load(f)
        print(f"[done] ARC-Challenge robustness split: {len(arc_ids)} questions")

    print("[8/8] Verifying output files...")
    artifacts = [
        MMLU_RAW_PATH,
        MMLU_NORMALIZED_PATH,
        SPLITS_DIR / "benchmark" / "robustness_ids.json",
        ARC_RAW_PATH,
        ARC_NORMALIZED_PATH,
        ARC_SPLITS_DIR / f"{ARC_SPLIT_NAME}_ids.json",
    ]
    all_ok = True
    for p in artifacts:
        if p.exists():
            print(f"  OK  {p}")
        else:
            print(f"  MISSING  {p}")
            all_ok = False

    if all_ok:
        print("[complete] Data preparation finished successfully.")
    else:
        print("[complete] Data preparation finished with missing artifacts (see above).")


if __name__ == "__main__":
    main()
