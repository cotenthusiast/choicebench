from pathlib import Path

# Root ------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parents[3]

# Directories ----------------------------------------
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
SPLITS_DIR = DATA_DIR / "splits"
REVIEWS_DIR = DATA_DIR / "reviews"

RUNS_DIR = ROOT_DIR / "runs"
REPORTS_DIR = ROOT_DIR / "reports"
LOG_DIR = ROOT_DIR / "logs"
PROMPTS_DIR = ROOT_DIR / "prompts"

for directory in [
    DATA_DIR,
    RAW_DIR,
    PROCESSED_DIR,
    SPLITS_DIR,
    REVIEWS_DIR,
    RUNS_DIR,
    REPORTS_DIR,
    LOG_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)

# Filenames ------------------------------------------
MMLU_RAW_FILENAME = "mmlu_raw.csv"
MMLU_NORMALIZED_FILENAME = "mmlu_normalized.csv"
HUMAN_REVIEW_FILENAME = "faithfulness_human_review.csv"

# Full file paths ------------------------------------
MMLU_RAW_PATH = RAW_DIR / MMLU_RAW_FILENAME
MMLU_NORMALIZED_PATH = PROCESSED_DIR / MMLU_NORMALIZED_FILENAME
MMLU_HUMAN_REVIEW_PATH = REVIEWS_DIR / HUMAN_REVIEW_FILENAME

# ARC-Challenge ---------------------------------------
ARC_RAW_PATH = RAW_DIR / "arc_challenge_raw.csv"
ARC_NORMALIZED_PATH = PROCESSED_DIR / "arc_challenge_normalized.csv"
ARC_SPLITS_DIR = SPLITS_DIR / "arc_challenge"
ARC_SPLIT_NAME = "robustness"
ARC_SAMPLE_SIZE = 1000
ARC_SAMPLE_SEED = 42