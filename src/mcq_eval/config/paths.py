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
RAW_QUESTIONS_FILENAME = "mmlu_raw.csv"
NORMALIZED_QUESTIONS_FILENAME = "mmlu_normalized.csv"
HUMAN_REVIEW_FILENAME = "faithfulness_human_review.csv"

# Full file paths ------------------------------------
RAW_QUESTIONS_PATH = RAW_DIR / RAW_QUESTIONS_FILENAME
NORMALIZED_QUESTIONS_PATH = PROCESSED_DIR / NORMALIZED_QUESTIONS_FILENAME
HUMAN_REVIEW_PATH = REVIEWS_DIR / HUMAN_REVIEW_FILENAME