from pathlib import Path

# Root ------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parents[3]

# Directories ----------------------------------------
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
SPLITS_DIR = DATA_DIR / "splits"

RUNS_DIR = ROOT_DIR / "runs"
REPORTS_DIR = ROOT_DIR / "reports"
LOG_DIR = ROOT_DIR / "logs"
PROMPTS_DIR = ROOT_DIR / "prompts"

def ensure_dirs() -> None:
    """Create all standard project directories. Call once at program startup."""
    for directory in [
        DATA_DIR,
        RAW_DIR,
        PROCESSED_DIR,
        SPLITS_DIR,
        RUNS_DIR,
        REPORTS_DIR,
        LOG_DIR,
    ]:
        directory.mkdir(parents=True, exist_ok=True)

# Filenames ------------------------------------------
MMLU_RAW_FILENAME = "mmlu_raw.csv"
MMLU_NORMALIZED_FILENAME = "mmlu_normalized.csv"

# Full file paths ------------------------------------
MMLU_RAW_PATH = RAW_DIR / MMLU_RAW_FILENAME
MMLU_NORMALIZED_PATH = PROCESSED_DIR / MMLU_NORMALIZED_FILENAME

# ARC-Challenge ---------------------------------------
ARC_NORMALIZED_PATH = PROCESSED_DIR / "arc_challenge_normalized.csv"

# TOY benchmark ---------------------------------------
TOY_BENCHMARK_PATH = PROCESSED_DIR / "toy_normalized.csv"
