from pathlib import Path

# Root ------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parents[3]

# Directories ----------------------------------------
DATA_DIR = ROOT_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"

RUNS_DIR = ROOT_DIR / "runs"
REPORTS_DIR = ROOT_DIR / "reports"
PROMPTS_DIR = ROOT_DIR / "prompts"

def ensure_dirs() -> None:
    """Create all standard project directories. Call once at program startup."""
    for directory in [
        DATA_DIR,
        PROCESSED_DIR,
        RUNS_DIR,
        REPORTS_DIR,
    ]:
        directory.mkdir(parents=True, exist_ok=True)

# Filenames ------------------------------------------
MMLU_NORMALIZED_FILENAME = "mmlu_normalized.csv"

# Full file paths ------------------------------------
MMLU_NORMALIZED_PATH = PROCESSED_DIR / MMLU_NORMALIZED_FILENAME

# ARC-Challenge ---------------------------------------
ARC_NORMALIZED_PATH = PROCESSED_DIR / "arc_challenge_normalized.csv"

# TOY benchmark ---------------------------------------
TOY_BENCHMARK_PATH = PROCESSED_DIR / "toy_normalized.csv"
