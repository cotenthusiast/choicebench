import shutil
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

# TOY benchmark ---------------------------------------
TOY_BENCHMARK_PATH = PROCESSED_DIR / "toy_normalized.csv"


def get_benchmark_path(name: str) -> Path:
    """Return the normalized CSV path for a registered benchmark."""
    return PROCESSED_DIR / f"{name}_normalized.csv"


def safe_reset_run_dir(run_dir: Path, allowed_root: Path = RUNS_DIR) -> Path:
    """Clear a run directory, guarded so it can only ever touch runs output.

    Resolves both ``run_dir`` and ``allowed_root`` (following symlinks) *before*
    the containment check, so a symlink pointing outside ``allowed_root`` cannot
    be used to delete an arbitrary path. Deletion is refused unless the resolved
    ``run_dir`` is a strict subdirectory of the resolved ``allowed_root`` — in
    particular, ``allowed_root`` itself is never deleted.

    On success the directory's contents are removed and it is left existing but
    empty. Resetting a directory that does not exist yet is a no-op that still
    yields an empty directory.

    Args:
        run_dir: The run directory to clear.
        allowed_root: The only tree under which deletion is permitted.

    Returns:
        The resolved ``run_dir``.

    Raises:
        ValueError: if ``run_dir`` resolves to ``allowed_root`` itself or to a
            path outside ``allowed_root`` (including via symlink traversal).
    """
    run_dir = Path(run_dir).resolve()
    allowed_root = Path(allowed_root).resolve()

    if run_dir == allowed_root:
        raise ValueError(
            f"Refusing to reset the runs root itself ({run_dir}); "
            f"safe_reset_run_dir only clears a run subdirectory."
        )
    if allowed_root not in run_dir.parents:
        raise ValueError(
            f"Refusing to reset {run_dir}: it does not resolve to a path under "
            f"the allowed root {allowed_root}."
        )

    # rmtree does not follow symlinks, so any symlink *inside* run_dir is removed
    # as a link rather than followed to its target. Recreate the empty directory
    # so callers can rely on it existing afterwards.
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)
    return run_dir
