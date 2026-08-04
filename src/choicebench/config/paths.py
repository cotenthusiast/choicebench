import os
import shutil
import re
from importlib import resources
from pathlib import Path

# Workspace policy: CHOICEBENCH_HOME is applied before imports;
# imports; otherwise user-generated artifacts live beneath the current working
# directory. Runtime resources are package data and are never written here.
ROOT_DIR = Path(os.environ.get("CHOICEBENCH_HOME", Path.cwd())).expanduser().resolve()

# Directories ----------------------------------------
DATA_DIR = ROOT_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"

RUNS_DIR = ROOT_DIR / "runs"
REPORTS_DIR = ROOT_DIR / "reports"
PROMPTS_DIR = resources.files("choicebench.resources").joinpath("prompts")

def ensure_dirs() -> None:
    """Create all standard project directories. Call once at program startup."""
    for directory in [
        DATA_DIR,
        PROCESSED_DIR,
        RUNS_DIR,
        REPORTS_DIR,
    ]:
        try:
            directory.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise RuntimeError(
                f"ChoiceBench workspace is not writable at {ROOT_DIR}. Set CHOICEBENCH_HOME "
                "to a writable directory before starting the command."
            ) from exc


_RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def validate_run_id(run_id: str) -> str:
    if not isinstance(run_id, str) or not _RUN_ID_RE.fullmatch(run_id):
        raise ValueError(
            "run ID must be 1-128 ASCII letters, digits, dots, underscores, or hyphens, "
            "starting with a letter or digit; path separators and traversal are forbidden."
        )
    return run_id

# TOY benchmark ---------------------------------------
TOY_BENCHMARK_PATH = PROCESSED_DIR / "toy_normalized.csv"


def get_benchmark_path(name: str, split: str = "test") -> Path:
    """Return the legacy generic path; verified runtime loaders do not use it."""
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
    raw_run_dir = Path(run_dir)
    allowed_root = Path(allowed_root).resolve()

    if raw_run_dir.is_symlink():
        raise ValueError(f"Refusing to reset symlinked run directory {raw_run_dir}.")
    run_dir = raw_run_dir.resolve()

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
