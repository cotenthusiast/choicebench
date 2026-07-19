"""Atomic, no-replace staged publication for imported runs.

Reduced scope: covers the concrete safety guarantees the repair-import
workflow needs -- a manifest lock, sibling staging on the same filesystem,
full-tree fsync before publication, and a genuine no-replace rename (never
an existence-check-then-os.replace race) -- without the fuller historical
crash-recovery/cleanup-scope richness of the original plan.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import os
from pathlib import Path
from typing import Callable
import shutil
import uuid

from choicebench.infra.artifacts import FileLock

_RENAME_NOREPLACE = 0x1
_AT_FDCWD = -100


class TransactionError(ValueError):
    """Raised when staged publication cannot proceed safely."""


def fsync_tree(root: Path) -> None:
    """Best-effort fsync of every file and directory under root, so a crash
    immediately after this call cannot lose or half-write staged content."""
    root = Path(root)
    for dirpath, dirnames, filenames in os.walk(root):
        for name in filenames:
            path = Path(dirpath) / name
            try:
                fd = os.open(path, os.O_RDONLY)
            except OSError:
                continue
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
        try:
            dir_fd = os.open(dirpath, os.O_RDONLY)
        except OSError:
            continue
        try:
            os.fsync(dir_fd)
        except OSError:
            pass
        finally:
            os.close(dir_fd)


def _renameat2_no_replace(old: Path, new: Path) -> None:
    """Thin wrapper around the libc renameat2(2) syscall with
    RENAME_NOREPLACE, isolated so tests can mock its absence without
    depending on kernel version. Never falls back to os.replace or an
    existence-check race: if the primitive is unavailable, this fails
    closed."""
    library_name = ctypes.util.find_library("c")
    if not library_name:
        raise TransactionError("libc is not available; refusing an unsafe rename fallback.")
    libc = ctypes.CDLL(library_name, use_errno=True)
    if not hasattr(libc, "renameat2"):
        raise TransactionError(
            "renameat2 is not available on this system; refusing an unsafe rename fallback."
        )
    result = libc.renameat2(
        ctypes.c_int(_AT_FDCWD),
        os.fsencode(str(old)),
        ctypes.c_int(_AT_FDCWD),
        os.fsencode(str(new)),
        ctypes.c_uint(_RENAME_NOREPLACE),
    )
    if result != 0:
        errno = ctypes.get_errno()
        raise TransactionError(
            f"renameat2 failed renaming {old} -> {new}: errno {errno} ({os.strerror(errno)})."
        )


def atomic_publish_directory_no_replace(source: Path, destination: Path) -> None:
    """Atomically move a fully-staged directory into its final location.
    Refuses (never silently overwrites or races an existence check) if the
    destination already exists."""
    source = Path(source)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    _renameat2_no_replace(source, destination)


class ImportTransaction:
    """Stage a new run under a same-filesystem sibling directory, run a
    caller-supplied full-graph validator, and either publish it atomically
    (new run) or verify an already-published run in place (existing run) --
    never partially write the final run directory."""

    def __init__(self, *, runs_dir: Path, run_id: str) -> None:
        self.runs_dir = Path(runs_dir)
        self.run_id = run_id
        self.final_run = self.runs_dir / run_id
        self._staged_run: Path | None = None
        self._lock: FileLock | None = None
        self._published = False

    def __enter__(self) -> "ImportTransaction":
        lock_path = self.runs_dir / ".locks" / f"{self.run_id}.manifest.lock"
        self._lock = FileLock(lock_path, f"import run {self.run_id}")
        self._lock.__enter__()
        if not self.final_run.exists():
            staged = self.runs_dir / f".staging-{self.run_id}-{uuid.uuid4().hex}"
            staged.mkdir(parents=True)
            (staged / ".staging-owner").write_text(f"pid={os.getpid()}\n")
            self._staged_run = staged
        return self

    @property
    def staged_run(self) -> Path:
        if self._staged_run is None:
            raise TransactionError(
                f"Run {self.run_id!r} already exists; there is no staging directory to "
                "write into. Only verify_import_run-style validation is available."
            )
        return self._staged_run

    def publish(self, validator: Callable[[Path], None]) -> Path:
        if self.final_run.exists():
            validator(self.final_run)
            return self.final_run
        staged = self.staged_run
        if not (staged / ".staging-owner").is_file():
            raise TransactionError(
                f"Refusing to publish {staged}: it is not marked as this transaction's "
                "own staging directory."
            )
        validator(staged)
        fsync_tree(staged)
        (staged / ".staging-owner").unlink()
        atomic_publish_directory_no_replace(staged, self.final_run)
        self._published = True
        self._staged_run = None
        return self.final_run

    def __exit__(self, exc_type, exc, traceback) -> None:
        try:
            if self._staged_run is not None and self._staged_run.exists():
                # Publication never happened (error, or caller chose not to
                # publish); clean up only this transaction's own
                # owner-marked staging directory.
                if (self._staged_run / ".staging-owner").is_file():
                    shutil.rmtree(self._staged_run, ignore_errors=True)
        finally:
            if self._lock is not None:
                self._lock.__exit__(exc_type, exc, traceback)
                self._lock = None
