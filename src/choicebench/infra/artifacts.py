"""Atomic filesystem writes and advisory process locks."""

from __future__ import annotations

import json
import os
import socket
import uuid
from pathlib import Path
from typing import Any


def atomic_write_bytes(path: Path, data: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    try:
        with tmp.open("xb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def atomic_write_text(path: Path, text: str) -> None:
    atomic_write_bytes(path, text.encode("utf-8"))


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_text(path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


class LockHeldError(RuntimeError):
    """Raised when another live process holds the requested advisory lock."""


class FileLock:
    """Non-blocking POSIX advisory lock with actionable owner metadata."""

    def __init__(self, path: Path, purpose: str) -> None:
        self.path = Path(path)
        self.purpose = purpose
        self._handle = None

    def __enter__(self) -> "FileLock":
        try:
            import fcntl
        except ImportError as exc:  # pragma: no cover - ChoiceBench targets POSIX research systems
            raise RuntimeError("ChoiceBench process locking requires a POSIX platform.") from exc
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(self._handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self._handle.seek(0)
            owner = self._handle.read().strip() or "unknown owner"
            self._handle.close()
            self._handle = None
            raise LockHeldError(f"Cannot {self.purpose}: lock {self.path} is held by {owner}.") from exc
        self._handle.seek(0)
        self._handle.truncate()
        self._handle.write(f"pid={os.getpid()} host={socket.gethostname()} purpose={self.purpose}\n")
        self._handle.flush()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        if self._handle is None:
            return
        import fcntl
        fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
        self._handle.close()
        self._handle = None
