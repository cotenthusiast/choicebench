"""Advisory process locks."""

from __future__ import annotations

import os
import socket
from pathlib import Path


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
