"""Checkpoint management for resumable experiment runs."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


class CheckpointManager:
    """Manages per-job checkpoint files for resumable experiment runs.

    One checkpoint file per (run_id, condition, model, benchmark) job,
    stored as JSON under checkpoint_dir/run_id/.

    The file is written atomically (write to .tmp, then rename) so a crash
    during the write never leaves a corrupt checkpoint.
    """

    def __init__(
        self,
        checkpoint_dir: Path,
        run_id: str,
        condition: str,
        model: str,
        benchmark: str,
    ) -> None:
        safe_model = model.replace("/", "_")
        self._path = checkpoint_dir / run_id / f"{condition}__{safe_model}__{benchmark}.json"
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def load(self) -> dict | None:
        """Load checkpoint state if it exists.

        Returns:
            Dict with keys ``completed_ids``, ``results``, ``started_at``,
            ``last_checkpoint_at``, or None if no checkpoint exists.
        """
        if not self._path.exists():
            return None

        try:
            with self._path.open() as f:
                state = json.load(f)
            logger.info(
                "Loaded checkpoint from %s (%d completed)",
                self._path,
                len(state.get("completed_ids", [])),
            )
            return state
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning(
                "Failed to load checkpoint %s: %s — starting fresh",
                self._path,
                exc,
            )
            return None

    def save(
        self,
        completed_ids: list[str],
        results: list[dict],
        started_at: str,
    ) -> None:
        """Write current progress to the checkpoint file.

        Args:
            completed_ids: All question IDs completed so far in this job.
            results: All result rows accumulated so far in this job.
            started_at: ISO-format UTC timestamp when this job started.
        """
        state = {
            "completed_ids": completed_ids,
            "results": results,
            "started_at": started_at,
            "last_checkpoint_at": datetime.now(timezone.utc).isoformat(),
        }
        tmp = self._path.with_suffix(".tmp")
        try:
            with tmp.open("w") as f:
                json.dump(state, f)
            tmp.replace(self._path)
        except OSError as exc:
            logger.error("Failed to write checkpoint %s: %s", self._path, exc)

    def delete(self) -> None:
        """Remove the checkpoint file after a job completes successfully."""
        try:
            self._path.unlink(missing_ok=True)
        except OSError as exc:
            logger.warning("Failed to delete checkpoint %s: %s", self._path, exc)
