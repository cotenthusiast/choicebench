"""Checkpoint management for resumable experiment runs."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from choicebench.identity import integrity_digest
from choicebench.infra.atomic_io import atomic_write_json

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
        condition_id: str | None = None,
        experiment_id: str | None = None,
        selection_id: str | None = None,
    ) -> None:
        self._condition_id = condition_id
        self._experiment_id = experiment_id
        self._selection_id = selection_id
        safe_model = model.replace("/", "_")
        self._path = (
            checkpoint_dir / f"{condition_id}.json" if condition_id
            else checkpoint_dir / run_id / f"{condition}__{safe_model}__{benchmark}.json"
        )
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
            if self._condition_id:
                digest = state.get("checkpoint_digest")
                payload = {k: v for k, v in state.items() if k != "checkpoint_digest"}
                if state.get("schema_version") != "choicebench.checkpoint.v1" or digest != integrity_digest(payload):
                    raise RuntimeError(f"Checkpoint integrity check failed: {self._path}")
                expected = (self._experiment_id, self._condition_id, self._selection_id)
                actual = (state.get("experiment_id"), state.get("condition_id"), state.get("selection_id"))
                if actual != expected:
                    raise RuntimeError(
                        f"Checkpoint {self._path} belongs to {actual}, expected {expected}."
                    )
            return state
        except (json.JSONDecodeError, OSError) as exc:
            if self._condition_id:
                raise RuntimeError(f"Verified checkpoint is unreadable: {self._path}: {exc}") from exc
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
            "schema_version": "choicebench.checkpoint.v1" if self._condition_id else None,
            "experiment_id": self._experiment_id,
            "condition_id": self._condition_id,
            "selection_id": self._selection_id,
            "completed_ids": completed_ids,
            "results": results,
            "started_at": started_at,
            "last_checkpoint_at": datetime.now(timezone.utc).isoformat(),
        }
        if self._condition_id:
            state["checkpoint_digest"] = integrity_digest(state)
        try:
            atomic_write_json(self._path, state)
        except OSError as exc:
            raise RuntimeError(f"Failed to write checkpoint {self._path}: {exc}") from exc

    def delete(self) -> None:
        """Remove the checkpoint file after a job completes successfully."""
        try:
            self._path.unlink(missing_ok=True)
        except OSError as exc:
            logger.warning("Failed to delete checkpoint %s: %s", self._path, exc)
