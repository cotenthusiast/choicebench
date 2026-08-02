"""Tests for atomic staged publication (Task 11, reduced scope)."""

from __future__ import annotations

import pytest

from choicebench.infra.artifacts import LockHeldError
from choicebench.importing.transaction import (
    ImportTransaction,
    TransactionError,
    atomic_publish_directory_no_replace,
    fsync_tree,
)
import choicebench.importing.transaction as transaction_module


def test_fsync_tree_does_not_raise_on_a_populated_directory(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "file.txt").write_text("hello")
    fsync_tree(tmp_path)  # must not raise


def test_atomic_publish_directory_no_replace_moves_staged_tree(tmp_path):
    staged = tmp_path / "staged"
    staged.mkdir()
    (staged / "file.txt").write_text("content")
    destination = tmp_path / "runs" / "run-1"
    atomic_publish_directory_no_replace(staged, destination)
    assert destination.is_dir()
    assert (destination / "file.txt").read_text() == "content"
    assert not staged.exists()


def test_atomic_publish_directory_no_replace_refuses_existing_destination(tmp_path):
    staged = tmp_path / "staged"
    staged.mkdir()
    destination = tmp_path / "runs" / "run-1"
    destination.mkdir(parents=True)
    (destination / "existing.txt").write_text("already here")
    with pytest.raises(TransactionError):
        atomic_publish_directory_no_replace(staged, destination)
    assert (destination / "existing.txt").read_text() == "already here"  # untouched
    assert staged.exists()  # never moved


def test_renameat2_wrapper_fails_closed_when_unavailable(tmp_path, monkeypatch):
    """If renameat2 cannot be located, this must refuse -- never silently
    fall back to os.replace (which would race an existence check)."""
    monkeypatch.setattr(transaction_module.ctypes.util, "find_library", lambda name: None)
    staged = tmp_path / "staged"
    staged.mkdir()
    destination = tmp_path / "runs" / "run-1"
    with pytest.raises(TransactionError, match="libc"):
        atomic_publish_directory_no_replace(staged, destination)
    assert staged.exists()
    assert not destination.exists()


def test_import_transaction_publishes_a_new_run(tmp_path):
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    validated_paths = []

    def validator(path):
        validated_paths.append(path)
        (path / "manifest.json").write_text("{}")

    with ImportTransaction(runs_dir=runs_dir, run_id="run-1") as txn:
        (txn.staged_run / "evidence.txt").write_text("data")
        final_path = txn.publish(validator)

    assert final_path == runs_dir / "run-1"
    assert (final_path / "evidence.txt").read_text() == "data"
    assert (final_path / "manifest.json").read_text() == "{}"
    # the validator ran against the staging directory, before the rename
    assert len(validated_paths) == 1
    assert validated_paths[0].name.startswith(".staging-run-1")
    # staging directory is gone (renamed into place, not left behind)
    leftover = [p for p in runs_dir.iterdir() if p.name.startswith(".staging-run-1")]
    assert leftover == []


def test_import_transaction_verifies_existing_run_without_staging(tmp_path):
    runs_dir = tmp_path / "runs"
    final_run = runs_dir / "run-1"
    final_run.mkdir(parents=True)
    (final_run / "manifest.json").write_text("{}")

    validated_paths = []

    def validator(path):
        validated_paths.append(path)

    with ImportTransaction(runs_dir=runs_dir, run_id="run-1") as txn:
        with pytest.raises(TransactionError, match="already exists"):
            _ = txn.staged_run
        result = txn.publish(validator)

    assert result == final_run
    assert validated_paths == [final_run]
    leftover = [p for p in runs_dir.iterdir() if p.name.startswith(".staging-run-1")]
    assert leftover == []  # no staging directory was ever created


def test_import_transaction_cleans_up_staging_on_validator_failure(tmp_path):
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()

    def failing_validator(path):
        raise ValueError("synthetic validation failure")

    with pytest.raises(ValueError, match="synthetic validation failure"):
        with ImportTransaction(runs_dir=runs_dir, run_id="run-1") as txn:
            txn.publish(failing_validator)

    assert not (runs_dir / "run-1").exists()
    leftover = [p for p in runs_dir.iterdir() if p.name.startswith(".staging-run-1")]
    assert leftover == []


def test_import_transaction_holds_the_manifest_lock(tmp_path):
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    with ImportTransaction(runs_dir=runs_dir, run_id="run-1"):
        with pytest.raises(LockHeldError):
            with ImportTransaction(runs_dir=runs_dir, run_id="run-1"):
                pass
