# tests/infra/test_checkpoint.py

import json
import pytest
from pathlib import Path

from choicebench.infra.checkpoint import CheckpointManager


@pytest.fixture
def manager(tmp_path: Path) -> CheckpointManager:
    return CheckpointManager(
        checkpoint_dir=tmp_path / "checkpoints",
        condition_id="cond_1",
        experiment_id="exp_1",
        selection_id="sel_1",
    )


class TestCheckpointManagerLoad:
    def test_returns_none_when_no_file_exists(self, manager):
        assert manager.load() is None

    def test_raises_for_corrupt_json(self, tmp_path):
        mgr = CheckpointManager(
            checkpoint_dir=tmp_path / "checkpoints",
            condition_id="cond_1",
            experiment_id="exp_1",
            selection_id="sel_1",
        )
        path = tmp_path / "checkpoints" / "cond_1.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{corrupted json ][")
        with pytest.raises(RuntimeError, match="unreadable"):
            mgr.load()

    def test_returns_state_dict_after_save(self, manager):
        manager.save(["q1", "q2"], [{"question_id": "q1"}, {"question_id": "q2"}], "2026-01-01T00:00:00Z")
        state = manager.load()
        assert state is not None
        assert isinstance(state, dict)

    def test_loaded_completed_ids_match_saved(self, manager):
        ids = ["qA", "qB", "qC"]
        manager.save(ids, [], "2026-01-01T00:00:00Z")
        state = manager.load()
        assert state["completed_ids"] == ids

    def test_loaded_results_match_saved(self, manager):
        results = [{"question_id": "q1", "is_correct": True}]
        manager.save(["q1"], results, "2026-01-01T00:00:00Z")
        state = manager.load()
        assert state["results"] == results

    def test_loaded_state_has_all_required_keys(self, manager):
        manager.save(["q1"], [{}], "2026-01-01T00:00:00Z")
        state = manager.load()
        assert "completed_ids" in state
        assert "results" in state
        assert "started_at" in state
        assert "last_checkpoint_at" in state

class TestCheckpointManagerSave:
    def test_save_creates_file_on_disk(self, manager, tmp_path):
        manager.save(["q1"], [{}], "2026-01-01T00:00:00Z")
        expected = tmp_path / "checkpoints" / "cond_1.json"
        assert expected.exists()

    def test_save_preserves_started_at(self, manager):
        ts = "2026-03-01T12:00:00+00:00"
        manager.save([], [], ts)
        assert manager.load()["started_at"] == ts

    def test_save_records_last_checkpoint_at(self, manager):
        manager.save([], [], "2026-01-01T00:00:00Z")
        state = manager.load()
        assert isinstance(state["last_checkpoint_at"], str)
        assert len(state["last_checkpoint_at"]) > 0

    def test_save_overwrites_previous_checkpoint(self, manager):
        manager.save(["q1"], [{"question_id": "q1"}], "2026-01-01T00:00:00Z")
        manager.save(["q1", "q2"], [{"question_id": "q1"}, {"question_id": "q2"}], "2026-01-01T00:00:00Z")
        state = manager.load()
        assert len(state["completed_ids"]) == 2
        assert len(state["results"]) == 2

    def test_checkpoint_file_is_valid_json(self, manager, tmp_path):
        manager.save(["q1"], [{"x": 1}], "2026-01-01T00:00:00Z")
        path = tmp_path / "checkpoints" / "cond_1.json"
        with path.open() as f:
            parsed = json.load(f)
        assert isinstance(parsed, dict)

    def test_checkpoint_path_encodes_condition_id(self, tmp_path):
        mgr = CheckpointManager(
            checkpoint_dir=tmp_path / "cp",
            condition_id="cond_two_stage",
        )
        mgr.save([], [], "2026-01-01T00:00:00Z")
        expected = tmp_path / "cp" / "cond_two_stage.json"
        assert expected.exists()

    def test_no_tmp_file_left_after_save(self, manager, tmp_path):
        manager.save(["q1"], [{}], "2026-01-01T00:00:00Z")
        tmp_file = tmp_path / "checkpoints" / "cond_1.tmp"
        assert not tmp_file.exists()

    def test_parent_directories_created_automatically(self, tmp_path):
        deep_path = tmp_path / "a" / "b" / "c"
        mgr = CheckpointManager(
            checkpoint_dir=deep_path,
            condition_id="cond_cyclic",
        )
        mgr.save([], [], "2026-01-01T00:00:00Z")
        assert (deep_path / "cond_cyclic.json").exists()

class TestCheckpointManagerDelete:
    def test_delete_removes_file(self, manager):
        manager.save(["q1"], [{}], "2026-01-01T00:00:00Z")
        assert manager.load() is not None
        manager.delete()
        assert manager.load() is None

    def test_delete_is_idempotent_when_no_file(self, manager):
        manager.delete()
        manager.delete()  # must not raise

    def test_delete_after_save_removes_file(self, manager, tmp_path):
        manager.save([], [], "2026-01-01T00:00:00Z")
        path = tmp_path / "checkpoints" / "cond_1.json"
        assert path.exists()
        manager.delete()
        assert not path.exists()
