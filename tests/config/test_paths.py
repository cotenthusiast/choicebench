# tests/config/test_paths.py
#
# safe_reset_run_dir() must only ever clear a run subdirectory of its allowed
# root, and must refuse anything that escapes it (path outside, symlink to
# outside, or the root itself).

import pytest

from choicebench.config.paths import safe_reset_run_dir


def test_reset_clears_contents_under_allowed_root(tmp_path):
    root = tmp_path / "runs"
    run_dir = root / "run1"
    run_dir.mkdir(parents=True)
    (run_dir / "result.csv").write_text("data")
    checkpoints = run_dir / "checkpoints"
    checkpoints.mkdir()
    (checkpoints / "ckpt.json").write_text("{}")

    returned = safe_reset_run_dir(run_dir, allowed_root=root)

    assert returned == run_dir.resolve()
    assert run_dir.exists()
    assert list(run_dir.iterdir()) == []


def test_reset_on_missing_dir_yields_empty_dir(tmp_path):
    root = tmp_path / "runs"
    root.mkdir()
    run_dir = root / "does_not_exist_yet"

    safe_reset_run_dir(run_dir, allowed_root=root)

    assert run_dir.exists()
    assert list(run_dir.iterdir()) == []


def test_reset_rejects_path_outside_allowed_root(tmp_path):
    root = tmp_path / "runs"
    root.mkdir()
    outside = tmp_path / "other" / "run1"
    outside.mkdir(parents=True)
    (outside / "keep.txt").write_text("x")

    with pytest.raises(ValueError, match="under the allowed root"):
        safe_reset_run_dir(outside, allowed_root=root)

    assert (outside / "keep.txt").exists()


def test_reset_rejects_symlink_resolving_outside(tmp_path):
    root = tmp_path / "runs"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "keep.txt").write_text("x")

    # A run-dir-looking symlink inside the allowed root that actually points out.
    link = root / "sneaky"
    link.symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="under the allowed root"):
        safe_reset_run_dir(link, allowed_root=root)

    assert (outside / "keep.txt").exists()


def test_reset_rejects_allowed_root_itself(tmp_path):
    root = tmp_path / "runs"
    root.mkdir()
    (root / "run1").mkdir()

    with pytest.raises(ValueError, match="runs root itself"):
        safe_reset_run_dir(root, allowed_root=root)

    assert (root / "run1").exists()
