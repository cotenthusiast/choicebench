# tests/test_a7_lifecycle_drill.py
#
# BATCH 8 — A7: lifecycle drill through the REAL CLI in an isolated
# ChoiceBench workspace (CHOICEBENCH_HOME -> tmp_path), dummy backend only:
#   fresh run -> resume refusal -> resume skip (+ P5-1 checkpoint GC)
#   -> tamper detection by the evaluator -> corrupt-result recovery from a
#   verified checkpoint -> partial accounting (one failed condition).
#
# No network, no GPU, no API keys: toy dataset + DummyBackend.

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]

_CONFIG_TEMPLATE = """
experiment:
  name: drill_experiment

models:
  - backend: dummy
    model_name_or_path: dummy_1
    device: cpu
    generation_kwargs:
      max_new_tokens: 16
      temperature: 0.0
      do_sample: false
  - backend: dummy
    model_name_or_path: dummy_2
    device: cpu
    generation_kwargs:
      max_new_tokens: 16
      temperature: 0.0
      do_sample: false

benchmarks:
  - name: toy
    split: test
    n_samples: 10

methods:
  - name: direct_mcq

metrics:
  - accuracy

run:
  seed: 42
  resume: {resume}
  dry_run: false
  checkpoint_every_n: 3
  prompt_version: "v1"
  concurrency_limit: 5
"""


def _write_config(workspace: Path, resume: bool) -> Path:
    path = workspace / "drill_config.yaml"
    path.write_text(_CONFIG_TEMPLATE.format(resume=str(resume).lower()))
    return path


@pytest.fixture()
def workspace(tmp_path, monkeypatch):
    ws = tmp_path / "choicebench_home"
    ws.mkdir()
    monkeypatch.setenv("CHOICEBENCH_HOME", str(ws))
    return ws


def _run_cli(workspace: Path, script: str, *args: str) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env["CHOICEBENCH_HOME"] = str(workspace)
    return subprocess.run(
        [sys.executable, str(_REPO_ROOT / "scripts" / script), *args],
        cwd=_REPO_ROOT, env=env, capture_output=True, text=True, timeout=300,
    )


def _experiment(workspace: Path):
    """Run the experiment; returns CompletedProcess."""
    cfg = _write_config(workspace, resume=False)
    return _run_cli(workspace, "run_experiment.py", "--config", str(cfg),
                    "--run-id", "drill", "--yes")


def _resume(workspace: Path):
    cfg = _write_config(workspace, resume=True)
    return _run_cli(workspace, "run_experiment.py", "--config", str(cfg),
                    "--run-id", "drill", "--yes")


def _results_dir(workspace: Path) -> Path:
    return workspace / "runs" / "drill" / "results"


def _result_csvs(workspace: Path) -> list[Path]:
    return sorted((_results_dir(workspace)).glob("*.csv"))


def _sanitize_for_checkpoint(rows: list[dict]) -> list[dict]:
    """pd.read_csv turns empty cells into NaN; real in-memory result rows
    carry None. Checkpoint identities reject non-finite floats, so restore
    None the way the live pipeline had them."""
    return [
        {
            k: (None if isinstance(v, float) and v != v else v)
            for k, v in row.items()
        }
        for row in rows
    ]


class TestA7LifecycleDrill:
    def test_full_drill(self, workspace):
        # --- Stage 0: prepare toy data -------------------------------
        prep = _run_cli(workspace, "prepare_toy_data.py")
        assert prep.returncode == 0, prep.stderr[-2000:]

        # --- Stage 1: FRESH run completes ----------------------------
        fresh = _experiment(workspace)
        assert fresh.returncode == 0, (
            f"stdout:\n{fresh.stdout[-3000:]}\nstderr:\n{fresh.stderr[-3000:]}"
        )
        csvs = _result_csvs(workspace)
        assert len(csvs) == 2  # two dummy models -> two conditions
        run_dir = workspace / "runs" / "drill"
        manifest = json.loads((run_dir / "manifest.json").read_text())
        conditions = manifest["payload"]["conditions"]
        assert {c["condition_id"] for c in conditions} == {
            p.stem for p in csvs
        }
        state = json.loads((run_dir / "run_state.json").read_text())
        assert all(
            info.get("status") == "completed"
            for info in state["conditions"].values()
        )
        assert not (run_dir / "checkpoints").exists() or not any(
            (run_dir / "checkpoints").iterdir()
        ), "completed run must leave no checkpoints behind"

        good_rows = {
            p.stem: pd.read_csv(p) for p in csvs
        }

        # --- Stage 2: REFUSAL without resume -------------------------
        refusal = _experiment(workspace)
        assert refusal.returncode == 1
        combined = refusal.stdout + refusal.stderr
        assert "resume" in combined.lower()

        # --- Stage 3: RESUME skips verified-complete conditions and
        #     garbage-collects a stale planted checkpoint (P5-1) -------
        checkpoints = run_dir / "checkpoints"
        checkpoints.mkdir(exist_ok=True)
        stale = checkpoints / f"{conditions[0]['condition_id']}.json"
        stale.write_text('{"stale": true}')
        skipped = _resume(workspace)
        assert skipped.returncode == 0, skipped.stderr[-2000:]
        assert "Verified completed result" in skipped.stderr + skipped.stdout
        assert not stale.exists(), "skip path must GC the stale checkpoint"

        # --- Stage 4: TAMPERED result is caught by the evaluator -----
        target = csvs[0]
        frame = pd.read_csv(target)
        frame.loc[0, "is_correct"] = not bool(frame.loc[0, "is_correct"])
        frame.to_csv(target, index=False)
        evaluation = _run_cli(workspace, "evaluate_run.py", "--run-id", "drill")
        assert evaluation.returncode == 1, (
            "evaluator must refuse a tampered committed result\n"
            f"{evaluation.stdout[-1500:]}\n{evaluation.stderr[-1500:]}"
        )

        # --- Stage 5: CORRUPT RESULT + VERIFIED CHECKPOINT recovers --
        from choicebench.identity import integrity_digest

        condition_zero = next(
            c for c in conditions if c["condition_id"] == target.stem
        )
        original_rows = _sanitize_for_checkpoint(
            good_rows[target.stem].to_dict(orient="records")
        )
        checkpoint_state = {
            "schema_version": "choicebench.checkpoint.v1",
            "experiment_id": manifest["experiment_id"],
            "condition_id": condition_zero["condition_id"],
            "selection_id": condition_zero["selection_id"],
            "completed_ids": [str(r["question_id"]) for r in original_rows],
            "results": original_rows,
            "started_at": "2026-08-23T00:00:00+00:00",
            "last_checkpoint_at": "2026-08-23T00:10:00+00:00",
        }
        checkpoint_state["checkpoint_digest"] = integrity_digest(
            {k: v for k, v in checkpoint_state.items()}
        )
        (checkpoints / f"{condition_zero['condition_id']}.json").write_text(
            json.dumps(checkpoint_state)
        )
        # Corrupt the committed result beyond repair.
        target.write_text("garbage,not,a,result\n1,2,3\n")

        recovered = _resume(workspace)
        assert recovered.returncode == 0, recovered.stderr[-2000:]
        assert "Discarding an uncommitted/corrupt result" in (
            recovered.stdout + recovered.stderr
        )
        rebuilt = pd.read_csv(target)
        # Round-trip the expected rows through the same CSV serialization
        # so dtype/NaN treatment matches before comparing values.
        expected_csv_path = Path(target).with_suffix(".expected.csv")
        pd.DataFrame(original_rows).to_csv(expected_csv_path, index=False)
        expected_frame = pd.read_csv(expected_csv_path)
        expected_csv_path.unlink()
        assert len(rebuilt) == len(expected_frame)
        for col in expected_frame.columns:
            assert (
                rebuilt[col].fillna("<NA>").astype(str).tolist()
                == expected_frame[col].fillna("<NA>").astype(str).tolist()
            ), col
        assert not (
            checkpoints / f"{condition_zero['condition_id']}.json"
        ).exists(), "recovery must consume the checkpoint"

        # Evaluator is green again after recovery.
        post_recovery = _run_cli(workspace, "evaluate_run.py", "--run-id", "drill")
        assert post_recovery.returncode == 0, post_recovery.stderr[-1500:]

        # --- Stage 6: PARTIAL ACCOUNTING — one condition fails, the
        #     other still verifies-and-skips; evaluator reports partial --
        target.write_text("still garbage\n")
        partial_run = _resume(workspace)
        assert partial_run.returncode == 1, (
            "the unrecoverable condition must fail the run"
        )
        state_after = json.loads((run_dir / "run_state.json").read_text())
        statuses = {
            cid: info.get("status") for cid, info in
            state_after["conditions"].items()
        }
        assert statuses[target.stem] == "failed"
        assert statuses[csvs[1].stem] == "completed"

        final_eval = _run_cli(workspace, "evaluate_run.py", "--run-id", "drill")
        # The evaluator fail-closed on the FIRST attempt above because a
        # failed condition must not leave a result artifact behind; remove
        # the garbage file so the run is in a coherent partial state.
        if final_eval.returncode != 0:
            assert "failed condition has a result artifact" in (
                final_eval.stdout + final_eval.stderr
            )
            target.unlink()
            final_eval = _run_cli(
                workspace, "evaluate_run.py", "--run-id", "drill"
            )
        assert final_eval.returncode == 0, final_eval.stderr[-1500:]
        combined_out = final_eval.stdout + final_eval.stderr
        assert "PARTIAL accounting" in combined_out
