# tests/scripts/test_reset_run.py
#
# The --reset-run CLI flag must gate the safe_reset_run_dir() call: reset only
# happens when the flag is passed, and the default (reuse-in-place) behavior is
# otherwise untouched. Deletion safety itself is covered by
# tests/config/test_paths.py; here we only assert the wiring.

import importlib
import pathlib
import sys
import types
import pytest

from choicebench.config.schema import (
    BenchmarkConfig,
    ExperimentConfig,
    MethodConfig,
    ModelConfig,
    RunConfig,
)

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]


def _load_run_experiment():
    import choicebench.cli.run_experiment as module
    return importlib.reload(module)


def _config(resume: bool = True) -> ExperimentConfig:
    return ExperimentConfig(
        name="unit",
        models=[ModelConfig(backend="dummy", model_name_or_path="d1", device="cpu")],
        benchmarks=[BenchmarkConfig(name="toy")],
        methods=[MethodConfig(name="direct_mcq")],
        metrics=["accuracy"],
        run=RunConfig(seed=0, prompt_version="v1", resume=resume),
    )


def _run_main(tmp_path, monkeypatch, reset_run: bool, resume: bool = True):
    """Drive main() with everything but the reset wiring stubbed out.

    Returns (reset_calls, output_dir).
    """
    run_exp = _load_run_experiment()
    runs_root = tmp_path / "runs"
    monkeypatch.setattr(run_exp, "RUNS_DIR", runs_root)
    monkeypatch.setattr(run_exp, "ensure_dirs", lambda: None)
    monkeypatch.setattr(run_exp, "validate_logprob_compatibility", lambda cfg: None)

    async def _fake_async_main(*args, **kwargs):
        return [], []

    monkeypatch.setattr(run_exp, "_async_main", _fake_async_main)

    reset_calls: list[pathlib.Path] = []

    def _recording_reset(run_dir, allowed_root=None):
        reset_calls.append(pathlib.Path(run_dir))
        return pathlib.Path(run_dir)

    monkeypatch.setattr(run_exp, "safe_reset_run_dir", _recording_reset)

    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text("experiment:\n  name: unit\n")
    monkeypatch.setattr(run_exp, "load_config", lambda path: _config(resume=resume))
    monkeypatch.setattr(
        run_exp,
        "parse_args",
        lambda: types.SimpleNamespace(
            config=str(cfg_path),
            dry_run=False,
            run_id="rid",
            yes=True,
            reset_run=reset_run,
        ),
    )

    run_exp.main()
    return reset_calls, runs_root / "rid"


def test_reset_run_flag_triggers_reset(tmp_path, monkeypatch):
    reset_calls, output_dir = _run_main(tmp_path, monkeypatch, reset_run=True)
    assert reset_calls == [output_dir]


def test_no_reset_flag_rejects_unverified_legacy_run(tmp_path, monkeypatch, caplog):
    # Pre-populate the run directory to represent a prior run's output.
    output_dir = tmp_path / "runs" / "rid"
    output_dir.mkdir(parents=True)
    marker = output_dir / "prior_result.csv"
    marker.write_text("kept")

    # main() converts the expected refusal into a clean logged error + exit 1.
    with pytest.raises(SystemExit) as excinfo:
        _run_main(tmp_path, monkeypatch, reset_run=False)
    assert excinfo.value.code == 1
    assert "no verified manifest" in caplog.text
    assert marker.exists()


def test_compatible_rerun_without_resume_refuses_before_touching_state(tmp_path, monkeypatch, caplog):
    """CB-2: rerunning an identical experiment with resume disabled must refuse
    up front and leave the existing run byte-identical (never downgrade
    completed conditions to failed)."""
    _run_main(tmp_path, monkeypatch, reset_run=False, resume=False)
    run_dir = tmp_path / "runs" / "rid"
    before = {
        path.name: path.read_bytes()
        for path in run_dir.rglob("*") if path.is_file()
    }
    assert "manifest.json" in before and "run_state.json" in before

    with pytest.raises(SystemExit) as excinfo:
        _run_main(tmp_path, monkeypatch, reset_run=False, resume=False)
    assert excinfo.value.code == 1
    assert "already exists for this exact experiment" in caplog.text

    after = {
        path.name: path.read_bytes()
        for path in run_dir.rglob("*") if path.is_file()
    }
    assert after == before


def test_compatible_rerun_with_resume_is_accepted(tmp_path, monkeypatch):
    _run_main(tmp_path, monkeypatch, reset_run=False, resume=True)
    # Second invocation with resume enabled proceeds (no SystemExit).
    _run_main(tmp_path, monkeypatch, reset_run=False, resume=True)


def test_parse_args_reset_run_flag(monkeypatch):
    run_exp = _load_run_experiment()
    monkeypatch.setattr(
        sys, "argv", ["run_experiment.py", "--config", "c.yaml", "--reset-run"]
    )
    assert run_exp.parse_args().reset_run is True


def test_parse_args_reset_run_defaults_false(monkeypatch):
    run_exp = _load_run_experiment()
    monkeypatch.setattr(sys, "argv", ["run_experiment.py", "--config", "c.yaml"])
    assert run_exp.parse_args().reset_run is False
