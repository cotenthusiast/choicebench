import json
import os
import shlex
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path


def _run_checked(command, **kwargs):
    result = subprocess.run(
        command, check=False, capture_output=True, text=True, **kwargs,
    )
    if result.returncode != 0:
        raise AssertionError(
            f"Command failed ({result.returncode}): "
            f"{shlex.join(str(part) for part in command)}\n"
            f"stdout:\n{result.stdout or '<empty>'}\n"
            f"stderr:\n{result.stderr or '<empty>'}"
        )
    return result


def _run_workflow(venv: Path, workspace: Path, label: str) -> None:
    workspace.mkdir()
    config = workspace / "toy.yaml"
    config.write_text("""experiment:
  name: installed-toy
models:
  - backend: dummy
    model_name_or_path: installed-dummy
    device: cpu
benchmarks:
  - name: toy
    split: test
    n_samples: 2
methods:
  - name: direct_mcq
metrics: [accuracy]
run:
  seed: 42
  resume: true
  checkpoint_every_n: 1
  concurrency_limit: 1
  prompt_version: v1
""")
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env["CHOICEBENCH_HOME"] = str(workspace)
    commands = [
        ("choicebench-prepare-toy", []),
        ("choicebench-run", ["--config", str(config), "--run-id", label, "--yes"]),
        # Identical completed runs must resume/verify without rewriting results.
        ("choicebench-run", ["--config", str(config), "--run-id", label, "--yes"]),
        ("choicebench-evaluate", ["--run-id", label]),
    ]
    for command, args in commands:
        _run_checked(
            [str(venv / "bin" / command), *args], cwd=workspace, env=env,
        )
    assert (workspace / "runs" / label / "manifest.json").exists()
    reports = list((workspace / "reports").glob(f"{label}_eval_*_metrics.json"))
    assert len(reports) == 1
    report = json.loads(reports[0].read_text())
    assert report["source"]["run_id"] == label
    assert report["evaluation_id"].startswith("eval_")


def test_built_wheel_and_sdist_run_outside_repository(tmp_path):
    root = Path(__file__).resolve().parents[1]
    dist = tmp_path / "dist"
    build_python = shutil.which("python3") or sys.executable
    _run_checked(
        [build_python, "-m", "build", "--outdir", str(dist), str(root)],
        cwd=tmp_path,
    )
    wheel = next(dist.glob("choicebench-*.whl"))
    sdist = next(dist.glob("choicebench-*.tar.gz"))
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
    assert any(name.endswith("resources/prompts/v1/direct_mcq.txt") for name in names)
    assert not any("/runs/" in name or "/data/processed/" in name for name in names)

    clean_venv = tmp_path / "clean-venv"
    subprocess.run([sys.executable, "-m", "venv", str(clean_venv)], check=True)
    _run_checked(
        [str(clean_venv / "bin" / "pip"), "install", "--no-deps", str(wheel)],
    )
    _run_checked(
        [str(clean_venv / "bin" / "python"), "-c",
         "import choicebench; from choicebench.pipeline.prompt_builder import load_prompt_templates; "
         "assert 'Answer' in load_prompt_templates('v1')['direct_mcq']"],
        cwd=tmp_path,
    )

    # A fully isolated venv with the wheel's declared dependencies installed
    # alongside it. --system-site-packages previously stood in for this, but
    # under a truly clean venv/container it inherits an empty (or unrelated)
    # site-packages rather than this test's own dependencies, so the CLI
    # commands below would crash on missing imports (e.g. pandas) outside
    # this one machine's incidentally-populated user site-packages.
    venv = tmp_path / "workflow-venv"
    subprocess.run([sys.executable, "-m", "venv", str(venv)], check=True)
    pip = venv / "bin" / "pip"
    _run_checked([str(pip), "install", str(wheel)])
    for command in ("choicebench-prepare-toy", "choicebench-run", "choicebench-evaluate", "choicebench-prepare"):
        _run_checked([str(venv / "bin" / command), "--help"])
    _run_workflow(venv, tmp_path / "wheel workspace é", "wheel")

    _run_checked([str(pip), "uninstall", "-y", "choicebench"])
    _run_checked([str(pip), "install", "--no-deps", str(sdist)])
    _run_workflow(venv, tmp_path / "sdist workspace é", "sdist")


def test_sdist_builds_from_repo_root_and_registers_entry_points(tmp_path):
    """A stray untracked build artifact (e.g. a leftover build/ directory
    from a prior local `python -m build`) left in the repo root can shadow
    or otherwise confuse packaging tools for anyone building from a repo
    checkout instead of an external directory, producing a malformed
    package that loses its console-script entry points. Guard that building
    the sdist with cwd=repo root still produces a correctly-named artifact
    whose entry points resolve after a fresh install."""
    root = Path(__file__).resolve().parents[1]
    dist = tmp_path / "dist"
    build_python = shutil.which("python3") or sys.executable
    _run_checked(
        [build_python, "-m", "build", "--sdist", "--outdir", str(dist), str(root)],
        cwd=root,
    )
    sdists = list(dist.glob("choicebench-*.tar.gz"))
    assert len(sdists) == 1
    assert not sdists[0].name.startswith("UNKNOWN"), sdists[0].name

    venv = tmp_path / "sdist-entrypoint-venv"
    subprocess.run([sys.executable, "-m", "venv", str(venv)], check=True)
    _run_checked([str(venv / "bin" / "pip"), "install", "--no-deps", str(sdists[0])])
    for command in ("choicebench-run", "choicebench-prepare", "choicebench-evaluate", "choicebench-prepare-toy"):
        assert (venv / "bin" / command).exists(), f"entry point {command} missing after sdist install"
