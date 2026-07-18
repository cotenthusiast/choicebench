import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path


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
        subprocess.run(
            [str(venv / "bin" / command), *args], cwd=workspace, env=env,
            check=True, capture_output=True, text=True,
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
    subprocess.run(
        [build_python, "-m", "build", "--no-isolation", "--outdir", str(dist), str(root)],
        cwd=tmp_path, check=True, capture_output=True, text=True,
    )
    wheel = next(dist.glob("choicebench-*.whl"))
    sdist = next(dist.glob("choicebench-*.tar.gz"))
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
    assert any(name.endswith("resources/prompts/v1/direct_mcq.txt") for name in names)
    assert not any("/runs/" in name or "/data/processed/" in name for name in names)

    clean_venv = tmp_path / "clean-venv"
    subprocess.run([sys.executable, "-m", "venv", str(clean_venv)], check=True)
    subprocess.run(
        [str(clean_venv / "bin" / "pip"), "install", "--no-deps", str(wheel)],
        check=True, capture_output=True, text=True,
    )
    subprocess.run(
        [str(clean_venv / "bin" / "python"), "-c",
         "import choicebench; from choicebench.pipeline.prompt_builder import load_prompt_templates; "
         "assert 'Answer' in load_prompt_templates('v1')['direct_mcq']"],
        cwd=tmp_path, check=True, capture_output=True, text=True,
    )

    # Functional CLI smoke reuses the test environment's already-installed
    # dependencies. The release verification separately performs a true clean
    # dependency install from the built wheel.
    venv = tmp_path / "workflow-venv"
    subprocess.run([sys.executable, "-m", "venv", "--system-site-packages", str(venv)], check=True)
    pip = venv / "bin" / "pip"
    subprocess.run([str(pip), "install", "--no-deps", str(wheel)], check=True, capture_output=True, text=True)
    for command in ("choicebench-prepare-toy", "choicebench-run", "choicebench-evaluate", "choicebench-prepare"):
        subprocess.run([str(venv / "bin" / command), "--help"], check=True, capture_output=True, text=True)
    _run_workflow(venv, tmp_path / "wheel workspace é", "wheel")

    subprocess.run([str(pip), "uninstall", "-y", "choicebench"], check=True, capture_output=True, text=True)
    subprocess.run(
        [str(pip), "install", "--no-deps", "--no-build-isolation", str(sdist)],
        check=True, capture_output=True, text=True,
    )
    _run_workflow(venv, tmp_path / "sdist workspace é", "sdist")
