# tests/test_a11_packaging_smoke.py
#
# BATCH 8 — A11: local packaging build + out-of-tree import smoke
# (F-P7-1 restoration; the network-free half of B6).
#
# Builds a wheel from this tree with --no-build-isolation --no-deps
# (no downloads), extracts it OUT of tree, and imports/uses the package
# from there with only the current venv's site-packages on the path.
# Verifies wheel content integrity: modules present, prompt resources
# packaged byte-identically, CLI entry modules importable.
#
# Skips cleanly when no setuptools-capable interpreter is available
# (record as PENDING in that environment rather than failing).

import os
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]

_EXPECTED_PROMPT_FILES = [
    "pride_repro/direct_mcq.txt", "pride_repro/free_text.txt",
    "pride_repro/option_matching.txt", "v1/direct_mcq.txt",
    "v1/free_text.txt", "v1/option_matching.txt",
]


def _build_capable_python() -> str | None:
    for candidate in (str(Path(sys.executable)), "python3"):
        try:
            probe = subprocess.run(
                [candidate, "-c", "import setuptools"],
                capture_output=True, timeout=30,
            )
        except (OSError, subprocess.TimeoutExpired):
            continue
        if probe.returncode == 0:
            return candidate
    return None


pytestmark = pytest.mark.skipif(
    _build_capable_python() is None,
    reason="no setuptools-capable interpreter available for local builds",
)


def test_wheel_builds_and_imports_out_of_tree(tmp_path):
    builder = _build_capable_python()
    assert builder is not None

    dist = tmp_path / "dist"
    dist.mkdir()
    build = subprocess.run(
        [builder, "-m", "pip", "wheel", str(_REPO_ROOT),
         "--no-build-isolation", "--no-deps", "-w", str(dist)],
        cwd=tmp_path, capture_output=True, text=True, timeout=300,
    )
    assert build.returncode == 0, build.stderr[-3000:]
    wheels = list(dist.glob("choicebench-*.whl"))
    assert len(wheels) == 1

    # Out-of-tree extraction.
    wheel_root = tmp_path / "wheelroot"
    wheel_root.mkdir()
    extract = subprocess.run(
        [sys.executable, "-m", "zipfile", "-e", str(wheels[0]), str(wheel_root)],
        capture_output=True, text=True, timeout=60,
    )
    assert extract.returncode == 0, extract.stderr

    venv_site_packages = Path(sysconfig.get_paths()["purelib"])
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(wheel_root), str(venv_site_packages)])

    smoke_script = r'''
import choicebench
from pathlib import Path
assert choicebench.__file__.startswith("@WHEEL_ROOT@"), choicebench.__file__
from choicebench.registry import METHOD_REGISTRY, CLIENT_REGISTRY
assert len(METHOD_REGISTRY) >= 7 and len(CLIENT_REGISTRY) >= 6
prompts = Path(choicebench.__file__).parent / "resources" / "prompts"
files = sorted(p.relative_to(prompts).as_posix() for p in prompts.rglob("*.txt"))
assert files == @EXPECTED_PROMPTS@, files
v1 = (prompts / "v1" / "direct_mcq.txt").read_text()
rendered = v1.format(question="Q?", options="A. x\nB. y")
assert rendered.rstrip().endswith("Respond with only the letter.")
from choicebench.cli.run_experiment import main, build_backend
from choicebench.cli.evaluate_run import parse_args
print("WHEEL SMOKE OK")
'''.replace("@WHEEL_ROOT@", str(wheel_root)).replace(
        "@EXPECTED_PROMPTS@", repr(_EXPECTED_PROMPT_FILES)
    )
    smoke = subprocess.run(
        [sys.executable, "-c", smoke_script],
        capture_output=True, text=True, timeout=120, env=env, cwd=tmp_path,
    )
    assert "WHEEL SMOKE OK" in smoke.stdout, (
        f"stdout:\n{smoke.stdout[-1500:]}\nstderr:\n{smoke.stderr[-1500:]}"
    )


def test_wheel_prompt_resources_match_source_prompts():
    """The wheel's packaged prompts must mirror prompts/ at build time."""
    source = sorted(
        p.relative_to(_REPO_ROOT / "prompts").as_posix()
        for p in (_REPO_ROOT / "prompts").rglob("*.txt")
    )
    assert source == _EXPECTED_PROMPT_FILES


def test_source_and_packaged_prompts_are_byte_identical():
    root = Path(__file__).resolve().parents[1]
    source = root / "prompts"
    packaged = root / "src" / "choicebench" / "resources" / "prompts"
    source_files = sorted(path.relative_to(source) for path in source.rglob("*.txt"))
    assert source_files == sorted(
        path.relative_to(packaged) for path in packaged.rglob("*.txt")
    )
    for relative in source_files:
        assert (source / relative).read_bytes() == (packaged / relative).read_bytes()
