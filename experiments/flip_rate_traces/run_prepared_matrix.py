"""Execute one approved remaining flip-rate inference matrix.

This module has no smoke-test or row-limit mode. Merely importing it or
loading a config performs no provider call and does not load local weights.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def _load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if value and key not in os.environ:
            os.environ[key] = value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    _load_env_file(repo_root.parent / "two-stage-prompting" / ".env")
    from experiments.flip_rate_traces.prepared_runner import run_config

    run_config(args.config)


if __name__ == "__main__":
    main()
