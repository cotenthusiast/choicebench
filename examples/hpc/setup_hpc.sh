#!/usr/bin/env bash
# One-time ChoiceBench setup helper for editable HPC/Slurm environments.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_REPO="$(cd "$SCRIPT_DIR/../.." && pwd)"
DEFAULT_BASE="${HOME:-$PWD}/choicebench_hpc"

export CHOICEBENCH_BASE="${CHOICEBENCH_BASE:-$DEFAULT_BASE}"
export CHOICEBENCH_REPO="${CHOICEBENCH_REPO:-$DEFAULT_REPO}"
export CHOICEBENCH_VENV="${CHOICEBENCH_VENV:-$CHOICEBENCH_BASE/.venv}"
export HF_HOME="${HF_HOME:-$CHOICEBENCH_BASE/hf}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export CHOICEBENCH_PYTHON_MODULE="${CHOICEBENCH_PYTHON_MODULE:-}"
export PYTHON_BIN="${PYTHON_BIN:-}"

if [[ -n "$CHOICEBENCH_PYTHON_MODULE" ]]; then
    if type module >/dev/null 2>&1; then
        module load "$CHOICEBENCH_PYTHON_MODULE"
    else
        echo "WARNING: CHOICEBENCH_PYTHON_MODULE is set, but 'module' is not available." >&2
    fi
fi

if [[ ! -d "$CHOICEBENCH_REPO" ]]; then
    echo "ERROR: ChoiceBench repo directory not found: $CHOICEBENCH_REPO" >&2
    exit 1
fi

if [[ ! -f "$CHOICEBENCH_REPO/pyproject.toml" ]]; then
    echo "ERROR: $CHOICEBENCH_REPO does not look like the ChoiceBench repo." >&2
    exit 1
fi

if [[ -z "$PYTHON_BIN" ]]; then
    if command -v python3 >/dev/null 2>&1; then
        PYTHON_BIN="$(command -v python3)"
    elif command -v python >/dev/null 2>&1; then
        PYTHON_BIN="$(command -v python)"
    else
        echo "ERROR: Could not find python3 or python. Load a Python module or set PYTHON_BIN." >&2
        exit 1
    fi
fi

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "ERROR: PYTHON_BIN is not executable or not on PATH: $PYTHON_BIN" >&2
    exit 1
fi

mkdir -p "$CHOICEBENCH_BASE" "$(dirname "$CHOICEBENCH_VENV")" "$HF_HOME" "$HF_HUB_CACHE"

"$PYTHON_BIN" -m venv "$CHOICEBENCH_VENV"
# shellcheck source=/dev/null
source "$CHOICEBENCH_VENV/bin/activate"

cd "$CHOICEBENCH_REPO"

python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python -c "import choicebench; print('ChoiceBench import OK')"

echo "ChoiceBench HPC setup complete."
echo "Repo: $CHOICEBENCH_REPO"
echo "Venv: $CHOICEBENCH_VENV"
echo "HF cache: $HF_HUB_CACHE"
