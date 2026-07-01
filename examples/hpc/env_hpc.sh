#!/usr/bin/env bash
# Source this file before running ChoiceBench on an HPC/Slurm node.

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    echo "ERROR: source this file instead of executing it:" >&2
    echo "  source ${BASH_SOURCE[0]}" >&2
    exit 1
fi

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
    return 1
fi

if [[ ! -f "$CHOICEBENCH_REPO/pyproject.toml" ]]; then
    echo "ERROR: $CHOICEBENCH_REPO does not look like the ChoiceBench repo." >&2
    return 1
fi

if [[ ! -f "$CHOICEBENCH_VENV/bin/activate" ]]; then
    echo "ERROR: virtualenv not found: $CHOICEBENCH_VENV" >&2
    echo "Run examples/hpc/setup_hpc.sh first, or set CHOICEBENCH_VENV." >&2
    return 1
fi

# shellcheck source=/dev/null
source "$CHOICEBENCH_VENV/bin/activate"

export PYTHONPATH="$CHOICEBENCH_REPO/src:${PYTHONPATH:-}"
export HF_HOME
export HF_HUB_CACHE

cd "$CHOICEBENCH_REPO" || return 1

echo "ChoiceBench HPC environment active."
echo "Repo: $CHOICEBENCH_REPO"
echo "Venv: $CHOICEBENCH_VENV"
echo "HF cache: $HF_HUB_CACHE"
