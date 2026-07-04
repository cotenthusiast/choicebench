#!/bin/bash
# Single job submission for choicebench.
#
# Runs one config file on one GPU node. Suitable for API-backed models (no GPU
# needed) and for local models up to ~30B parameters on a single A100 (80 GB).
# For 70B+ models increase --gres / memory as needed. Use submit_array.sh for
# multiple independent configs, not for sharding one model across tasks.
#
# Prerequisites:
#   - Repo cloned to a scratch or project directory on the HPC cluster
#   - A Python virtualenv (or conda env) at $VENV_DIR with dependencies installed
#   - For HuggingFace models: HF_HOME pointing to a fast scratch filesystem
#   - For API models: .env file in the repo root with API keys
#     (never put secrets in this script; .env should be in .gitignore)
#
# Usage:
#   CONFIG=config/my_experiment.yaml sbatch scripts/slurm/submit_job.sh
#   CONFIG=config/my_experiment.yaml RUN_ID=my_run_20260627 sbatch scripts/slurm/submit_job.sh
#
# Check available partitions on your cluster:
#   sinfo -o "%P %D %G %m %l %N"

#SBATCH --job-name=mcq_job
#SBATCH --output=logs/job_%j.out         # stdout — %j is the SLURM job ID
#SBATCH --error=logs/job_%j.err          # stderr
#SBATCH --time=24:00:00                  # wall-clock time limit (HH:MM:SS)
#SBATCH --nodes=1                        # single node
#SBATCH --ntasks=1                       # one task (the Python process)
#SBATCH --cpus-per-task=8               # CPU threads for data loading / tokenization
#SBATCH --mem=80G                        # system RAM; increase for very large models
#SBATCH --partition=gpu                  # CHANGE to your cluster's GPU partition name
#SBATCH --gres=gpu:1                     # one GPU; change to gpu:a100:1 etc. as needed

set -euo pipefail

# ── Paths ─────────────────────────────────────────────────────────────────────
# CHANGE these to match your cluster layout.
# BASH_SOURCE breaks here: sbatch copies the submitted script to a spool
# directory before executing it, so BASH_SOURCE resolves to the spool path,
# not this file's real location. SLURM_SUBMIT_DIR is the directory sbatch
# was invoked from and stays stable across that copy.
REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
VENV_DIR="$HOME/venvs/choicebench"        # path to virtualenv

# ── HuggingFace cache (only needed for local HF models) ─────────────────────
# Point HF_HOME at a fast filesystem (scratch) rather than $HOME to avoid
# quota issues when downloading large model weights.
export HF_HOME="${HF_HOME:-$HOME/hf}"
export HF_HUB_CACHE="$HF_HOME/hub"

# ── Setup ─────────────────────────────────────────────────────────────────────
cd "$REPO_ROOT"
mkdir -p logs

# Load Python if your cluster uses environment modules.
# Comment out or change the module name to match your system.
# module load python3/3.11 cuda/12.1

source "$VENV_DIR/bin/activate"

# ── Validate required inputs ──────────────────────────────────────────────────
if [[ -z "${CONFIG:-}" ]]; then
    echo "ERROR: CONFIG environment variable is not set." >&2
    echo "Usage: CONFIG=config/my_experiment.yaml sbatch scripts/slurm/submit_job.sh" >&2
    exit 1
fi

if [[ ! -f "$CONFIG" ]]; then
    echo "ERROR: Config file not found: $CONFIG" >&2
    exit 1
fi

# ── Diagnostics ───────────────────────────────────────────────────────────────
echo "Job ID:  ${SLURM_JOB_ID:-local}"
echo "Node:    ${SLURMD_NODENAME:-$(hostname)}"
echo "GPU:     ${CUDA_VISIBLE_DEVICES:-none}"
echo "Python:  $(python --version)"
echo "Repo:    $REPO_ROOT"
echo "Config:  $CONFIG"
echo "Run ID:  ${RUN_ID:-auto}"

# ── Run ───────────────────────────────────────────────────────────────────────
if [[ -n "${RUN_ID:-}" ]]; then
    python scripts/run_experiment.py --config "$CONFIG" --run-id "$RUN_ID" --yes
else
    python scripts/run_experiment.py --config "$CONFIG" --yes
fi

echo "Job complete."
