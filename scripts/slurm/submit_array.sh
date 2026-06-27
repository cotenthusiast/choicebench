#!/bin/bash
# Array job submission for mcq-framework.
#
# Runs multiple config files in parallel as a SLURM job array — one array
# task per config. Useful for sweeping over models, benchmarks, or methods
# without submitting each job manually.
#
# Array task i runs CONFIGS[i-1] (1-indexed because SLURM arrays start at 1).
# The number of tasks is set automatically from the length of the CONFIGS array
# below — edit that list to change what runs.
#
# Prerequisites: same as submit_job.sh.
#
# Usage:
#   sbatch scripts/slurm/submit_array.sh
#
# Monitor progress:
#   squeue -u $USER
#   tail -f logs/array_<JOB_ID>_<TASK_ID>.out

# ── Edit this list — one config per line ─────────────────────────────────────
CONFIGS=(
    "config/toy_experiment.yaml"
    # "config/my_experiment_model_a.yaml"
    # "config/my_experiment_model_b.yaml"
)
# ─────────────────────────────────────────────────────────────────────────────

NUM_CONFIGS=${#CONFIGS[@]}

#SBATCH --job-name=mcq_array
#SBATCH --output=logs/array_%A_%a.out    # %A = job array ID, %a = task index
#SBATCH --error=logs/array_%A_%a.err
#SBATCH --time=24:00:00                  # wall-clock limit per task
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --partition=gpu                  # CHANGE to your cluster's GPU partition name
#SBATCH --gres=gpu:1                     # one GPU per task
#SBATCH --array=1-${NUM_CONFIGS}         # one task per config (1-indexed)

set -euo pipefail

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV_DIR="$HOME/venvs/mcq-framework"

export HF_HOME="${HF_HOME:-$HOME/hf}"
export HF_HUB_CACHE="$HF_HOME/hub"

cd "$REPO_ROOT"
mkdir -p logs

# module load python3/3.11 cuda/12.1

source "$VENV_DIR/bin/activate"

# ── Select config for this array task ────────────────────────────────────────
# SLURM_ARRAY_TASK_ID is 1-indexed; bash arrays are 0-indexed.
TASK_INDEX=$(( SLURM_ARRAY_TASK_ID - 1 ))
CONFIG="${CONFIGS[$TASK_INDEX]}"

if [[ ! -f "$CONFIG" ]]; then
    echo "ERROR: Config file not found: $CONFIG" >&2
    exit 1
fi

# ── Diagnostics ───────────────────────────────────────────────────────────────
echo "Array job:   ${SLURM_ARRAY_JOB_ID:-local}"
echo "Task index:  ${SLURM_ARRAY_TASK_ID:-1}"
echo "Node:        ${SLURMD_NODENAME:-$(hostname)}"
echo "GPU:         ${CUDA_VISIBLE_DEVICES:-none}"
echo "Python:      $(python --version)"
echo "Repo:        $REPO_ROOT"
echo "Config:      $CONFIG"

# ── Run ───────────────────────────────────────────────────────────────────────
# RUN_ID encodes both the job array ID and task index for traceability.
RUN_ID="${SLURM_ARRAY_JOB_ID:-local}_task${SLURM_ARRAY_TASK_ID:-1}"

python scripts/run_experiment.py --config "$CONFIG" --run-id "$RUN_ID" --yes

echo "Task complete."
