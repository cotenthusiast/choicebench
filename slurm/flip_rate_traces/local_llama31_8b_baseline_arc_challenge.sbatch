#!/bin/bash
#SBATCH --job-name=fr_llama31_arc_base
#SBATCH --partition=k2-gpu-a100
#SBATCH --gres=gpu:a100:1
#SBATCH --time=24:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --output=/mnt/scratch2/users/40482774/logs/flip_rate_traces/%x_%j.log
#SBATCH --error=/mnt/scratch2/users/40482774/logs/flip_rate_traces/%x_%j.log

set -euo pipefail

REPO="/mnt/scratch2/users/40482774/repos/choicebench-flip-rate-traces"
PINNED_SHA="e3fd2cfa492a6d4f7a3d8799de9b6620ad31ecca"

cd "${REPO}"
git fetch origin
git reset --hard "${PINNED_SHA}"

export CHOICEBENCH_FLIP_RATE_REPO="${REPO}"
export FLIP_RATE_TRACES_FREEZE_ROOT="/mnt/scratch2/users/40482774/paper_data_freeze_subset"
export HF_HOME="/mnt/scratch2/users/40482774/hf"
export HF_HUB_CACHE="${HF_HOME}/hub"
export PYTHONPATH="${REPO}/src:${REPO}"

source /mnt/scratch2/users/40482774/venvs/mcq-generalization/bin/activate

python -m experiments.flip_rate_traces.run_prepared_matrix \
    --config configs/flip_rate_traces/prepared_kelvin2_llama31_8b_baseline_arc_challenge.yaml
