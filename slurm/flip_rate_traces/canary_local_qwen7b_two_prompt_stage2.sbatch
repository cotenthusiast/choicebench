#!/bin/bash
#SBATCH --job-name=fr_tp_stage2_canary
#SBATCH --partition=k2-gpu-a100
#SBATCH --gres=gpu:a100:1
#SBATCH --time=00:30:00
#SBATCH --mem=48G
#SBATCH --output=fr_tp_stage2_canary_%j.log

set -euo pipefail

REPO="/mnt/scratch2/users/40482774/repos/choicebench-flip-rate-traces"
MODELGEN_REPO="/mnt/scratch2/users/40482774/repos/model-generalization"
PINNED_SHA="572018cae72b35c8db1af2440d9e5eb998978b82"

cd "${REPO}"
git fetch origin
git reset --hard "${PINNED_SHA}"

export FLIP_RATE_TRACES_FREEZE_ROOT="/mnt/scratch2/users/40482774/paper_data_freeze_subset"
export HF_HOME="/mnt/scratch2/users/40482774/hf"
export HF_HUB_CACHE="${HF_HOME}/hub"
export CHOICEBENCH_FLIP_RATE_REPO="${REPO}"
export MODELGEN_REPO="${MODELGEN_REPO}"
export PYTHONPATH="${REPO}/src:${REPO}"

source /mnt/scratch2/users/40482774/venvs/mcq-generalization/bin/activate

python -m experiments.flip_rate_traces.task1a.gen_two_prompt_stage2_local --benchmark arc_challenge --canary 5
