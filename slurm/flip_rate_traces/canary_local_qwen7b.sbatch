#!/bin/bash
#SBATCH --job-name=flip_rate_canary_qwen7b
#SBATCH --partition=k2-gpu-a100
#SBATCH --gres=gpu:a100:1
#SBATCH --time=00:30:00
#SBATCH --mem=32G
#SBATCH --output=flip_rate_canary_qwen7b_%j.log
#
# Two real local canaries: one ordinary 4-option ARC question, one genuine
# 3-option ARC question (KNOWN_3OPTION_ARC_QUESTION_IDS). Writes only to
# runs/flip_rate_traces/_canary/ (is_diagnostic_canary=True on every row).

set -euo pipefail

REPO="/mnt/scratch2/users/40482774/repos/choicebench-flip-rate-traces"
PINNED_SHA="60db048299ee71bd50c4d85bffc25daaab8bc4f0"

cd "${REPO}"
git fetch origin
git checkout --detach "${PINNED_SHA}"

export HF_HOME="/mnt/scratch2/users/40482774/hf"
export FLIP_RATE_TRACES_FREEZE_ROOT="/mnt/scratch2/users/40482774/paper_data_freeze_subset"
export HF_HUB_CACHE="${HF_HOME}/hub"
export PYTHONPATH="${REPO}/src:${REPO}"

source /mnt/scratch2/users/40482774/venvs/mcq-generalization/bin/activate

python -m experiments.flip_rate_traces.run_canary \
    --backend local --cell cbp__qwen-qwen2-5-7b-instruct__arc_challenge__cyclic_generation_majority \
    --model-name Qwen/Qwen2.5-7B-Instruct --benchmark arc_challenge --question-kind normal --device cuda

python -m experiments.flip_rate_traces.run_canary \
    --backend local --cell cbp__qwen-qwen2-5-7b-instruct__arc_challenge__cyclic_generation_majority \
    --model-name Qwen/Qwen2.5-7B-Instruct --benchmark arc_challenge --question-kind damaged_arc --device cuda
