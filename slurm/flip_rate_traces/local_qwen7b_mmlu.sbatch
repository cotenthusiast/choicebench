#!/bin/bash
#SBATCH --job-name=flip_rate_qwen7b_mmlu
#SBATCH --partition=k2-gpu-a100
#SBATCH --gres=gpu:a100:1
#SBATCH --time=08:00:00
#SBATCH --mem=48G
#SBATCH --output=flip_rate_qwen7b_mmlu_%j.log
#
# Flip-rate traces: Qwen/Qwen2.5-7B-Instruct x mmlu, cyclic-generation-majority.
# See local_qwen7b_arc_challenge.sbatch for partition/pin rationale.

set -euo pipefail

REPO="/mnt/scratch2/users/40482774/repos/choicebench-flip-rate-traces-mmlu"
PINNED_SHA="982a3c54db6ef8b33afefea607c06cdff3f2d5ae"

cd "${REPO}"
git fetch origin
git checkout --detach "${PINNED_SHA}"

export HF_HOME="/mnt/scratch2/users/40482774/hf"
export FLIP_RATE_TRACES_FREEZE_ROOT="/mnt/scratch2/users/40482774/paper_data_freeze_subset"
export HF_HUB_CACHE="${HF_HOME}/hub"
export PYTHONPATH="${REPO}/src:${REPO}"

source /mnt/scratch2/users/40482774/venvs/mcq-generalization/bin/activate

python -m experiments.flip_rate_traces.run_cell \
    --config configs/flip_rate_traces/local_qwen7b_mmlu.yaml
