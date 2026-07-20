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

REPO="/mnt/scratch2/users/40482774/repos/choicebench-flip-rate-traces"
PINNED_SHA="1ee760862215cfc94c61cb96bca487d85481f848"

cd "${REPO}"
git fetch origin
git checkout --detach "${PINNED_SHA}"

export HF_HOME="/mnt/scratch2/users/40482774/hf"
export HF_HUB_CACHE="${HF_HOME}/hub"
export PYTHONPATH="${REPO}/src:${REPO}"

source /mnt/scratch2/users/40482774/venvs/mcq-generalization/bin/activate

python -m experiments.flip_rate_traces.run_cell \
    --config configs/flip_rate_traces/local_qwen7b_mmlu.yaml
