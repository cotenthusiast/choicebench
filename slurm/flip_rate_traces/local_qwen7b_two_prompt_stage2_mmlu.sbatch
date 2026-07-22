#!/bin/bash
#SBATCH --job-name=fr_tp_stage2_qwen7b_mmlu
#SBATCH --partition=k2-gpu-a100
#SBATCH --gres=gpu:a100:1
#SBATCH --time=08:00:00
#SBATCH --mem=48G
#SBATCH --output=fr_tp_stage2_qwen7b_mmlu_%j.log
#
# Task1a: two_prompt real Stage-2 LLM-matching traces, Qwen/Qwen2.5-7B-Instruct x mmlu.
# Stage 1 free-text reused from historical run 20260529_145812 (no new Stage-1 calls).

set -euo pipefail

REPO="/mnt/scratch2/users/40482774/repos/choicebench-flip-rate-traces-task1a"
MODELGEN_REPO="/mnt/scratch2/users/40482774/repos/model-generalization"
PINNED_SHA="5ca0095"

cd "${REPO}"
git fetch origin
git checkout --detach "${PINNED_SHA}"

export HF_HOME="/mnt/scratch2/users/40482774/hf"
export HF_HUB_CACHE="${HF_HOME}/hub"
export CHOICEBENCH_FLIP_RATE_REPO="${REPO}"
export MODELGEN_REPO="${MODELGEN_REPO}"
export PYTHONPATH="${REPO}/src:${REPO}"

source /mnt/scratch2/users/40482774/venvs/mcq-generalization/bin/activate

python -m experiments.flip_rate_traces.task1a.gen_two_prompt_stage2_local --benchmark mmlu
