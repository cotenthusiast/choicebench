#!/bin/bash
#SBATCH --job-name=vlm_Qwen_Qwen2.5-7B-Instruct_mmlu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --mem=48G
#SBATCH --output=vlm_Qwen_Qwen2.5-7B-Instruct_mmlu_%j.log
#
# Group 3 (fourth cell, options-visible + LLM matcher): Qwen/Qwen2.5-7B-Instruct x mmlu.
# Verify partition/GRES names with `sinfo -o "%P %D %G %m %l %N"` before
# submitting -- gpu / gpu:1 above are placeholders, not verified against
# live Kelvin2 partition names (this config-generation phase has no Kelvin2
# access).
#
# PINNED COMMIT — checked out in detached-head mode, not on a branch tip,
# so this job runs exactly the config/runner code this repair phase
# produced, not whatever main has moved to since:
#   git -C /mnt/scratch2/users/40482774/repos/choicebench-visible-llm-matcher-cell fetch origin
#   git -C /mnt/scratch2/users/40482774/repos/choicebench-visible-llm-matcher-cell checkout --detach a2955b7338058b94a83326ccdabeb0dbade2dcb0
#
# This script assumes /mnt/scratch2/users/40482774/repos/choicebench-visible-llm-matcher-cell is already a checkout of the
# choicebench repo (branch worktree-visible-llm-matcher-cell) on this
# Kelvin2 filesystem -- clone it first if it is not:
#   git clone git@github.com:cotenthusiast/choicebench.git /mnt/scratch2/users/40482774/repos/choicebench-visible-llm-matcher-cell

set -euo pipefail

CHOICEBENCH_ROOT="/mnt/scratch2/users/40482774/repos/choicebench-visible-llm-matcher-cell"
MG_ROOT="/mnt/scratch2/users/40482774/repos/model-generalization"

cd "${CHOICEBENCH_ROOT}"
git fetch origin
git checkout --detach a2955b7338058b94a83326ccdabeb0dbade2dcb0

source env_kelvin2.sh 2>/dev/null || true  # if this worktree carries one; else set HF_HOME/PYTHONPATH manually

python experiments/visible_llm_matcher/repairs/group3_fourth_cell/run_fourth_cell.py \
    --config experiments/visible_llm_matcher/repairs/group3_fourth_cell/local/configs/Qwen_Qwen2.5-7B-Instruct__mmlu.yaml
