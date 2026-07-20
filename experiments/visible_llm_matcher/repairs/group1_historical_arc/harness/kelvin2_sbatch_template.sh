#!/bin/bash
# Template used to generate each local Group1 repair cell's sbatch script.
# Not executed directly. Parameters substituted by the operator per cell:
#   {{CELL_ID}}, {{JOB_NAME}}
#
# Local Group1 repair cells (post-canary): only two_stage_v2/v3 for
# Qwen/Qwen2.5-7B-Instruct and meta-llama/Llama-3.1-8B-Instruct (4 cells
# total; two_stage_v2 for Qwen already run as the Section B canary).
#
#SBATCH --job-name={{JOB_NAME}}
#SBATCH --partition=k2-gpu-v100
#SBATCH --gres=gpu:v100:1
#SBATCH --time=00:30:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --output=/mnt/scratch2/users/40482774/repos/choicebench-visible-llm-matcher-cell/{{JOB_NAME}}_%j.log

set -euo pipefail

REPO=/mnt/scratch2/users/40482774/repos/choicebench-visible-llm-matcher-cell
VENV=/mnt/scratch2/users/40482774/venvs/mcq-generalization
export HF_HOME=/mnt/scratch2/users/40482774/hf
export PYTHONPATH="${REPO}/src:${REPO}"

cd "${REPO}"
echo "commit: $(git rev-parse HEAD)"
git status --short

"${VENV}/bin/python3" -m experiments.visible_llm_matcher.repairs.group1_historical_arc.harness.run_repair \
    --cell-id {{CELL_ID}}
