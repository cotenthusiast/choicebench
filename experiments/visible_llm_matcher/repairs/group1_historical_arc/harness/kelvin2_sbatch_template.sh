#!/bin/bash
# Template used to generate each local Group1 repair cell's sbatch script.
# Not executed directly. Parameters substituted by the operator per cell:
#   {{CELL_ID}}, {{JOB_NAME}}
#
# Local Group1 repair cells (post-canary): only two_stage_v2/v3 for
# Qwen/Qwen2.5-7B-Instruct and meta-llama/Llama-3.1-8B-Instruct (4 cells
# total; two_stage_v2 for Qwen already run as the Section B canary).
#
# Partition MUST be k2-gpu-a100 (or another CC>=7.5 GPU), never
# k2-gpu-v100: discovered live via the Section B canary
# (cbp__qwen-qwen2-5-7b-instruct__arc_challenge__two_stage_v2, job
# 9399739) that the shared mcq-generalization venv's torch build
# (2.12.0+cu130) has no compiled kernels for V100's compute capability
# 7.0 ("CUDA error: no kernel image is available for execution on the
# device"), silently producing failure responses. Re-ran on k2-gpu-a100
# (job 9399740) with identical code and it passed cleanly.
#
#SBATCH --job-name={{JOB_NAME}}
#SBATCH --partition=k2-gpu-a100
#SBATCH --gres=gpu:a100:1
#SBATCH --time=00:30:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --output=/mnt/scratch2/users/40482774/repos/choicebench-visible-llm-matcher-cell/{{JOB_NAME}}_%j.log

set -euo pipefail

REPO=/mnt/scratch2/users/40482774/repos/choicebench-visible-llm-matcher-cell
VENV=/mnt/scratch2/users/40482774/venvs/mcq-generalization
export HF_HOME=/mnt/scratch2/users/40482774/hf
export VLM_FREEZE_ROOT=/mnt/scratch2/users/40482774/vlm_freeze
export VLM_SOURCE_DATA_ROOT=/mnt/scratch2/users/40482774/vlm_source_data
export PYTHONPATH="${REPO}/src:${REPO}"

cd "${REPO}"
echo "commit: $(git rev-parse HEAD)"
git status --short

"${VENV}/bin/python3" -m experiments.visible_llm_matcher.repairs.group1_historical_arc.harness.run_repair \
    --cell-id {{CELL_ID}}
