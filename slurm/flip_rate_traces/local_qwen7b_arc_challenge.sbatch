#!/bin/bash
#SBATCH --job-name=flip_rate_qwen7b_arc
#SBATCH --partition=k2-gpu-a100
#SBATCH --gres=gpu:a100:1
#SBATCH --time=08:00:00
#SBATCH --mem=48G
#SBATCH --output=flip_rate_qwen7b_arc_%j.log
#
# Flip-rate traces: Qwen/Qwen2.5-7B-Instruct x arc_challenge, cyclic-generation-majority.
# Partition verified live via `sinfo -o "%P %D %G %m %l"` on Kelvin2
# (k2-gpu-a100: 3 nodes, gpu:a100:4, 3-day limit). Avoids k2-gpu-v100 —
# ChoiceBench v0.2.0 Audit / prior session found V100's compute capability
# 7.0 has no torch kernel support for the installed stack.
#
# PINNED COMMIT — clean detached checkout, not a branch tip, so this job
# runs exactly the runner/config code this experiment produced:
#   git -C /mnt/scratch2/users/40482774/repos/choicebench-flip-rate-traces fetch origin
#   git -C /mnt/scratch2/users/40482774/repos/choicebench-flip-rate-traces checkout --detach <PINNED_SHA>
#
# Clone first if the directory does not exist:
#   git clone git@github.com:cotenthusiast/choicebench.git /mnt/scratch2/users/40482774/repos/choicebench-flip-rate-traces

set -euo pipefail

REPO="/mnt/scratch2/users/40482774/repos/choicebench-flip-rate-traces"
PINNED_SHA="38e4daf3caaa4f256cbaa89b5b7264e85ef0bb99"

cd "${REPO}"
git fetch origin
git checkout --detach "${PINNED_SHA}"

export HF_HOME="/mnt/scratch2/users/40482774/hf"
export HF_HUB_CACHE="${HF_HOME}/hub"
export PYTHONPATH="${REPO}/src:${REPO}"

source /mnt/scratch2/users/40482774/venvs/mcq-generalization/bin/activate

python -m experiments.flip_rate_traces.run_cell \
    --config configs/flip_rate_traces/local_qwen7b_arc_challenge.yaml
