#!/usr/bin/env bash
# Generic job wrapper for the batched HF-FP8 local production run. Env: CB_CMD (command to run in the repo).
set -euo pipefail
export CHOICEBENCH_REPO="$HOME/choicebench-eacl-hfbatch"
export CHOICEBENCH_BASE="$HOME/choicebench_hpc_eacl"
export CHOICEBENCH_VENV="$CHOICEBENCH_BASE/.venv"
export HF_HOME="/mnt/scratch2/users/40482774/hf_cache"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
source "$CHOICEBENCH_REPO/examples/hpc/env_hpc.sh"
echo "Slurm job: ${SLURM_JOB_ID}  Git SHA: $(git -C "$CHOICEBENCH_REPO" rev-parse HEAD)"
nvidia-smi --query-gpu=name,memory.total --format=csv
echo "CMD: ${CB_CMD}"
bash -c "${CB_CMD}"
