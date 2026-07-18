#!/usr/bin/env bash
# Slurm template for the PriDe single-cell reproduction (MMLU / llama-13B /
# 0-shot / direct_logprob + cyclic_logprob only — see
# examples/pride_reproduction.yaml for the full design rationale).
#
# VRAM sizing: huggyllama/llama-13b in fp16 is ~26GB of weights alone;
# activations for a single forward pass (no generation — both methods here
# are score_options()-only) add a modest amount on top. A single 40GB or
# 80GB A100 is sufficient; this does not need multi-GPU device_map=auto.
#
# Cost shape: direct_logprob is 1 forward pass/question; cyclic_logprob is
# N forward passes/question (N=4 for MMLU's 4 options). Over the ~13,592
# permutation-filtered MMLU test questions that is ~5x13,592 ≈ 68,000 total
# forward passes — the --time budget below is a generous upper bound for a
# single A100, not a tuned estimate.

#SBATCH --job-name=pride_repro
#SBATCH --output=logs/pride_repro_%j.out
#SBATCH --error=logs/pride_repro_%j.err
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
##SBATCH --partition=gpu

set -euo pipefail

# BASH_SOURCE breaks here: sbatch copies the submitted script to a spool
# directory before executing it, so BASH_SOURCE resolves to the spool path,
# not this file's real location. SLURM_SUBMIT_DIR is the directory sbatch
# was invoked from and stays stable across that copy.
REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
SCRIPT_DIR="$REPO_ROOT/examples/hpc"

mkdir -p "$REPO_ROOT/logs"

# shellcheck source=examples/hpc/env_hpc.sh
source "$SCRIPT_DIR/env_hpc.sh"

CONFIG="${CHOICEBENCH_CONFIG:-examples/pride_reproduction.yaml}"
RUN_ID="${CHOICEBENCH_RUN_ID:-pride_reproduction_${SLURM_JOB_ID:-manual}}"

echo "Slurm job: ${SLURM_JOB_ID:-manual}"
echo "Node: ${SLURMD_NODENAME:-$(hostname)}"
echo "Config: $CONFIG"
echo "Run ID: $RUN_ID"

# The runner resolves and verifies the exact split-addressed norm-v2 artifact.
# If absent, it prints the required preparation command. This job deliberately
# does not download data on a compute node.

python scripts/run_experiment.py \
    --config "$CONFIG" \
    --run-id "$RUN_ID" \
    --yes

python scripts/evaluate_run.py \
    --run-id "$RUN_ID"
