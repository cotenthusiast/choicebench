#!/usr/bin/env bash
# Minimal Slurm template for running the bundled toy ChoiceBench experiment.

#SBATCH --job-name=choicebench
#SBATCH --output=logs/choicebench_%j.out
#SBATCH --error=logs/choicebench_%j.err
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
##SBATCH --partition=compute
##SBATCH --gres=gpu:1

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

echo "Slurm job: ${SLURM_JOB_ID:-manual}"
echo "Node: ${SLURMD_NODENAME:-$(hostname)}"
echo "Config: ${CHOICEBENCH_CONFIG:-config/toy_experiment.yaml}"
echo "Run ID: ${CHOICEBENCH_RUN_ID:-hpc_test_${SLURM_JOB_ID:-manual}}"

# Idempotent and required on a cold clone: the toy dataset is generated in the
# configured ChoiceBench workspace, never assumed to exist in the source tree.
python scripts/prepare_toy_data.py

# Requeue note: the default toy config sets run.resume: false, so a requeued
# job reusing the same run ID is refused (exit 1) with instructions instead of
# touching the existing run. Real experiments should set run.resume: true (as
# config/experiment_template.yaml does) so requeues verify and continue.

python scripts/run_experiment.py \
    --config "${CHOICEBENCH_CONFIG:-config/toy_experiment.yaml}" \
    --run-id "${CHOICEBENCH_RUN_ID:-hpc_test_${SLURM_JOB_ID:-manual}}" \
    --yes

python scripts/evaluate_run.py \
    --run-id "${CHOICEBENCH_RUN_ID:-hpc_test_${SLURM_JOB_ID:-manual}}"
