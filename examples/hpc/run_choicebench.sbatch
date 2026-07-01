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

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

mkdir -p "$REPO_ROOT/logs"

# shellcheck source=examples/hpc/env_hpc.sh
source "$SCRIPT_DIR/env_hpc.sh"

echo "Slurm job: ${SLURM_JOB_ID:-manual}"
echo "Node: ${SLURMD_NODENAME:-$(hostname)}"
echo "Config: ${CHOICEBENCH_CONFIG:-config/toy_experiment.yaml}"
echo "Run ID: ${CHOICEBENCH_RUN_ID:-hpc_test_${SLURM_JOB_ID:-manual}}"

python scripts/run_experiment.py \
    --config "${CHOICEBENCH_CONFIG:-config/toy_experiment.yaml}" \
    --run-id "${CHOICEBENCH_RUN_ID:-hpc_test_${SLURM_JOB_ID:-manual}}" \
    --yes

python scripts/evaluate_run.py \
    --run-id "${CHOICEBENCH_RUN_ID:-hpc_test_${SLURM_JOB_ID:-manual}}"
