#!/usr/bin/env bash
# experiments/visible_llm_matcher/repairs/group1_historical_arc/launch_repair_cell.sh
#
# Launches ONE Group 1 historical-repair cell (3 targeted question
# inferences). Does nothing until you explicitly run it — this file is not
# invoked by anything else in this phase.
#
# Usage:
#   ./launch_repair_cell.sh <cell_id>
#
# Example:
#   ./launch_repair_cell.sh cbp__gpt-4-1-mini__arc_challenge__two_stage_v2
#
# What it does, in order:
#   1. Looks up <cell_id> in manifest_summary.json (written by
#      build_group1_artifacts.py).
#   2. For an API cell: copies the staged checkpoint seed into
#      two-stage-prompting/checkpoints/<run_id>/, then runs
#      `python scripts/run_experiment.py --config <scoped config> --run-id
#      <run_id> --yes` from the two-stage-prompting repo root, at its
#      pinned commit (see README.md for the exact SHA).
#   3. For a local (Kelvin2) cell: refuses to run directly (this machine
#      has no GPU / Kelvin2 filesystem access) and instead prints the exact
#      SLURM submission this cell needs — see slurm/ for the actual sbatch
#      script, which must be run FROM a Kelvin2 login node.
#
# Idempotent: rerunning after a partial failure resumes from whatever the
# (real, non-seeded) checkpoint at that point says is completed — this is
# run_experiment.py's own existing checkpoint/resume behavior, not
# something this script adds.

set -euo pipefail

CELL_ID="${1:?Usage: launch_repair_cell.sh <cell_id>}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MANIFEST="${HERE}/manifest_summary.json"

if [[ ! -f "${MANIFEST}" ]]; then
  echo "error: ${MANIFEST} not found — run build_group1_artifacts.py first." >&2
  exit 1
fi

CELL_JSON="$(python3 -c "
import json, sys
manifest = json.load(open('${MANIFEST}'))
matches = [c for c in manifest if c['cell_id'] == '${CELL_ID}']
if not matches:
    sys.exit(f'error: cell_id ${CELL_ID} not found in manifest')
print(json.dumps(matches[0]))
")"

IS_LOCAL="$(echo "${CELL_JSON}" | python3 -c "import json,sys; print(json.load(sys.stdin)['is_local'])")"
RUN_ID="$(echo "${CELL_JSON}" | python3 -c "import json,sys; print(json.load(sys.stdin)['run_id'])")"
CONFIG_REL="$(echo "${CELL_JSON}" | python3 -c "import json,sys; print(json.load(sys.stdin)['generated_config'])")"
CHECKPOINT_REL="$(echo "${CELL_JSON}" | python3 -c "import json,sys; print(json.load(sys.stdin)['generated_checkpoint_seed'])")"
TARGET_CHECKPOINT_REL="$(echo "${CELL_JSON}" | python3 -c "import json,sys; print(json.load(sys.stdin)['target_checkpoint_path_relative_to_repo_root'])")"

CHOICEBENCH_WORKTREE_ROOT="$(cd "${HERE}/../../../.." && pwd)"

if [[ "${IS_LOCAL}" == "True" ]]; then
  echo "cell_id ${CELL_ID} runs on Kelvin2 (local HuggingFace model)."
  echo "This machine cannot execute it. Submit via SLURM from a Kelvin2 login node:"
  echo
  echo "  See: ${HERE}/../group3_fourth_cell/local/slurm/ for the sbatch pattern"
  echo "  (Group 1's local repair cells use the same setup/checkout/copy-checkpoint"
  echo "  steps as Group 3's local sbatch scripts — see README.md in this directory"
  echo "  for the exact per-cell sbatch invocation)."
  exit 0
fi

TSP_ROOT="/home/cotenthusiast/Projects/two-stage-prompting"
SRC_CHECKPOINT="${CHOICEBENCH_WORKTREE_ROOT}/${CHECKPOINT_REL}"
DST_CHECKPOINT="${TSP_ROOT}/${TARGET_CHECKPOINT_REL}"
CONFIG_PATH="${CHOICEBENCH_WORKTREE_ROOT}/${CONFIG_REL}"

echo "cell_id:            ${CELL_ID}"
echo "run_id:              ${RUN_ID}"
echo "config:               ${CONFIG_PATH}"
echo "checkpoint seed:      ${SRC_CHECKPOINT}"
echo "checkpoint target:    ${DST_CHECKPOINT}"
echo

mkdir -p "$(dirname "${DST_CHECKPOINT}")"
cp -n "${SRC_CHECKPOINT}" "${DST_CHECKPOINT}" || {
  echo "error: ${DST_CHECKPOINT} already exists — refusing to overwrite an" >&2
  echo "existing checkpoint (it may hold real progress). Investigate before" >&2
  echo "removing it manually." >&2
  exit 1
}

echo "Checkpoint staged. To launch (NOT done automatically by this script):"
echo
echo "  cd ${TSP_ROOT}"
echo "  git checkout <pinned TSP SHA — see README.md>"
echo "  python scripts/run_experiment.py --config ${CONFIG_PATH} --run-id ${RUN_ID} --yes"
