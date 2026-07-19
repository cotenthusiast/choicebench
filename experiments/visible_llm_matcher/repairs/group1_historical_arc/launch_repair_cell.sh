#!/usr/bin/env bash
# experiments/visible_llm_matcher/repairs/group1_historical_arc/launch_repair_cell.sh
#
# Launches ONE Group 1 historical-repair cell (3 targeted question
# repairs) via the harness (harness/run_repair.py) — NOT via
# two-stage-prompting's/model-generalization's own unmodified
# run_experiment.py, which cannot fix the phantom-D contamination (it just
# reproduces the identical broken prompt on resume; see
# INVALID_diagnostics/ for the evidence that led to this rewrite).
#
# Usage:
#   ./launch_repair_cell.sh <cell_id>
#
# Example:
#   ./launch_repair_cell.sh cbp__gpt-4-1-mini__arc_challenge__text_extraction
#
# What it does:
#   - API cells: exports nothing itself — you must have already exported
#     the relevant provider API key into THIS shell's environment (e.g.
#     `set -a; source /home/cotenthusiast/Projects/two-stage-prompting/.env;
#     set +a`) — then runs
#     `python -m experiments.visible_llm_matcher.repairs.group1_historical_arc.harness.run_repair
#     --cell-id <cell_id>` from this ChoiceBench worktree. That module:
#       1. loads manifest_summary.json's 997 seeded "keep" rows (unchanged,
#          reused, not regenerated — the validated checkpoint-seed scope
#          from the earlier phase);
#       2. fetches the 3 repaired questions' content from the checksummed
#          immutable freeze;
#       3. runs the method-appropriate repair (baseline/cyclic/text_extraction/
#          two_stage_v1/v2/v3) through the harness, using a dedicated,
#          per-(method,model) cache namespace that starts empty every time;
#       4. verifies the cache grew by exactly the expected number of fresh
#          calls (see repair_infra.assert_cache_grew_by) — refuses to
#          proceed otherwise;
#       5. writes a NEW 1000-row staged CSV under staged_repairs/<cell_id>/
#          — the historical CSV under two-stage-prompting/runs/ and the
#          immutable freeze are never written to by this path.
#   - Local (Kelvin2) cells: refuses to run directly (this machine has no
#     GPU) — see group3_fourth_cell/local/slurm/ for the sbatch pattern;
#     local Group 1 repairs are not yet wired to a Kelvin2 entry point in
#     this phase (harness/run_repair.py raises NotImplementedError for
#     is_local cells by design, rather than silently doing nothing useful).
#
# This script itself makes no network calls — it only validates inputs and
# prints the command to run. Money is spent only when you actually execute
# the printed `python -m ...` invocation.

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
METHOD="$(echo "${CELL_JSON}" | python3 -c "import json,sys; print(json.load(sys.stdin)['method'])")"
MODEL="$(echo "${CELL_JSON}" | python3 -c "import json,sys; print(json.load(sys.stdin)['model'])")"

CHOICEBENCH_WORKTREE_ROOT="$(cd "${HERE}/../../../.." && pwd)"

echo "cell_id:  ${CELL_ID}"
echo "method:   ${METHOD}"
echo "model:    ${MODEL}"
echo "is_local: ${IS_LOCAL}"
echo

if [[ "${IS_LOCAL}" == "True" ]]; then
  echo "This is a Kelvin2 cell. This machine cannot execute it directly."
  echo "See group3_fourth_cell/local/slurm/ for the sbatch pattern; local"
  echo "Group 1 repairs are not yet wired to a Kelvin2 entry point in this phase."
  exit 0
fi

echo "To launch (NOT done automatically by this script — this makes a real"
echo "API call once you run it):"
echo
echo "  cd ${CHOICEBENCH_WORKTREE_ROOT}"
echo "  set -a; source /home/cotenthusiast/Projects/two-stage-prompting/.env; set +a"
echo "  python -m experiments.visible_llm_matcher.repairs.group1_historical_arc.harness.run_repair --cell-id ${CELL_ID}"
