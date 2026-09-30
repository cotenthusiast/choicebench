#!/usr/bin/env bash
# Submit the batched HF-FP8 local production (12 jobs) + stochasticity chain (16 jobs, afterok on the
# matching production job). Run on Kelvin2 from the repo root after sourcing env_hpc.sh.
# DRY=1 prints every sbatch command without submitting.
set -euo pipefail
D=docs/reproducibility/eacl2026_local_hfbatch_run
REPO="$HOME/choicebench-eacl-hfbatch"
LOGS="$HOME/choicebench_hpc_eacl/tonight_logs"
COMMON=(--parsable --partition=k2-gpu-a100 --gres=gpu:a100:1 --time=3-00:00:00 --cpus-per-task=8 --mem=64G
        --output="$LOGS/%x_%j.out" --error="$LOGS/%x_%j.err")
sub() { if [[ "${DRY:-0}" == 1 ]]; then echo "sbatch $*" >&2; echo "DRYJOB"; else sbatch "$@"; fi; }
condid() { python - "$1" "$2" <<'PY'
import sys
from choicebench.config.schema import load_config
from choicebench.cli.run_experiment import build_execution_plan
cfg = load_config(sys.argv[1]); plan = build_execution_plan(cfg)
for (bi, mi, model_i), rec in plan.conditions.items():
    if cfg.methods[mi].name == sys.argv[2]:
        print(rec["condition_id"])
PY
}
for bench in mmlu arc; do
  for group in core_and_cyclic reasoning_and_cyclic independent_hypothesis; do
    for model in llama qwen; do
      cfg="$D/${bench}_${group}_${model}.yaml"; rid="eacl2026_local_b32_${bench}_${group}_${model}"
      pj=$(sub "${COMMON[@]}" --job-name=cb_b32 --export=ALL,CB_CMD="python scripts/run_experiment.py --config $cfg --run-id $rid --yes" "$D/hfbatch_job.sbatch")
      echo "prod $rid -> $pj"
      case $group in
        core_and_cyclic) cyc=cyclic_permutation; base=direct_mcq; two=two_stage; pv=v1 ;;
        reasoning_and_cyclic) cyc=reasoning_cyclic; base=reasoning_mcq; two=reasoning_two_stage; pv=v1_reasoning ;;
        *) continue ;;
      esac
      res="$REPO/runs/$rid/results"
      cyc_id=$(condid "$cfg" "$cyc"); two_id=$(condid "$cfg" "$two")
      der="$REPO/runs/eacl2026_local_b32_derived/${bench}_${base}_${model}.csv"
      stoch="--questions-csv data/manifests/${bench}_stochasticity_v2_full.csv --model-config $cfg --model-index 0 --prompt-version $pv --run-id eacl2026_local_b32_stochasticity --execution-mode sync"
      sub "${COMMON[@]}" --dependency=afterok:$pj --job-name=cb_b32_stoch --export=ALL,CB_CMD="python scripts/paper/derive_baseline_from_cyclic.py --source $res/$cyc_id.csv --output $der --method-name $base && python scripts/paper/run_stochasticity_repeats.py $stoch --canonical-obs0-csv $der --method-name $base --output $REPO/runs/eacl2026_local_b32_stochasticity/${bench}_${base}_${model}.csv" "$D/hfbatch_job.sbatch" >/dev/null
      sub "${COMMON[@]}" --dependency=afterok:$pj --job-name=cb_b32_stoch --export=ALL,CB_CMD="python scripts/paper/run_stochasticity_repeats.py $stoch --canonical-obs0-csv $res/$two_id.csv --method-name $two --output $REPO/runs/eacl2026_local_b32_stochasticity/${bench}_${two}_${model}.csv" "$D/hfbatch_job.sbatch" >/dev/null
    done
  done
done
