# Group 1 — historical ARC inference repairs (28 cells, 84 calls)

Repairs exactly the 3 known 3-option ARC questions
(`79e8c959bbeb74a0`, `ad6b5d46ae54842c`, `c30e75b011696a95`) across the 28
malformed cells identified as `approved`+`executable` in
`model-generalization/paper_data_freeze/manifests/approved_rerun_queue.csv`,
filtered to `benchmark == arc_challenge` and the 6 approved methods
(`baseline`, `cyclic_generation_majority`, `text_extraction`,
`two_stage_v1`, `two_stage_v2`, `two_stage_v3`) — see
`build_group1_artifacts.py::build_and_verify_scope()`, which re-derives and
asserts this count (28) every run rather than trusting a hardcoded list.
Excludes (verified present in the source data but explicitly out of
scope): 1 `independent_hypothesis` (gemini), 1 `pride` (Qwen-Turbo), 2 `mmlu`
rows (wrong benchmark).

24 API cells (4 models × 6 methods) + 4 local/Kelvin2 cells
(`Qwen/Qwen2.5-7B-Instruct` and `meta-llama/Llama-3.1-8B-Instruct`, each ×
`two_stage_v2`/`two_stage_v3` only — `two_stage_v1` is not in the repair
scope for local models because MG's own runner already omits a missing
option rather than rendering `D. nan`; only the API path has this defect at
Stage 2/Stage 1-visible rendering).

## How the repair works

Each cell already has 997 clean rows and 3 contaminated ones (the model was
shown a phantom `D. nan` option). Rather than regenerating all 1000 rows,
`build_group1_artifacts.py` uses the SAME resume/checkpoint mechanism
`two-stage-prompting`'s and `model-generalization`'s own
`scripts/run_experiment.py` already have (`--run-id <existing-run-id>`,
which loads a checkpoint and skips any `question_id` in
`completed_ids`):

1. Read the historical run's own frozen `config.yaml` snapshot (from
   `source_configuration` in the rerun spec) and trim `run.jobs` to exactly
   one job — this one (model, method, benchmark). Every other field
   (`temperature`, `max_tokens`, `seed`, `prompt_version`, provider
   settings) is copied byte-identical from the snapshot, not
   retyped — see `configs/<cell_id>.yaml`.
2. Read the historical run's CSV, split into 997 "keep" rows (full row data
   preserved) and 3 "repair" rows (the contaminated ones — dropped
   entirely, not fed forward). Write a checkpoint JSON with
   `completed_ids` = the 997 keep IDs and `results` = their full row data
   — see `checkpoint_seeds/<run_id>/<job_method>__<model>__<benchmark>.json`.
   This checkpoint schema is verified byte-identical between TSP's and
   MG's `infra/checkpoint.py`.
3. `launch_repair_cell.sh <cell_id>` (not run in this phase) copies the
   seeded checkpoint into the target repo's real `checkpoints/<run_id>/`
   directory, then prints (does not itself execute) the
   `run_experiment.py --config ... --run-id ... --yes` invocation. On
   resume, `run_experiment.py` sees 997/1000 already completed and
   generates exactly the 3 remaining — producing a corrected, complete
   1000-row output under the SAME `run_id`.

Pinned commits for execution: `two-stage-prompting@b8e784f3eb5d2a727a97eb675140b383a34584fa`,
`model-generalization@24403d8da9c7c11189bf56439b4040bc7cad1efd` (both
current `main` as of this config-generation pass — confirm unchanged before
executing).

## Local (Kelvin2) cells

The 4 local repair cells' checkpoints target
`/mnt/scratch2/users/40482774/repos/model-generalization/checkpoints/<run_id>/...`
on Kelvin2, which this machine cannot write to. `launch_repair_cell.sh`
detects `is_local` and refuses to run; the actual launch must happen from a
Kelvin2 login node using the same setup/checkout/copy-checkpoint sequence
as Group 3's local `sbatch` scripts (`../group3_fourth_cell/local/slurm/`) —
copy the relevant `checkpoint_seeds/<run_id>/*.json` and
`configs/<cell_id>.yaml` from this ChoiceBench worktree (checked out on
Kelvin2 at the pinned SHA) into place, then run MG's
`scripts/run_experiment.py` the same way.

## Semantic-matching offline repair (18 rows, 6 cells — no inference)

6 ARC `semantic_matching_v1` cells (4 API + 2 local) derive purely from
`two_prompt`/`two_stage_v1`'s Stage-1 free-text answers, which are
**unaffected** by the phantom-D defect — Stage 1 for this condition never
shows options at all (`free_text.txt` has no option placeholders), so
nothing about it needed repairing. Confirmed directly against
`model-generalization/paper_data_freeze/manifests/canonical_results_manifest.json`:
all 6 cells carry `"final_status": "recoverable_from_existing_artifacts"` and
`"qualification": "option-hidden Stage-1 outputs are preserved; affected
semantic matches can be reconstructed offline without new inference"`.

Once each cell's underlying `two_prompt`/`two_stage_v1` CSV is repaired
(API: this group's `two_stage_v1` cells above; local: unaffected, already
clean — local `two_prompt` never needed a Group 1 repair), the corrected
`semantic_matching_v1` rows for these 3 questions are produced by
re-running the existing deterministic script — no model calls:

```bash
# API (per model, after that model's two_stage_v1 repair has landed):
cd /home/cotenthusiast/Projects/two-stage-prompting
python scripts/semantic_matching.py --run-id 20260601_183346 --benchmark arc_challenge

# Local (source: run_id 20260529_145812, two_prompt, unaffected/already clean):
# uses the same deterministic matcher logic as
# two-stage-prompting/scripts/semantic_matching.py + src/twoprompt/parsing/text_matcher.py
# (an archived copy is preserved at
# paper_data_freeze/raw/local_two_stage_prompting/scripts/semantic_matching.py
# for reference/provenance; do not execute paper_data_freeze contents directly —
# copy the underlying logic or point a working copy of the script at
# model-generalization's own two_prompt CSV, run_id 20260529_145812)
```

This produces all 18 recovered rows (6 cells × 3 questions) with zero new
inference calls, consistent with the manifest's own classification.
