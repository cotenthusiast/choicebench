# Group 3 — the fourth cell: 12 (model x benchmark) runs

6 models × 2 benchmarks = 12 cells: 8 API (`api/configs/`), 4 local/Kelvin2
(`local/configs/` + `local/slurm/`).

## Driver: `run_fourth_cell.py`

Not routed through `choicebench`'s own `scripts/run_experiment.py` CLI —
that CLI feeds question rows from ChoiceBench's own prepared benchmark
artifact, whose `question_id` is independently re-hashed and does not
reliably match the immutable Stage 1 freeze this experiment must stay
authoritative to (see `runner.py`'s module docstring). Instead:

1. Loads the cell's YAML config.
2. `stage1_sources.load_and_validate_stage1()` — fails closed on missing
   rows, wrong row count, or (as of the phantom-D prompt-text check added
   during this phase) contamination that a naive NaN-`choice_d` check alone
   would miss.
3. For `arc_challenge`: `arc_freeze_validation.validate_against_freeze()` —
   re-verifies the resulting 1000 rows against the checksummed immutable
   freeze before anything runs.
4. Builds the ChoiceBench backend via `choicebench.cli.run_experiment.build_backend()`
   (reused, not reimplemented) and constructs `VisibleLlmMatcherRunner`
   directly — not via `METHOD_REGISTRY` / `"module.path:ClassName"`
   resolution, since this driver isn't going through `instantiate_runner()`.
5. Iterates the 1000 Stage-1 rows, checkpointing via ChoiceBench's own
   `CheckpointManager` (legacy calling convention — condition/model/benchmark,
   no experiment/condition/selection ids) every 50 questions.
6. Writes the final 1000-row CSV to `output_csv`.

## A real bug found and fixed while wiring this up

`stage1_sources.py` assumed every historical `text_extraction` CSV has a
`free_text_response` column. TSP's own **API** CSVs don't — the free-text
answer is under `raw_text` only (MG's local CSVs happen to have both).
`_load_csv()` / the new `load_replacement_rows()` now derive
`free_text_response` from `raw_text` when the column is absent. Caught by
smoke-testing `prepare_stage1()` against the real MMLU config before
assuming it worked — see `tests/experiments/test_stage1_sources.py`.

A second gap, same investigation: `choice_d` being NaN can't distinguish
"already repaired" from "still contaminated" when a replacement source
points at the same file as the primary source (true for the API case,
since Group 1's repair updates the run's CSV in place). Fixed by checking
the stored `prompt` text directly for the literal `"D. nan"` contamination
signature — see `stage1_sources.py`'s new check and its 2 regression tests.
Confirmed against real data: `api/configs/gpt-4.1-mini__arc_challenge.yaml`
correctly fails closed right now (Group 1's repair hasn't run yet);
`local/configs/meta-llama_Llama-3.1-8B-Instruct__arc_challenge.yaml`
correctly validates (MG's data was already clean).

## API vs local separation

API configs (`api/configs/*.yaml`) run outside Kelvin2 — direct
`python run_fourth_cell.py --config ...` invocation from this ChoiceBench
worktree, using `openai`/`gemini`/`groq`/`together` clients. Local configs
(`local/configs/*.yaml`) target the HuggingFace backend and are launched
only via `local/slurm/*.sbatch` on Kelvin2 — this machine has no GPU.

Local ARC configs reuse `model-generalization/runs/20260617_162624/` as
their own `stage1_replacement_source` (self-replacement — see
`../group2_stage1_repair/README.md`; already verified valid, not
contaminated per the phantom-D prompt-text check above). API ARC configs'
`stage1_replacement_source` points at the same run/file Group 1's
`text_extraction` repair updates in place — these configs will correctly
refuse to run (`Stage1ValidationError`) until that repair has actually
landed, which is by design.

## SLURM

`local/slurm/generate_sbatch.py` produces the 4 `.sbatch` files. Each
checks out this ChoiceBench worktree at a **pinned, detached-HEAD commit**
on Kelvin2 before running — placeholder `PINNED_SHA_PLACEHOLDER_SEE_REPORT_MD`
is replaced with the real final SHA once this config-generation phase is
committed (see `../../REPORT.md`). Partition/GRES names (`gpu`, `gpu:1`)
are unverified placeholders — confirm with
`sinfo -o "%P %D %G %m %l %N"` on an actual Kelvin2 login node before
submitting; this phase has no Kelvin2 access to check them.
