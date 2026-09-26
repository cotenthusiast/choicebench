
## 2026-09-26 run log

- git SHA: f2a9e666447a6f23befa7ff13314505cfa84e525 (paper/eacl-2026-revision)
- Scope: OpenAI (gpt-4.1-mini-2025-04-14) + Anthropic (claude-haiku-4-5-20251001) only.
  Together/DeepInfra/OpenRouter excluded (billing-blocked/out-of-scope). Local HF models
  deferred to H100 job 10028778 (separate path).
- Subset configs generated at ~/choicebench_hpc_eacl/tonight_configs/ from the frozen
  config/paper/*.yaml originals (unmodified in the repo). Programmatic verification
  performed: for each of the 10 subset configs, confirmed (a) every non-\"models\" field
  is dict-equal to the original, (b) every retained model entry is a byte-identical
  member of the original models list, (c) retained providers == {openai, anthropic}
  exactly, (d) dropped entries are all openrouter/deepinfra/together/huggingface.
  Configs: mmlu_core_methods.yaml, mmlu_independent_hypothesis.yaml, mmlu_reasoning.yaml,
  mmlu_reasoning_batch.yaml, mmlu_core_methods_cyclic_batch.yaml, arc_core_methods.yaml,
  arc_independent_hypothesis.yaml, arc_reasoning.yaml, arc_reasoning_batch.yaml,
  arc_core_methods_cyclic_batch.yaml.
- Auth/inference smoke (sync path): both providers PASSED (real generate() call,
  PONG received) after fixing an anthropic SDK pin bug (see commit f2a9e66).
- Batch API smoke: in progress, see below.

- Batch API smoke test PASSED for both providers (2-prompt real batch via mmlu_core_methods_cyclic_batch.yaml models):
  openai[0]=PONG openai[1]=PING anthropic[0]=PONG anthropic[1]=PING, all status=success, no errors, correct order preserved.

- Launched 4 sync-mode sbatch jobs (k2-lowpri, 8h limit each):
  10028800 mmlu_core_methods (two_stage+text_extraction) run-id=eacl2026_oa_mmlu_core_methods
    frozen config: config/paper/mmlu_core_methods.yaml -> subset: tonight_configs/mmlu_core_methods.yaml
    output: runs/eacl2026_oa_mmlu_core_methods/
  10028801 arc_core_methods (two_stage+text_extraction) run-id=eacl2026_oa_arc_core_methods
    frozen config: config/paper/arc_core_methods.yaml -> subset: tonight_configs/arc_core_methods.yaml
    output: runs/eacl2026_oa_arc_core_methods/
  10028802 mmlu_reasoning (reasoning_two_stage) run-id=eacl2026_oa_mmlu_reasoning
    frozen config: config/paper/mmlu_reasoning.yaml -> subset: tonight_configs/mmlu_reasoning.yaml
    output: runs/eacl2026_oa_mmlu_reasoning/
  10028803 arc_reasoning (reasoning_two_stage) run-id=eacl2026_oa_arc_reasoning
    frozen config: config/paper/arc_reasoning.yaml -> subset: tonight_configs/arc_reasoning.yaml
    output: runs/eacl2026_oa_arc_reasoning/
  All queued PD (Priority) at submission, k2-lowpri. git SHA f2a9e666447a6f23befa7ff13314505cfa84e525.

- CORRECTION: k2-lowpri had 303 pending jobs ahead of mine and only 1 running --
  effectively could stall all night. Cancelled the 4 still-pending (zero work
  started, zero cost) jobs 10028800-10028803 and resubmitted on k2-medpri
  (11 running/176 pending at the time, 1-day time limit) instead:
  10028812 mmlu_core_methods, 10028813 arc_core_methods, 10028814 mmlu_reasoning,
  10028815 arc_reasoning. Same configs/run-ids as before, only partition changed.

- SECOND CORRECTION: k2-medpri showed zero movement across 3 checks (~15+ min),
  squeue --start reported N/A (no estimate). k2-hipri has far higher throughput
  (277 running vs medpris 11 at last check) but only a 3h wall-clock limit --
  acceptable given resume:true + checkpoint_every_n:50 in every frozen config
  (a timeout loses at most the last incomplete checkpoint batch, resumable via
  the same run-id later). Cancelled 10028812-10028815 (still pending, zero
  work/cost lost) and resubmitted on k2-hipri (time capped to 02:55:00):
  10028824 mmlu_core_methods, 10028825 arc_core_methods, 10028826 mmlu_reasoning,
  10028827 arc_reasoning. Same configs/run-ids, only partition+walltime changed.

- ROOT CAUSE + FIX: fresh Kelvin2 checkout never had prepare_data.py run (data/processed/ empty).
  Ran the documented prepare commands for mmlu and arc_challenge. VERIFIED artifact-identity
  integrity: resulting hashes matched exactly what the frozen manifests/crash logs expected
  (mmlu -> src_705fa6e8972d, arc_challenge -> src_fe8881c437c8) -- confirms the live HF dataset
  commits are unchanged from what the frozen question_id_manifests were built against. Not a
  protocol anomaly. ARC modal-k stats (1165/1172 = 99.4% k=4) match the expected 1165 four-option
  + 4 three-option + 3 five-option composition exactly.
- Relaunched the 4 sync jobs (same run-ids) on k2-hipri: 10028832 (mmlu_core_methods),
  10028833 (arc_core_methods), 10028834 (mmlu_reasoning), 10028835 (arc_reasoning).
  All confirmed RUNNING cleanly within seconds, past dataset loading, correct question
  counts (1140 mmlu / 1172 arc) and correct model identities (gpt-4.1-mini-2025-04-14,
  claude-haiku-4-5-20251001) in the logs.

- Launched 6 batch-mode conditions on k2-hipri (same resumable batch_state design
  means a 3h timeout mid-poll should reconnect to the in-flight provider batch on
  resubmit, not duplicate it):
  10028838 mmlu_independent_hypothesis (independent_hypothesis)
  10028839 arc_independent_hypothesis (independent_hypothesis)
  10028840 mmlu_reasoning_batch (reasoning_cyclic)
  10028841 arc_reasoning_batch (reasoning_cyclic)
  10028842 mmlu_cyclic (cyclic_permutation)
  10028843 arc_cyclic (cyclic_permutation)
  All 6 pending at submission (Priority), 4 sync jobs (10028832-10028835) confirmed
  still running clean at time of launch.

- First real output-content validation (2026-09-26 ~01:31): all 10 conditions RUNNING.
  mmlu_core_methods's anthropic/two_stage condition already complete: 1140/1140 unique
  question_ids, zero duplicates, answer_status success=1128/failure=12 (~1%, normal),
  parse_status mostly parse_ok with a few ambiguous/missing (normal model variance).
  arc_core_methods's anthropic/two_stage condition also complete: 1172/1172 unique,
  zero duplicates, success=1169/failure=3. Correct model_name/provider/method_name in
  every row. (Note: wc -l vastly overcounts these CSVs due to embedded newlines in
  quoted prompt/free-text fields -- use pandas row counts, not wc -l, for verification.)
  No anomalies. Spacing out checks to ~15-20 min per user's guidance now that content
  is confirmed healthy.

## Local (H100) production launch prep (2026-09-26, while API jobs run)

- Prepared MMLU validation split (1531 rows, matches frozen config comment exactly) and
  ARC-Challenge validation split (299 rows, modal-k=4 at 98.7% ~= 295 eligible, matches
  arc_pride.yamls stated 295 exactly) -- both needed for PriDe calibration, neither had
  been prepared before (only test splits were).
- Generated 10 local-only (RedHatAI Llama+Qwen backend=huggingface, dropping all api
  entries) subset configs at ~/choicebench_hpc_eacl/tonight_configs_local/, same
  programmatic-verification discipline as the API subset configs (only model-list
  entries differ from frozen originals; verified via load_config()). mmlu_pride.yaml
  and arc_pride.yaml need NO subsetting (already local-only in the frozen originals) --
  will use those two directly, unmodified.
- Confirmed execution_mode: batch is a documented no-op for huggingface-backend models
  (validate_batch_compatibility only checks backend==api), so the cyclic/reasoning_batch/
  independent_hypothesis local subset configs are safe to run as-is (just runs
  synchronously, as HF backend always does).
- New GPU sbatch template at ~/choicebench_hpc_eacl/tonight_run_local.sbatch
  (partition k2-gpu-h100, --gres=gpu:h100:1, 1-day time limit -- NOT the 3h interactive
  cap, appropriate for a much longer local-generation workload).
- NOT LAUNCHED YET: per the original gating instruction (smoke tests must pass before
  full local production), waiting on job 10028778 (still queued/ReqNodeNotAvail this
  entire session) to actually run and pass generate()/score_options() for both models
  before submitting anything from tonight_run_local.sbatch. Planned launch order once
  unblocked, respecting dependencies: cyclic_permutation and core_methods
  (two_stage+text_extraction) first (independent, and prerequisites for baseline/
  semantic_matching/visible_llm_matcher derivations), then independent_hypothesis and
  reasoning_* (independent), then PriDe last (needs nothing from the others). Will
  request 1 GPU per job so multiple conditions can run concurrently across the nodes 4
  H100s as slots free up.

- 2026-09-26 ~01:54: ALL 4 sync jobs COMPLETED cleanly (exit 0, ~20-22min each):
  10028832 mmlu_core_methods, 10028833 arc_core_methods, 10028834 mmlu_reasoning,
  10028835 arc_reasoning. Full validation: all 12 condition files (2 models x
  [two_stage,text_extraction] for core_methods, 2 models x reasoning_two_stage for
  reasoning, x2 benchmarks) have EXACTLY the expected row count (1140 mmlu / 1172 arc),
  zero duplicates, correct single method/model per file, healthy success rates
  (95-100%, e.g. gpt-4.1-mini/two_stage on arc=100%, claude-haiku/text_extraction on
  mmlu=97.1%). No anomalies.
  6 batch-mode jobs (10028838-10028843) still RUNNING healthy, steady rotation
  progress visible (e.g. cyclic_permutation 450/1140, independent_hypothesis
  450/1172, reasoning_cyclic 700/1172), clean 200 OK polling against both providers
  Batch APIs.
  H100 job 10028778 still PD/ReqNodeNotAvail -- unchanged, local production not
  triggered yet.

- 2026-09-26 ~02:15: all 6 batch-mode jobs still RUNNING (49min elapsed, well under 3h
  cap), progressing through successive rotation batches as expected (progress
  counters climb to N/N then reset for the next rotations batch -- normal for
  cyclic_permutation/reasoning_cyclic/independent_hypothesis, not a regression).
  H100 job 10028778 still PD/ReqNodeNotAvail, unchanged. No action needed.

## CONFIG INVENTORY (all configs used tonight, per user request to track + consolidate)

### API (OpenAI+Anthropic) subset configs -- USED, all 10 launched and completed:
Location: ~/choicebench_hpc_eacl/tonight_configs/
  mmlu_core_methods.yaml          <- config/paper/mmlu_core_methods.yaml            (run: eacl2026_oa_mmlu_core_methods)
  mmlu_independent_hypothesis.yaml <- config/paper/mmlu_independent_hypothesis.yaml  (run: eacl2026_oa_mmlu_independent_hypothesis)
  mmlu_reasoning.yaml              <- config/paper/mmlu_reasoning.yaml               (run: eacl2026_oa_mmlu_reasoning)
  mmlu_reasoning_batch.yaml        <- config/paper/mmlu_reasoning_batch.yaml         (run: eacl2026_oa_mmlu_reasoning_batch)
  mmlu_core_methods_cyclic_batch.yaml <- config/paper/mmlu_core_methods_cyclic_batch.yaml (run: eacl2026_oa_mmlu_cyclic)
  arc_core_methods.yaml            <- config/paper/arc_core_methods.yaml             (run: eacl2026_oa_arc_core_methods)
  arc_independent_hypothesis.yaml  <- config/paper/arc_independent_hypothesis.yaml   (run: eacl2026_oa_arc_independent_hypothesis)
  arc_reasoning.yaml               <- config/paper/arc_reasoning.yaml                (run: eacl2026_oa_arc_reasoning)
  arc_reasoning_batch.yaml         <- config/paper/arc_reasoning_batch.yaml          (run: eacl2026_oa_arc_reasoning_batch)
  arc_core_methods_cyclic_batch.yaml <- config/paper/arc_core_methods_cyclic_batch.yaml (run: eacl2026_oa_arc_cyclic)
These 10 could NOT be consolidated further: they were split from the frozen
originals for a reason unrelated to provider scope (different prompt_version
per method family: v1 core/cyclic, v1_reasoning reasoning/reasoning_batch,
v1_ihs independent_hypothesis -- prompt_version is a single run-level field,
methods needing different prompt bundles cannot share one config regardless
of provider subsetting).

### Local (RedHatAI H100/A100) subset configs -- BUILT, CONSOLIDATED before launch, not yet launched:
Original 10 at ~/choicebench_hpc_eacl/tonight_configs_local/ (1:1 mirror of the
API subset's 10, just local-only models) were CONSOLIDATED down to 6, since
(unlike the API case) nothing forced the core/cyclic and reasoning/reasoning_batch
pairs apart once Llama was dropped -- verified programmatically that each pair's
non-methods/models/experiment fields are identical, and execution_mode is a
per-method field so sync+batch methods coexist fine in one run_experiment.py call.
New merged configs at ~/choicebench_hpc_eacl/tonight_configs_local_merged/:
  mmlu_core_and_cyclic.yaml     (two_stage + text_extraction + cyclic_permutation)
  mmlu_reasoning_and_cyclic.yaml (reasoning_two_stage + reasoning_cyclic)
  arc_core_and_cyclic.yaml
  arc_reasoning_and_cyclic.yaml
Staying separate (different prompt_version, cannot merge):
  ~/choicebench_hpc_eacl/tonight_configs_local/mmlu_independent_hypothesis.yaml
  ~/choicebench_hpc_eacl/tonight_configs_local/arc_independent_hypothesis.yaml
Frozen originals, unmodified, used as-is (no subsetting needed, already local-only):
  config/paper/mmlu_pride.yaml
  config/paper/arc_pride.yaml
TOTAL local production launches planned: 6 (down from 10) + 2 frozen PriDe = 8,
down from the original 12.

## Llama (OpenRouter/DeepInfra) production launch (2026-09-26 ~11:11)

- User added OPENROUTER_API_KEY + DEEPINFRA_API_KEY ($10 credit each) to local .env,
  keys transferred securely to ~/.config/choicebench/.env same as before.
- Auth smoke PASSED both: openrouter(deepinfra-pinned, upstream_provider=deepinfra,
  allow_fallbacks=false) and direct deepinfra client, both real generate() calls
  returned PONG.
- Routing identity verified LIVE (not just trusting the frozen config comment):
  OpenRouter's endpoint listing for meta-llama/llama-3.1-8b-instruct reports
  DeepInfra route as tag=deepinfra/fp8, context=131072. DeepInfra's own live model
  catalog confirms meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo is quantization=fp8,
  context=131072, deprecated=None (the plain non-Turbo id is bfloat16 and DEPRECATED).
  Both routes hit the identical fp8/131072 deployment -- no anomaly.
- Batch API smoke PASSED for direct deepinfra (2-prompt real batch, correct order,
  no errors).
- Built 10 Llama-only exact-subset configs at ~/choicebench_hpc_eacl/tonight_configs_llama/
  (programmatically verified: only model-list differs from frozen originals, retained
  entry byte-identical). Could NOT merge sync+batch pairs the way the local HF configs
  were merged -- sync methods need provider=openrouter (deepinfra-pinned), batch
  methods need direct provider=deepinfra, so core_methods/cyclic_batch and
  reasoning/reasoning_batch stay split for Llama specifically (this is exactly why the
  frozen protocol split them in the first place).
- Dry-run sanity check on mmlu_core_methods.yaml confirmed correct scope (1 model,
  right benchmark/methods) before real launch.
- Launched all 10: 10029469 (mmlu_core_methods), 10029470 (arc_core_methods),
  10029471 (mmlu_reasoning), 10029472 (arc_reasoning), 10029473
  (mmlu_independent_hypothesis), 10029474 (arc_independent_hypothesis), 10029475
  (mmlu_reasoning_batch), 10029476 (arc_reasoning_batch), 10029477 (mmlu_cyclic),
  10029478 (arc_cyclic). All on k2-hipri, all queued PD/Priority at submission
  (healthy queue, not stuck).
