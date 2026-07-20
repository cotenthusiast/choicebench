# Flip-rate traces experiment

Targeted cyclic-permutation trace experiment: adds per-permutation trace
retention to the historical cyclic-generation-majority protocol for exactly
one API model (gpt-4.1-mini) and one local model (Qwen/Qwen2.5-7B-Instruct)
across ARC-Challenge and MMLU. See `docs/flip_rate_traces/PROTOCOL_DIFF.md`
for the full historical-protocol diff and `historical_protocol.py` for
source citations.

## Files

- `historical_protocol.py` — historically-faithful permutation/unpermute/
  majority-vote primitives, with exact source commit citations.
- `trace_schema.py` — per-permutation trace record schema + validation.
- `data_source.py` — frozen-dataset (Stage-1 paper freeze) question loader,
  sha256-verified against `model-generalization/paper_data_freeze/manifests/`.
- `runner.py` — `TraceRetainingCyclicRunner`: one trace row per permutation,
  for both API (`run_one_async`, batched via `generate_batch()`) and local
  (`run_one_sync`, sequential `generate()`) backends.
- `run_canary.py` — single-question canary driver; writes only to
  `runs/flip_rate_traces/_canary/`, every row stamped `is_diagnostic_canary=True`.
- `run_cell.py` — full 1000-question cell driver with checkpoint/resume,
  writes to `runs/flip_rate_traces/<run_id>/`.

## Configs

`configs/flip_rate_traces/*.yaml` — the 4 cells (api/local x arc_challenge/mmlu).

## Kelvin2

`slurm/flip_rate_traces/*.sbatch` — local-model jobs, `k2-gpu-a100` partition
(not `k2-gpu-v100`, which lacks torch kernel support for its compute
capability). Each script does a clean detached checkout of a pinned commit
SHA on a dedicated clone at
`/mnt/scratch2/users/40482774/repos/choicebench-flip-rate-traces`.

## Cache / checkpoint

Fresh cache namespace `flip_rate_traces_v1` at `.cache/flip_rate_traces_v1/`
(never shares or pollutes any prior experiment's cache). Checkpoints at
`checkpoints/flip_rate_traces/<run_id>/`.
