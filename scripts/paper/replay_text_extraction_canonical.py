# scripts/paper/replay_text_extraction_canonical.py
#
# Paper-specific (eacl-2026-revision): make the corrected text_extraction results canonical WITHOUT any new inference.
#
# The saved raw generations of the original (letter-parsed) text_extraction conditions are primed into a NEW run's response
# cache under the exact key a live request would use (provider, model, prompt, temperature, max_tokens, seed, provider pin).
# The fixed TextExtractionRunner is then executed through the normal run_experiment pipeline, so the new run gets a genuine
# manifest / artifact / SHA-256 chain, with every request a cache hit. Run it with provider keys that are NOT valid
# (a miss then becomes a failed row that `verify` reports; it can never spend money).
#
#   prime : python scripts/paper/replay_text_extraction_canonical.py prime --original-run runs/<core_methods run> \
#               --config <TE-only config.yaml> --run-id <new run id>
#   (then) python scripts/run_experiment.py --config <TE-only config.yaml> --run-id <new run id> --yes
#   verify: python scripts/paper/replay_text_extraction_canonical.py verify --original-run ... --new-run runs/<new run id> \
#               --corrected-dir runs/eacl2026_text_extraction_corrected
#
# Latency and timestamp columns of the new run describe the replay, not the original calls; the original run stays the record of those.

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from choicebench.clients.types import ModelResponse, SUCCESS_STATUS
from choicebench.infra.cache import CachingClientWrapper, _cache_key, client_extra_identity


def _nn(v):
    return None if v is None or (isinstance(v, float) and pd.isna(v)) else v


def prime_cache(backend, rows: list[dict]) -> int:
    """Store every successful original row's response under the key its live request would use; returns how many."""
    n = 0
    for row in rows:
        if row.get("transport_status") != SUCCESS_STATUS:
            continue
        response = ModelResponse(provider=backend.provider, model_name=backend.model_name, status=SUCCESS_STATUS, latency_seconds=0.0,
                                 raw_text=row["raw_text"], finish_reason=_nn(row.get("finish_reason")),
                                 actual_upstream_provider=_nn(row.get("actual_upstream_provider")))
        key = _cache_key(backend._make_request(row["prompt"]), extra_identity=client_extra_identity(backend._raw_client))
        backend._client._cache.put(key, CachingClientWrapper._serialize_response(response))
        n += 1
    return n


def _same(a, b) -> pd.Series:
    return (a == b) | (a.isna() & b.isna())


def verify_replay(original: pd.DataFrame, replayed: pd.DataFrame, corrected: pd.DataFrame) -> list[str]:
    """Problems (empty = faithful): same question_id set; prompt/raw_text/finish_reason equal to the original; every replayed row a
    transport success; parsed_choice/is_correct equal to the independently derived corrected file (a second, separate code path)."""
    problems: list[str] = []
    ids = [set(df["question_id"].astype(str)) for df in (original, replayed, corrected)]
    if not (ids[0] == ids[1] == ids[2]) or any(df["question_id"].duplicated().any() for df in (original, replayed, corrected)):
        return [f"question_id sets differ or repeat (original={len(ids[0])}, replayed={len(ids[1])}, corrected={len(ids[2])})"]
    o, r, c = (df.assign(question_id=df["question_id"].astype(str)).set_index("question_id").sort_index() for df in (original, replayed, corrected))
    for col in ("prompt", "raw_text", "finish_reason"):
        bad = int((~_same(o[col], r[col])).sum())
        if bad:
            problems.append(f"{col}: {bad} rows differ between the original and the replayed run")
    bad = int((r["transport_status"] != SUCCESS_STATUS).sum())
    if bad:
        problems.append(f"transport_status: {bad} replayed rows are not successes (a request missed the primed cache)")
    for col in ("parsed_choice", "is_correct"):
        bad = int((~_same(r[col].astype(object), c[col].astype(object))).sum())
        if bad:
            problems.append(f"{col}: {bad} rows differ from the independently derived corrected file")
    return problems


def _te_conditions(run_dir: Path) -> list[dict]:
    manifest = json.loads((run_dir / "manifest.json").read_text())["payload"]
    state = json.loads((run_dir / "run_state.json").read_text())["conditions"]
    names = {m["method_id"]: m["config"]["name"] for m in manifest["methods"]}
    out = []
    for c in manifest["conditions"]:
        if names[c["method_id"]] == "text_extraction":
            out.append({"condition_id": c["condition_id"], "model_id": c["identity"]["model_id"], "model": c["model_display_name"],
                        "benchmark": c["benchmark_name"], "result": run_dir / state[c["condition_id"]]["result_path"],
                        "sha256": state[c["condition_id"]]["result_sha256"]})
    return out


def _model_key(name: str) -> str:
    return next(k for k in ("gpt", "claude", "qwen", "llama") if k in name.lower())


def prime_run(*, original_run: Path, config_path: Path, run_id: str, seed: int = 42) -> dict:
    import hashlib

    from choicebench.cli.run_experiment import build_backend
    from choicebench.config.schema import load_config

    config = load_config(str(config_path))
    summary = {}
    for cond in _te_conditions(Path(original_run)):
        if hashlib.sha256(cond["result"].read_bytes()).hexdigest() != cond["sha256"]:
            raise ValueError(f"{cond['result']}: SHA-256 differs from run_state -- refusing to replay a modified result")
        cfg = next(m for m in config.models if m.model_name_or_path.lower().endswith(cond["model"].lower().split("/")[-1]))
        backend = build_backend(cfg, run_id, run_seed=seed, model_identity=cond["model_id"])
        summary[cond["condition_id"]] = prime_cache(backend, pd.read_csv(cond["result"]).to_dict(orient="records"))
    return summary


def verify_run(*, original_run: Path, new_run: Path, corrected_dir: Path) -> list[str]:
    problems: list[str] = []
    new_manifest = json.loads((Path(new_run) / "manifest.json").read_text())["payload"]
    new_state = json.loads((Path(new_run) / "run_state.json").read_text())["conditions"]
    new_by_model = {(c["identity"]["model_id"], c["benchmark_name"]): Path(new_run) / new_state[c["condition_id"]]["result_path"]
                    for c in new_manifest["conditions"]}
    for cond in _te_conditions(Path(original_run)):
        key = (cond["model_id"], cond["benchmark"])
        if key not in new_by_model:
            problems.append(f"{cond['model']}/{cond['benchmark']}: no matching condition (same model_id) in the new run")
            continue
        bench = "arc" if cond["benchmark"].startswith("arc") else "mmlu"
        corrected = pd.read_csv(Path(corrected_dir) / f"{bench}_text_extraction_corrected_{_model_key(cond['model'])}.csv")
        problems += [f"{cond['model']}/{cond['benchmark']}: {p}" for p in
                     verify_replay(pd.read_csv(cond["result"]), pd.read_csv(new_by_model[key]), corrected)]
    return problems


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("prime")
    a.add_argument("--original-run", required=True, type=Path); a.add_argument("--config", required=True, type=Path); a.add_argument("--run-id", required=True)
    v = sub.add_parser("verify")
    v.add_argument("--original-run", required=True, type=Path); v.add_argument("--new-run", required=True, type=Path)
    v.add_argument("--corrected-dir", required=True, type=Path)
    args = p.parse_args()
    if args.cmd == "prime":
        print("primed:", prime_run(original_run=args.original_run, config_path=args.config, run_id=args.run_id))
        return
    problems = verify_run(original_run=args.original_run, new_run=args.new_run, corrected_dir=args.corrected_dir)
    print("VERIFIED: replay is faithful" if not problems else "PROBLEMS:\n  " + "\n  ".join(problems))
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
