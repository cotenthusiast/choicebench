# scripts/paper/run_flip_rotations_reuse0.py
#
# Paper-specific (eacl-2026-revision): flip-rate rotations for two_stage / reasoning_two_stage / text_extraction /
# visible_llm_matcher where ROTATION 0 IS THE CANONICAL PRODUCTION RESULT and only rotations 1..n-1 are generated.
#
#   rotation 0     : imported from --production-csv (never sent to a backend). Verified to be exactly what a rerun would
#                    send: the production prompt must equal the rotation-0 prompt rebuilt here, and for the three
#                    letter-parsed families the production parsed_choice must equal a fresh parse of the saved raw_text.
#   rotations 1..  : one new call each, all pending questions of a chunk in ONE generate_batch() call.
#   text_extraction: EVERY rotation, including rotation 0 (re-parsed from the saved production raw_text), is scored with
#                    the exact->containment->cosine cascade. The production parsed_choice (letter parser) is kept only as
#                    rotation0_original_parsed_choice.
#   visible matcher: rotation 0's Stage-1 text is the production text (reused_extracted_text); rotations 1.. use the
#                    per-rotation raw texts of the text_extraction FLIP run (--text-extraction-flip-csv).
#
# Output rows carry per-rotation choices, sources ("production"/"generated"/"skipped_no_stage1_text"), raw outputs,
# transport statuses, upstream providers and latencies, plus the reused production file's path/SHA-256/run/condition.
# `--replay-rotation0 N` (pilot only) re-sends rotation 0 for the first N questions into a SEPARATE file to measure
# agreement with production; it is never merged into the flip output.
#
# Run with:
#   python scripts/paper/run_flip_rotations_reuse0.py --family two_stage --production-csv <prod csv> \
#       --model-config <cfg.yaml> --model-index 0 --run-id <unique id> --output runs/<dir>/<file>.csv [--limit N]

from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import json
from pathlib import Path

import pandas as pd

from choicebench.cli.run_experiment import build_backend
from choicebench.config.schema import load_config
from choicebench.methods.library.text_extraction import TextExtractionRunner
from choicebench.methods.library.two_stage import TwoStageRunner
from choicebench.methods.library.visible_llm_matcher import VisibleLLMMatcherRunner
from choicebench.pipeline.prompt_builder import build_direct_mcq_prompt, build_option_matching_prompt, build_rotations

FAMILIES = {
    "two_stage": ("two_stage", "v1", TwoStageRunner),
    "reasoning_two_stage": ("reasoning_two_stage", "v1_reasoning", TwoStageRunner),
    "text_extraction": ("text_extraction", "v1", TextExtractionRunner),
    "visible_llm_matcher": ("visible_llm_matcher", "v1", VisibleLLMMatcherRunner),
}
PROTOCOL = "rotation0=production;rotations1..n-1=generated"
_COPY = ("benchmark_name", "question_id", "split_name", "subject", "prompt_version", "provider", "model_name", "temperature",
         "max_tokens", "seed", "question_text", "choices_json", "correct_index", "correct_option", "n_choices")


def _nn(v):
    return None if v is None or (isinstance(v, float) and pd.isna(v)) else v


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _load_production(path: Path, family: str, limit: int | None) -> pd.DataFrame:
    method, _, _ = FAMILIES[family]
    df = pd.read_csv(path)
    if set(df["method_name"].astype(str)) != {method}:
        raise ValueError(f"{path} is not a {method} result file (method_name={sorted(set(df['method_name'].astype(str)))})")
    for col in ("model_name", "benchmark_name"):
        if df[col].nunique() != 1:
            raise ValueError(f"{path} mixes multiple {col} values")
    if df["question_id"].duplicated().any():
        raise ValueError(f"{path} has duplicate question_id values")
    return df.head(limit) if limit else df


def _make_runner(family: str, model_config, run_id: str, seed: int, backend, benchmark_name: str):
    method, pv, cls = FAMILIES[family]
    return cls(backend=backend, method_name=method, split_name="test", prompt_version=pv, prompts_dir=Path("prompts"),
               run_id=run_id, seed=seed, benchmark_name=benchmark_name,
               temperature=model_config.generation_kwargs.temperature, max_tokens=model_config.generation_kwargs.max_new_tokens)


def _prompt(runner, family: str, row: dict, rotation, stage1_text: str | None) -> str:
    if family == "text_extraction":
        return build_direct_mcq_prompt(template=runner._prompts["text_extraction"], question=row["question_text"],
                                       options=rotation.mapping, subject=row["subject"])
    return build_option_matching_prompt(template=runner._prompts["option_matching"], question=row["question_text"],
                                        free_text=stage1_text, options=rotation.mapping)


def _stage1_text0(family: str, row: dict):
    return {"two_stage": row.get("free_text_response"), "reasoning_two_stage": row.get("free_text_response"),
            "visible_llm_matcher": row.get("reused_extracted_text")}.get(family)


def _choice(runner, family: str, raw, rotation, letters, row) -> str | None:
    raw = _nn(raw)
    if raw is None:
        return None
    if family == "text_extraction":
        return runner._match_and_score(raw, row)[0].final_choice
    parsed = runner._parse(raw, rotation.mapping)
    return None if parsed.final_choice is None else letters[rotation.slot_to_canonical[letters.index(parsed.final_choice)]]


def _send(backend, runner, prompts: list[str]):
    if not prompts:
        return []
    if backend.is_async_capable():
        return asyncio.run(backend.generate_batch(prompts))
    return [runner._call_backend_generate(p) for p in prompts]


def _rotation0(runner, family: str, row: dict):
    """Rebuild rotation 0 and check it IS the production condition; returns (rotations, letters, rotation-0 choice)."""
    opts = runner._build_options(row)
    letters = list(opts.keys())
    rotations = build_rotations(opts)
    if rotations[0].mapping != opts or rotations[0].slot_to_canonical != tuple(range(len(letters))):
        raise ValueError(f"{row['question_id']}: rotation 0 is not the canonical option order")
    rebuilt = _prompt(runner, family, row, rotations[0], _stage1_text0(family, row))
    if rebuilt != row["prompt"]:
        raise ValueError(f"{row['question_id']}: production prompt differs from the rebuilt rotation 0 prompt -- refusing to reuse it as rotation 0")
    choice0 = _choice(runner, family, row.get("raw_text"), rotations[0], letters, row)
    if family != "text_extraction" and choice0 != _nn(row.get("parsed_choice")):
        raise ValueError(f"{row['question_id']}: production parsed_choice {_nn(row.get('parsed_choice'))!r} != parse of its saved raw_text {choice0!r}")
    return rotations, letters, choice0


def _load_flip_texts(path: Path | None, production: pd.DataFrame) -> dict[str, list]:
    if path is None:
        raise ValueError("visible_llm_matcher requires text_extraction_flip_csv (the text_extraction flip run's per-rotation raw texts)")
    flip = pd.read_csv(path)
    if set(flip["method_name"].astype(str)) != {"text_extraction"}:
        raise ValueError(f"{path} is not a text_extraction flip file")
    for col in ("model_name", "benchmark_name"):
        if set(flip[col].astype(str)) != set(production[col].astype(str)):
            raise ValueError(f"{path}: {col} does not match the production file")
    return {str(q): json.loads(t) for q, t in zip(flip["question_id"], flip["per_rotation_raw_text_json"])}


def _columns():
    return [*_COPY, "run_id", "method_name", "flip_protocol", "per_rotation_choices_json", "per_rotation_source_json",
            "per_rotation_raw_text_json", "per_rotation_transport_status_json", "per_rotation_upstream_provider_json",
            "per_rotation_error_json", "per_rotation_usage_json", "per_rotation_latency_json", "per_rotation_prompt_sha256_json", "generated_rotation_indices_json",
            "rotation0_source_file", "rotation0_source_sha256", "rotation0_source_run_id", "rotation0_source_condition_id",
            "rotation0_source_method_name", "rotation0_parser", "rotation0_original_parsed_choice", "free_text_response",
            "stage1_text_flip_file", "stage1_text_flip_sha256"]


def run(*, production_csv: Path, family: str, model_config, run_id: str, output_path: Path, backend=None, seed: int = 42,
        execution_mode: str = "sync", chunk_size: int = 50, limit: int | None = None, text_extraction_flip_csv: Path | None = None,
        max_failure_rate: float = 0.5) -> int:
    production_csv, output_path = Path(production_csv), Path(output_path)
    if output_path.resolve() == production_csv.resolve():
        raise ValueError("output must never be the canonical production file")
    prod = _load_production(production_csv, family, limit)
    flip_texts = _load_flip_texts(text_extraction_flip_csv, prod) if family == "visible_llm_matcher" else None
    prod_sha = _sha256(production_csv)
    flip_sha = _sha256(text_extraction_flip_csv) if flip_texts is not None else None
    done = set(pd.read_csv(output_path, usecols=["question_id"])["question_id"].astype(str)) if output_path.exists() else set()
    pending = prod[~prod["question_id"].astype(str).isin(done)]
    if pending.empty:
        return 0
    backend = backend or build_backend(model_config, run_id, run_seed=seed, execution_mode=execution_mode)
    runner = _make_runner(family, model_config, run_id, seed, backend, str(prod["benchmark_name"].iloc[0]))
    method = FAMILIES[family][0]
    parser0 = "text_matcher_cascade_reparse_of_saved_raw_text" if family == "text_extraction" else "production_parsed_choice_verified_equal_to_reparse"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    for start in range(0, len(pending), chunk_size):
        chunk = pending.iloc[start:start + chunk_size].to_dict(orient="records")
        plans, prompts = [], []
        for row in chunk:
            rotations, letters, choice0 = _rotation0(runner, family, row)
            plan = {"row": row, "rotations": rotations, "letters": letters, "choice0": choice0,
                    "slots": ["production"] + [None] * (len(rotations) - 1), "prompts": [None] * len(rotations)}
            for k in range(1, len(rotations)):
                text = None
                if family == "visible_llm_matcher":
                    texts = flip_texts.get(str(row["question_id"]))
                    if texts is None or len(texts) != len(rotations):
                        raise ValueError(f"{row['question_id']}: text_extraction flip file has no {len(rotations)} per-rotation texts")
                    text = _nn(texts[k])
                    if text is None:
                        plan["slots"][k] = "skipped_no_stage1_text"
                        continue
                elif family != "text_extraction":
                    text = row["free_text_response"]
                plan["slots"][k] = "generated"
                plan["prompts"][k] = _prompt(runner, family, row, rotations[k], text)
                prompts.append(plan["prompts"][k])
            plans.append(plan)
        sent = _send(backend, runner, prompts)
        failed = [r for r in sent if not r.is_success()]
        if sent and len(failed) / len(sent) > max_failure_rate:
            first = failed[0].error.error_type if failed[0].error else "unknown"
            raise RuntimeError(f"transport failure rate {len(failed)}/{len(sent)} on this chunk exceeds {max_failure_rate:.0%} "
                               f"(first error: {first}) -- aborting before writing anything; fix the backend/pin and resume")
        responses = iter(sent)

        rows_out = []
        for plan in plans:
            row, n = plan["row"], len(plan["rotations"])
            choices, raws, status, upstream, latency, sha = [plan["choice0"]], [_nn(row.get("raw_text"))], [_nn(row.get("transport_status"))], \
                [_nn(row.get("actual_upstream_provider"))], [_nn(row.get("latency_seconds"))], [hashlib.sha256(row["prompt"].encode()).hexdigest()]
            errors = [_nn(row.get("error_type"))]
            usage = [None]
            for k in range(1, n):
                if plan["slots"][k] != "generated":
                    choices.append(None); raws.append(None); status.append(None); upstream.append(None); latency.append(None); sha.append(None); errors.append(None); usage.append(None)
                    continue
                r = next(responses)
                ok = r.is_success()
                choices.append(_choice(runner, family, r.raw_text if ok else None, plan["rotations"][k], plan["letters"], row))
                raws.append(r.raw_text if ok else None); status.append(r.status); upstream.append(r.actual_upstream_provider)
                latency.append(r.latency_seconds); sha.append(hashlib.sha256(plan["prompts"][k].encode()).hexdigest())
                errors.append(r.error.error_type if r.error else None)
                usage.append({"prompt_tokens": r.usage.prompt_tokens, "completion_tokens": r.usage.completion_tokens} if r.usage else None)
            out = {c: _nn(row.get(c)) for c in _COPY}
            out.update(run_id=run_id, method_name=method, flip_protocol=PROTOCOL,
                       per_rotation_choices_json=json.dumps(choices), per_rotation_source_json=json.dumps(plan["slots"]),
                       per_rotation_raw_text_json=json.dumps(raws), per_rotation_transport_status_json=json.dumps(status),
                       per_rotation_upstream_provider_json=json.dumps(upstream), per_rotation_error_json=json.dumps(errors), per_rotation_usage_json=json.dumps(usage),
                       per_rotation_latency_json=json.dumps(latency),
                       per_rotation_prompt_sha256_json=json.dumps(sha),
                       generated_rotation_indices_json=json.dumps([k for k, s in enumerate(plan["slots"]) if s == "generated"]),
                       rotation0_source_file=str(production_csv), rotation0_source_sha256=prod_sha,
                       rotation0_source_run_id=_nn(row.get("run_id")), rotation0_source_condition_id=_nn(row.get("condition_id")),
                       rotation0_source_method_name=_nn(row.get("method_name")), rotation0_parser=parser0,
                       rotation0_original_parsed_choice=_nn(row.get("parsed_choice")),
                       free_text_response=_nn(row.get("free_text_response")),
                       stage1_text_flip_file=str(text_extraction_flip_csv) if flip_sha else None, stage1_text_flip_sha256=flip_sha)
            rows_out.append(out)
        new_file = not output_path.exists()
        with open(output_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=_columns())
            if new_file:
                writer.writeheader()
            writer.writerows(rows_out)
        written += len(rows_out)
    return written


def replay_rotation0(*, production_csv: Path, family: str, model_config, run_id: str, output_path: Path, n: int, backend=None,
                     seed: int = 42, execution_mode: str = "sync", text_extraction_flip_csv: Path | None = None) -> dict:
    """PILOT ONLY: re-send rotation 0 for the first n questions and compare with production (API-nondeterminism probe)."""
    production_csv, output_path = Path(production_csv), Path(output_path)
    if output_path.resolve() == production_csv.resolve():
        raise ValueError("output must never be the canonical production file")
    prod = _load_production(production_csv, family, n)
    backend = backend or build_backend(model_config, run_id, run_seed=seed, execution_mode=execution_mode)
    runner = _make_runner(family, model_config, run_id, seed, backend, str(prod["benchmark_name"].iloc[0]))
    rows = prod.to_dict(orient="records")
    checked = [_rotation0(runner, family, r) for r in rows]
    responses = _send(backend, runner, [r["prompt"] for r in rows])
    out = []
    for row, (rotations, letters, choice0), resp in zip(rows, checked, responses):
        ok = resp.is_success()
        replay = _choice(runner, family, resp.raw_text if ok else None, rotations[0], letters, row)
        out.append({"question_id": row["question_id"], "production_choice": choice0, "replay_choice": replay,
                    "agree": ok and replay == choice0, "production_raw_text": _nn(row.get("raw_text")),
                    "replay_raw_text": resp.raw_text if ok else None, "transport_status": resp.status,
                    "actual_upstream_provider": resp.actual_upstream_provider, "latency_seconds": resp.latency_seconds,
                    "rotation0_source_sha256": _sha256(production_csv)})
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(out).to_csv(output_path, index=False)
    agree = int(sum(bool(o["agree"]) for o in out))
    return {"n": len(out), "agree": agree, "agreement": agree / len(out) if out else float("nan")}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--family", required=True, choices=sorted(FAMILIES))
    p.add_argument("--production-csv", required=True, type=Path)
    p.add_argument("--model-config", required=True, type=Path)
    p.add_argument("--model-index", type=int, default=0)
    p.add_argument("--run-id", required=True)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--execution-mode", choices=["sync", "batch"], default="sync")
    p.add_argument("--chunk-size", type=int, default=50)
    p.add_argument("--limit", type=int, default=None, help="only the first N production rows (pilot)")
    p.add_argument("--text-extraction-flip-csv", type=Path, default=None)
    p.add_argument("--max-transport-failure-rate", type=float, default=0.5,
                   help="abort a chunk (writing nothing) when more than this share of its calls fail")
    p.add_argument("--replay-rotation0", type=int, default=0, help="pilot only: re-send rotation 0 for the first N questions")
    p.add_argument("--replay-output", type=Path, default=None)
    a = p.parse_args()
    cfg = load_config(str(a.model_config)).models[a.model_index]
    if a.replay_rotation0:
        if a.replay_output is None:
            raise SystemExit("--replay-rotation0 needs --replay-output (a file separate from the flip output)")
        summary = replay_rotation0(production_csv=a.production_csv, family=a.family, model_config=cfg, run_id=f"{a.run_id}_replay",
                                   output_path=a.replay_output, n=a.replay_rotation0, seed=a.seed, execution_mode=a.execution_mode)
        print("rotation-0 replay:", summary)
    n = run(production_csv=a.production_csv, family=a.family, model_config=cfg, run_id=a.run_id, output_path=a.output, seed=a.seed,
            execution_mode=a.execution_mode, chunk_size=a.chunk_size, limit=a.limit, text_extraction_flip_csv=a.text_extraction_flip_csv,
            max_failure_rate=a.max_transport_failure_rate)
    print(f"wrote {n} new rows -> {a.output}")


if __name__ == "__main__":
    main()
