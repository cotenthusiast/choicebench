# experiments/visible_llm_matcher/repairs/group3_fourth_cell/generate_configs.py
#
# Generates the 12 (6 models x 2 benchmarks) run_fourth_cell.py configs —
# 8 API (api/configs/) + 4 local/Kelvin2 (local/configs/). Makes no network
# calls; only writes YAML into this ChoiceBench worktree.

from __future__ import annotations

from pathlib import Path

import yaml

THIS_DIR = Path(__file__).resolve().parent
TSP_ROOT = "/home/cotenthusiast/Projects/two-stage-prompting"
MG_ROOT = "/home/cotenthusiast/Projects/model-generalization"

RUN_ID = "fourth_cell_v1"
PROMPTS_DIR = "experiments/visible_llm_matcher/prompts"

# Group 2's repaired output: same run_id/model/benchmark as the original
# text_extraction cell, produced by Group 1's checkpoint-seeded repair
# landing (see ../group1_historical_arc/). NOT present yet — run_fourth_cell.py
# will fail closed (Stage1ValidationError) if this path doesn't exist or is
# still missing the 3 repaired rows, exactly as required.
_API_TEXT_EXTRACTION_RUN_ID = "20260603_154649"


def _api_stage1_source(model_name: str, model_safe: str, benchmark: str) -> dict:
    return {
        "repo": "two-stage-prompting",
        "path": (
            f"{TSP_ROOT}/paper_results/eval_ready/paper_api_main/"
            f"{_API_TEXT_EXTRACTION_RUN_ID}_text_extraction_{model_safe}_{benchmark}.csv"
        ),
        "model_name": model_name,
        "benchmark": benchmark,
    }


def _api_replacement_source(model_name: str, model_safe: str) -> dict:
    """The repaired (post Group-1-repair) ARC text_extraction CSV — same
    run_id, same file path (the repair updates it in place via
    run_experiment.py's normal --run-id resume/merge behavior, per
    ../group1_historical_arc/README.md). Only used for benchmark=arc_challenge.
    """
    return _api_stage1_source(model_name, model_safe, "arc_challenge")


API_MODELS = [
    {"model_name": "gpt-4.1-mini", "safe": "gpt-4.1-mini", "provider": "openai", "concurrency": 10},
    {"model_name": "gemini-2.5-flash", "safe": "gemini-2.5-flash", "provider": "gemini", "concurrency": 1},
    {"model_name": "llama-3.1-8b-instant", "safe": "llama-3.1-8b-instant", "provider": "groq", "concurrency": 2},
    {"model_name": "Qwen/Qwen2.5-7B-Instruct-Turbo", "safe": "Qwen_Qwen2.5-7B-Instruct-Turbo", "provider": "together", "concurrency": 5},
]

LOCAL_MODELS = [
    {"model_name": "Qwen/Qwen2.5-7B-Instruct", "safe": "Qwen_Qwen2.5-7B-Instruct"},
    {"model_name": "meta-llama/Llama-3.1-8B-Instruct", "safe": "meta-llama_Llama-3.1-8B-Instruct"},
]

BENCHMARKS = ["mmlu", "arc_challenge"]


def _mmlu_source_for_api(model_name: str, model_safe: str) -> dict:
    return _api_stage1_source(model_name, model_safe, "mmlu")


def _mmlu_source_for_local(model_name: str, model_safe: str) -> dict:
    run_id = "20260603_213613" if "Qwen" in model_name else "20260603_214857"
    return {
        "repo": "two-stage-prompting",
        "path": (
            f"{TSP_ROOT}/paper_results/eval_ready/paper_local_main/"
            f"{run_id}_text_extraction_{model_safe}_mmlu.csv"
        ),
        "model_name": model_name,
        "benchmark": "mmlu",
    }


def _arc_source_for_local(model_name: str, model_safe: str) -> dict:
    return {
        "repo": "model-generalization",
        "path": f"{MG_ROOT}/runs/20260617_162624/20260617_162624_text_extraction_{model_safe}_arc_challenge.csv",
        "model_name": model_name,
        "benchmark": "arc_challenge",
    }


def write_api_configs() -> list[Path]:
    out_dir = THIS_DIR / "api" / "configs"
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for m in API_MODELS:
        for benchmark in BENCHMARKS:
            source = (
                _mmlu_source_for_api(m["model_name"], m["safe"])
                if benchmark == "mmlu"
                else _api_stage1_source(m["model_name"], m["safe"], "arc_challenge")
            )
            replacement = (
                None if benchmark == "mmlu" else _api_replacement_source(m["model_name"], m["safe"])
            )
            cfg = {
                "run_id": RUN_ID,
                "model": {
                    "backend": "api",
                    "provider": m["provider"],
                    "model_name_or_path": m["model_name"],
                    "generation_kwargs": {"temperature": 0.0, "max_new_tokens": 500},
                    "concurrency_limit": m["concurrency"],
                },
                "benchmark": benchmark,
                "seed": 42,
                "prompt_version": "v1",
                "prompts_dir": PROMPTS_DIR,
                "stage1_sources": [source],
                "stage1_replacement_source": replacement,
                "output_csv": (
                    f"runs/{RUN_ID}/{RUN_ID}_visible_llm_matcher_{m['safe']}_{benchmark}.csv"
                ),
                "checkpoint_dir": "checkpoints",
            }
            path = out_dir / f"{m['safe']}__{benchmark}.yaml"
            path.write_text(yaml.safe_dump(cfg, sort_keys=False))
            written.append(path)
    return written


def write_local_configs() -> list[Path]:
    out_dir = THIS_DIR / "local" / "configs"
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for m in LOCAL_MODELS:
        for benchmark in BENCHMARKS:
            source = (
                _mmlu_source_for_local(m["model_name"], m["safe"])
                if benchmark == "mmlu"
                else _arc_source_for_local(m["model_name"], m["safe"])
            )
            # Local ARC is self-replacing: the same file's own 3
            # correctly-elicited rows stand in as their own
            # replacement_rows (see ../group2_stage1_repair/README.md —
            # already verified valid, no separate repair file exists or is
            # needed).
            replacement = source if benchmark == "arc_challenge" else None
            cfg = {
                "run_id": RUN_ID,
                "model": {
                    "backend": "huggingface",
                    "model_name_or_path": m["model_name"],
                    "device": "cuda",
                    "generation_kwargs": {"temperature": 0.0, "max_new_tokens": 500, "do_sample": False},
                },
                "benchmark": benchmark,
                "seed": 42,
                "prompt_version": "v1",
                "prompts_dir": PROMPTS_DIR,
                "stage1_sources": [source],
                "stage1_replacement_source": replacement,
                "output_csv": (
                    f"runs/{RUN_ID}/{RUN_ID}_visible_llm_matcher_{m['safe']}_{benchmark}.csv"
                ),
                "checkpoint_dir": "checkpoints",
            }
            path = out_dir / f"{m['safe']}__{benchmark}.yaml"
            path.write_text(yaml.safe_dump(cfg, sort_keys=False))
            written.append(path)
    return written


if __name__ == "__main__":
    api_paths = write_api_configs()
    local_paths = write_local_configs()
    print(f"Wrote {len(api_paths)} API configs, {len(local_paths)} local configs.")
    for p in api_paths + local_paths:
        print(" ", p)
