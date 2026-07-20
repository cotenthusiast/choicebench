# experiments/visible_llm_matcher/repairs/group1_historical_arc/harness/stage1_reuse_sources.py
#
# Exact historical Stage-1 (hidden-options, two_prompt) source paths per
# (prompt_version, model), for the two_stage repair's Stage-1 reuse
# decision (see two_stage_repair.py's STAGE1_REUSE_DECISION docstring).
# All verified present, complete (non-empty free_text_response for the 3
# repaired IDs), and content-plausible on 2026-07-20.

from __future__ import annotations

from pathlib import Path

TSP_ROOT = Path("/home/cotenthusiast/Projects/two-stage-prompting")

# (prompt_version, model_name) -> historical two_prompt CSV path.
_API_SOURCES: dict[tuple[str, str], Path] = {
    ("v1", "gpt-4.1-mini"): TSP_ROOT / "runs/20260601_183346/20260601_183346_two_prompt_gpt-4.1-mini_arc_challenge.csv",
    ("v1", "gemini-2.5-flash"): TSP_ROOT / "runs/20260601_183346/20260601_183346_two_prompt_gemini-2.5-flash_arc_challenge.csv",
    ("v1", "llama-3.1-8b-instant"): TSP_ROOT / "runs/20260601_183346/20260601_183346_two_prompt_llama-3.1-8b-instant_arc_challenge.csv",
    ("v1", "Qwen/Qwen2.5-7B-Instruct-Turbo"): TSP_ROOT / "runs/20260601_183346/20260601_183346_two_prompt_Qwen_Qwen2.5-7B-Instruct-Turbo_arc_challenge.csv",
    ("v2", "gpt-4.1-mini"): TSP_ROOT / "runs/20260603_105547/20260603_105547_two_prompt_gpt-4.1-mini_arc_challenge.csv",
    ("v2", "gemini-2.5-flash"): TSP_ROOT / "runs/20260603_105547/20260603_105547_two_prompt_gemini-2.5-flash_arc_challenge.csv",
    ("v2", "llama-3.1-8b-instant"): TSP_ROOT / "runs/20260603_105547/20260603_105547_two_prompt_llama-3.1-8b-instant_arc_challenge.csv",
    ("v2", "Qwen/Qwen2.5-7B-Instruct-Turbo"): TSP_ROOT / "runs/20260603_105547/20260603_105547_two_prompt_Qwen_Qwen2.5-7B-Instruct-Turbo_arc_challenge.csv",
    ("v3", "gpt-4.1-mini"): TSP_ROOT / "runs/20260603_134908/20260603_134908_two_prompt_gpt-4.1-mini_arc_challenge.csv",
    ("v3", "gemini-2.5-flash"): TSP_ROOT / "runs/20260603_134908/20260603_134908_two_prompt_gemini-2.5-flash_arc_challenge.csv",
    ("v3", "llama-3.1-8b-instant"): TSP_ROOT / "runs/20260603_134908/20260603_134908_two_prompt_llama-3.1-8b-instant_arc_challenge.csv",
    ("v3", "Qwen/Qwen2.5-7B-Instruct-Turbo"): TSP_ROOT / "runs/20260603_134908/20260603_134908_two_prompt_Qwen_Qwen2.5-7B-Instruct-Turbo_arc_challenge.csv",
}

_LOCAL_SOURCES: dict[tuple[str, str], Path] = {
    ("v2", "Qwen/Qwen2.5-7B-Instruct"): TSP_ROOT / "paper_results/raw_imports/runs_two_stage_paper/runs/20260602_162026/20260602_162026_two_prompt_Qwen_Qwen2.5-7B-Instruct_arc_challenge.csv",
    ("v2", "meta-llama/Llama-3.1-8B-Instruct"): TSP_ROOT / "paper_results/raw_imports/runs_two_stage_paper/runs/20260602_171722/20260602_171722_two_prompt_meta-llama_Llama-3.1-8B-Instruct_arc_challenge.csv",
    ("v3", "Qwen/Qwen2.5-7B-Instruct"): TSP_ROOT / "paper_results/raw_imports/runs_two_stage_paper/runs/20260602_183821/20260602_183821_two_prompt_Qwen_Qwen2.5-7B-Instruct_arc_challenge.csv",
    ("v3", "meta-llama/Llama-3.1-8B-Instruct"): TSP_ROOT / "paper_results/raw_imports/runs_two_stage_paper/runs/20260603_001901/20260603_001901_two_prompt_meta-llama_Llama-3.1-8B-Instruct_arc_challenge.csv",
}

_METHOD_TO_VERSION = {"two_stage_v1": "v1", "two_stage_v2": "v2", "two_stage_v3": "v3"}


class Stage1ReuseSourceError(RuntimeError):
    pass


def resolve_stage1_source_path(method: str, model_name: str) -> Path:
    from experiments.visible_llm_matcher.source_path_rewrite import rewrite_source_path

    version = _METHOD_TO_VERSION.get(method)
    if version is None:
        raise Stage1ReuseSourceError(f"Not a two_stage method: {method!r}")
    key = (version, model_name)
    path = _API_SOURCES.get(key) or _LOCAL_SOURCES.get(key)
    if path is None:
        raise Stage1ReuseSourceError(
            f"No known Stage-1 reuse source for method={method!r} model={model_name!r}. "
            "two_stage_v1 is not in the local repair scope (only v2/v3 are)."
        )
    path = rewrite_source_path(path)
    if not path.is_file():
        raise Stage1ReuseSourceError(f"Stage-1 reuse source does not exist: {path}")
    return path


def load_reused_free_text(method: str, model_name: str, question_id: str) -> str:
    import pandas as pd

    path = resolve_stage1_source_path(method, model_name)
    df = pd.read_csv(path)
    row = df[df["question_id"] == question_id]
    if row.empty:
        raise Stage1ReuseSourceError(
            f"question_id={question_id!r} not found in Stage-1 reuse source {path}"
        )
    ft = row.iloc[0]["free_text_response"]
    if not isinstance(ft, str) or not ft.strip():
        raise Stage1ReuseSourceError(
            f"question_id={question_id!r} has no usable free_text_response in {path}"
        )
    return ft
