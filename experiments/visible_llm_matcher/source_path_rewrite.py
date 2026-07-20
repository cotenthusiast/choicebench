# experiments/visible_llm_matcher/source_path_rewrite.py
#
# All Group1/Group3 config generation baked in absolute paths under this
# local sandbox's two-stage-prompting and model-generalization checkouts
# (e.g. "/home/cotenthusiast/Projects/two-stage-prompting/runs/...csv").
# Those paths don't exist on Kelvin2. Rather than re-hardcoding a second
# set of Kelvin2-specific paths throughout stage1_reuse_sources.py,
# run_fourth_cell.py's config-driven stage1_sources entries, etc., this
# module provides ONE rewrite point: when VLM_SOURCE_DATA_ROOT is set (e.g.
# on Kelvin2, to a directory containing synced copies of just the specific
# files actually needed, mirrored under two-stage-prompting/... and
# model-generalization/... subdirectories), the two known local repo root
# prefixes are replaced with {VLM_SOURCE_DATA_ROOT}/<repo-name>. When unset
# (the default -- this local sandbox), paths are returned unchanged.
#
# This does not change WHICH files are used, only WHERE they are read
# from -- every file this resolves to is still the exact same historical
# artifact, referenced by the exact same run_id/filename, just possibly at
# a different absolute prefix.

from __future__ import annotations

import os
from pathlib import Path

_LOCAL_TSP_ROOT = "/home/cotenthusiast/Projects/two-stage-prompting"
_LOCAL_MG_ROOT = "/home/cotenthusiast/Projects/model-generalization"


def rewrite_source_path(path: str | Path) -> Path:
    """Rewrite a hardcoded local-sandbox source path to
    VLM_SOURCE_DATA_ROOT's mirrored location, if that env var is set;
    otherwise return the path unchanged.
    """
    path_str = str(path)
    root = os.environ.get("VLM_SOURCE_DATA_ROOT")
    if not root:
        return Path(path_str)

    if path_str.startswith(_LOCAL_TSP_ROOT):
        return Path(root) / "two-stage-prompting" / path_str[len(_LOCAL_TSP_ROOT):].lstrip("/")
    if path_str.startswith(_LOCAL_MG_ROOT):
        return Path(root) / "model-generalization" / path_str[len(_LOCAL_MG_ROOT):].lstrip("/")
    return Path(path_str)
