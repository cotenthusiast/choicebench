# experiments/visible_llm_matcher/repairs/group3_fourth_cell/local/slurm/generate_sbatch.py
#
# Generates the 4 Kelvin2 sbatch scripts (2 local models x 2 benchmarks).
# PINNED_SHA is a placeholder filled in by the top-level repair report
# after the final commit of this config-generation phase — see
# ../../../REPORT.md and the note at the top of each generated .sbatch file.

from __future__ import annotations

from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent

PINNED_SHA_PLACEHOLDER = "PINNED_SHA_PLACEHOLDER_SEE_REPORT_MD"

CHOICEBENCH_WORKTREE_ON_KELVIN2 = "/mnt/scratch2/users/40482774/repos/choicebench-visible-llm-matcher-cell"
MG_ROOT_ON_KELVIN2 = "/mnt/scratch2/users/40482774/repos/model-generalization"

CELLS = [
    {"model": "Qwen/Qwen2.5-7B-Instruct", "safe": "Qwen_Qwen2.5-7B-Instruct", "benchmark": "mmlu"},
    {"model": "Qwen/Qwen2.5-7B-Instruct", "safe": "Qwen_Qwen2.5-7B-Instruct", "benchmark": "arc_challenge"},
    {"model": "meta-llama/Llama-3.1-8B-Instruct", "safe": "meta-llama_Llama-3.1-8B-Instruct", "benchmark": "mmlu"},
    {"model": "meta-llama/Llama-3.1-8B-Instruct", "safe": "meta-llama_Llama-3.1-8B-Instruct", "benchmark": "arc_challenge"},
]

TEMPLATE = """#!/bin/bash
#SBATCH --job-name=vlm_{safe}_{benchmark}
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --mem=48G
#SBATCH --output=vlm_{safe}_{benchmark}_%j.log
#
# Group 3 (fourth cell, options-visible + LLM matcher): {model} x {benchmark}.
# Verify partition/GRES names with `sinfo -o "%P %D %G %m %l %N"` before
# submitting -- gpu / gpu:1 above are placeholders, not verified against
# live Kelvin2 partition names (this config-generation phase has no Kelvin2
# access).
#
# PINNED COMMIT — checked out in detached-head mode, not on a branch tip,
# so this job runs exactly the config/runner code this repair phase
# produced, not whatever main has moved to since:
#   git -C {choicebench_root} fetch origin
#   git -C {choicebench_root} checkout --detach {pinned_sha}
#
# This script assumes {choicebench_root} is already a checkout of the
# choicebench repo (branch worktree-visible-llm-matcher-cell) on this
# Kelvin2 filesystem -- clone it first if it is not:
#   git clone git@github.com:cotenthusiast/choicebench.git {choicebench_root}

set -euo pipefail

CHOICEBENCH_ROOT="{choicebench_root}"
MG_ROOT="{mg_root}"

cd "${{CHOICEBENCH_ROOT}}"
git fetch origin
git checkout --detach {pinned_sha}

source env_kelvin2.sh 2>/dev/null || true  # if this worktree carries one; else set HF_HOME/PYTHONPATH manually

python experiments/visible_llm_matcher/repairs/group3_fourth_cell/run_fourth_cell.py \\
    --config experiments/visible_llm_matcher/repairs/group3_fourth_cell/local/configs/{safe}__{benchmark}.yaml
"""


def write_all() -> list[Path]:
    written = []
    for cell in CELLS:
        content = TEMPLATE.format(
            safe=cell["safe"],
            benchmark=cell["benchmark"],
            model=cell["model"],
            choicebench_root=CHOICEBENCH_WORKTREE_ON_KELVIN2,
            mg_root=MG_ROOT_ON_KELVIN2,
            pinned_sha=PINNED_SHA_PLACEHOLDER,
        )
        path = THIS_DIR / f"run_{cell['safe']}__{cell['benchmark']}.sbatch"
        path.write_text(content)
        written.append(path)
    return written


if __name__ == "__main__":
    for p in write_all():
        print(p)
