# scripts/paper/te_corrected_diagnostics.py
#
# Paper-specific (eacl-2026-revision): recomputes every statistic that depends on the parsed text_extraction answer,
# from SAVED outputs only (zero inference): per-cell Accuracy (end-to-end + conditional, Clopper-Pearson CIs), MAD and
# RecallRStd for the original (letter-parser) and corrected (cascade) text_extraction files, the 2x2 visibility x matcher
# diagnostic with the corrected cell, and exact-McNemar paired tests against the other cells. order_sensitivity (flip rate,
# causal RStd) needs per-rotation data text_extraction does not have yet, so it is not computed here.
#
# Run with:
#   python scripts/paper/te_corrected_diagnostics.py --revision-runs ~/choicebench-eacl-2026-revision/runs \
#       --hfbatch-runs ~/choicebench-eacl-hfbatch/runs --corrected-dir <dir of corrected files> --out-dir <dir>

from __future__ import annotations

import argparse
import glob
import hashlib
from pathlib import Path

import pandas as pd

from choicebench.analysis.derive_baseline_from_cyclic import derive_baseline_from_cyclic
from choicebench.analysis.paired_tests import mcnemar_exact_test
from choicebench.metrics.accuracy import Accuracy
from choicebench.metrics.mad import MAD
from choicebench.metrics.recall_rstd import RecallRStd

BENCH = {"mmlu": "mmlu", "arc": "arc_challenge"}
API_KEYS = {"gpt": "oa", "claude": "oa", "qwen": "qwen", "llama": "llama"}
LOCAL_KEYS = {"local_llama": "llama", "local_qwen": "qwen"}
PAIRS = ("baseline", "two_stage", "semantic_matching", "visible_llm_matcher")


def cell_metrics(df: pd.DataFrame) -> dict:
    out = {"rows": len(df), "coverage": float(df["parsed_choice"].notna().mean()) if len(df) else float("nan")}
    for metric in (Accuracy(), MAD(), RecallRStd()):
        out.update({k: v for k, v in metric.compute(df).items() if not k.startswith(("n_per_letter", "recall_"))})
    return out


def paired_rows(cell: dict, key: str, bench: str) -> list[dict]:
    rows = []
    te = cell["text_extraction_corrected"]
    comparisons = [("text_extraction_original", cell["text_extraction_original"])] + [(n, cell[n]) for n in PAIRS if cell.get(n) is not None]
    for name, other in comparisons:
        r = mcnemar_exact_test(te, other)
        rows.append({"model": key, "benchmark": bench, "a": "text_extraction_corrected", "b": name,
                     "n_paired": r["n_paired"], "a_correct_b_wrong": r["b"], "a_wrong_b_correct": r["c"], "p_value": r["p_value"]})
    return rows


def build_tables(cells: dict[tuple[str, str], dict]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metrics, grid, paired = [], [], []
    for (key, bench), cell in sorted(cells.items()):
        for variant in ("text_extraction_original", "text_extraction_corrected"):
            metrics.append({"model": key, "benchmark": bench, "variant": variant, **cell_metrics(cell[variant])})
        g = {"model": key, "benchmark": bench}
        for name in ("baseline", "two_stage", "semantic_matching", "text_extraction_original", "text_extraction_corrected", "visible_llm_matcher"):
            df = cell.get(name)
            if df is None:
                g[f"{name}_accuracy"] = g[f"{name}_conditional"] = g[f"{name}_coverage"] = float("nan")
                continue
            m = cell_metrics(df)
            g[f"{name}_accuracy"], g[f"{name}_conditional"], g[f"{name}_coverage"] = m["accuracy"], m["accuracy_conditional"], m["coverage"]
        grid.append(g)
        paired.extend(paired_rows(cell, key, bench))
    return pd.DataFrame(metrics), pd.DataFrame(grid), pd.DataFrame(paired)


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _by_method(pattern: str) -> dict[tuple[str, str], Path]:
    out = {}
    for f in glob.glob(pattern):
        h = pd.read_csv(f, nrows=1)
        out[(h.method_name.iloc[0], h.model_name.iloc[0])] = Path(f)
    return out


def load_cells(revision_runs: Path, hfbatch_runs: Path, corrected_dir: Path):
    cells, manifest = {}, []

    def read(path: Path, role: str, key: str, bench: str) -> pd.DataFrame:
        manifest.append({"model": key, "benchmark": bench, "role": role, "path": str(path), "sha256": _sha(path)})
        return pd.read_csv(path)

    for bench_short, bench in BENCH.items():
        for key, pre in API_KEYS.items():
            by = _by_method(str(revision_runs / f"eacl2026_{pre}_{bench_short}_core_methods/results/cond_*.csv"))
            hit = [(m, p) for (meth, m), p in by.items() if meth == "text_extraction" and key in m.lower()]
            if not hit:
                continue
            model_name, te_path = hit[0]
            d = revision_runs / f"eacl2026_{pre}_derived"
            cell = {"text_extraction_original": read(te_path, "text_extraction_original", key, bench),
                    "two_stage": read(by[("two_stage", model_name)], "two_stage", key, bench),
                    "text_extraction_corrected": read(corrected_dir / f"{bench_short}_text_extraction_corrected_{key}.csv", "text_extraction_corrected", key, bench)}
            for role, stem in (("baseline", "baseline"), ("semantic_matching", "semantic_matching"), ("visible_llm_matcher", "visible_llm_matcher")):
                cell[role] = read(d / f"{bench_short}_{stem}_{key}.csv", role, key, bench)
            cells[(key, bench)] = cell
        for key, m in LOCAL_KEYS.items():
            by = _by_method(str(hfbatch_runs / f"eacl2026_local_b32_{bench_short}_core_and_cyclic_{m}/results/cond_*.csv"))
            te = next(p for (meth, _), p in by.items() if meth == "text_extraction")
            cyc = next(p for (meth, _), p in by.items() if meth == "cyclic_permutation")
            cell = {"text_extraction_original": read(te, "text_extraction_original", key, bench),
                    "two_stage": read(next(p for (meth, _), p in by.items() if meth == "two_stage"), "two_stage", key, bench),
                    "text_extraction_corrected": read(corrected_dir / f"{bench_short}_text_extraction_corrected_{key}.csv", "text_extraction_corrected", key, bench),
                    "semantic_matching": read(hfbatch_runs / f"eacl2026_local_b32_derived/{bench_short}_semantic_matching_{m}.csv", "semantic_matching", key, bench),
                    "baseline": derive_baseline_from_cyclic(read(cyc, "cyclic_permutation(source of baseline)", key, bench), method_name="direct_mcq"),
                    "visible_llm_matcher": None}
            cells[(key, bench)] = cell
    return cells, pd.DataFrame(manifest)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--revision-runs", required=True, type=Path)
    p.add_argument("--hfbatch-runs", required=True, type=Path)
    p.add_argument("--corrected-dir", required=True, type=Path)
    p.add_argument("--out-dir", required=True, type=Path)
    a = p.parse_args()
    cells, manifest = load_cells(a.revision_runs, a.hfbatch_runs, a.corrected_dir)
    metrics, grid, paired = build_tables(cells)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    for name, df in (("te_metrics_original_vs_corrected", metrics), ("te_2x2_diagnostic", grid), ("te_paired_tests", paired), ("inputs_manifest", manifest)):
        df.to_csv(a.out_dir / f"{name}.csv", index=False)
    print(f"{len(cells)} cells -> {a.out_dir}")


if __name__ == "__main__":
    main()
