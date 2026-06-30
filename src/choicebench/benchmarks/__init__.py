# src/choicebench/benchmarks/__init__.py

"""Benchmark loading, normalization, and split generation."""

from choicebench.benchmarks import arc, hellaswag, mmlu, mmlu_pro, truthful_qa
from choicebench.benchmarks.registry import BENCHMARK_REGISTRY, benchmark, get_by_hf_path

__all__ = ["BENCHMARK_REGISTRY", "benchmark", "get_by_hf_path"]
