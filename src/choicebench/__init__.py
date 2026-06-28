# src/choicebench/__init__.py

"""choicebench — a framework for running reproducible MCQ evaluation experiments.

Supports multiple inference backends (API, HuggingFace, Dummy), pluggable
evaluation methods, disk-backed response caching, checkpointed batch runs,
and a validated YAML config schema. Methods and metrics can be built-in or
loaded dynamically via importlib paths.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("choicebench")
except PackageNotFoundError:
    __version__ = "0.0.0"
