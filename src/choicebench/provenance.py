"""Runtime component and model provenance resolution."""

from __future__ import annotations

import hashlib
import importlib
import inspect
import os
import re
from importlib.metadata import PackageNotFoundError, packages_distributions, version
from pathlib import Path
from typing import Any

from choicebench.identity import digest_directory_tree, file_digest, integrity_digest

_COMMIT_RE = re.compile(r"^[0-9a-fA-F]{40,64}$")


class ProvenanceResolutionError(RuntimeError):
    pass


def directory_digest(root: Path) -> tuple[str, list[dict[str, Any]]]:
    """Hash every regular file in a local model directory by logical path/content."""
    root = Path(root).expanduser().resolve(strict=True)
    if root.is_file():
        record = {"path": root.name, "size": root.stat().st_size, "sha256": file_digest(root)}
        return integrity_digest([record]), [record]
    records = digest_directory_tree(
        root,
        exclude=lambda rel: any(part in {".git", "__pycache__"} for part in rel.parts),
        hash_fn=file_digest,
    )
    if not records:
        raise ProvenanceResolutionError(f"Local model path contains no files: {root}")
    return integrity_digest(records), records


def _cached_hf_commit(repo_id: str, revision: str, repo_type: str = "models") -> str | None:
    cache_root = Path(
        os.environ.get("HF_HUB_CACHE")
        or os.environ.get("HUGGINGFACE_HUB_CACHE")
        or Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
    ).expanduser()
    repo = cache_root / f"{repo_type}--{repo_id.replace('/', '--')}"
    if _COMMIT_RE.fullmatch(revision) and (repo / "snapshots" / revision).is_dir():
        return revision.lower()
    ref = repo / "refs" / revision
    if ref.is_file():
        value = ref.read_text(encoding="utf-8").strip()
        if _COMMIT_RE.fullmatch(value):
            return value.lower()
    return None


def resolve_hf_model_identity(model_name_or_path: str, revision: str | None) -> dict[str, Any]:
    candidate = Path(model_name_or_path).expanduser()
    if candidate.exists():
        digest, files = directory_digest(candidate)
        return {
            "kind": "local", "logical_name": candidate.name,
            "content_digest": digest, "file_count": len(files),
            "total_bytes": sum(item["size"] for item in files),
        }

    requested = revision or "main"
    resolved = requested.lower() if _COMMIT_RE.fullmatch(requested) else _cached_hf_commit(model_name_or_path, requested)
    if resolved is None:
        try:
            from huggingface_hub import HfApi
            resolved = HfApi().model_info(model_name_or_path, revision=requested).sha
        except Exception as exc:
            raise ProvenanceResolutionError(
                f"Cannot resolve Hugging Face model {model_name_or_path!r} revision {requested!r} "
                "to an immutable commit. Connect to the Hub once, cache that revision, or set "
                "models[].revision to a full commit SHA."
            ) from exc
    if not resolved or not _COMMIT_RE.fullmatch(resolved):
        raise ProvenanceResolutionError(
            f"Hugging Face returned a non-immutable revision for {model_name_or_path!r}: {resolved!r}."
        )
    return {
        "kind": "huggingface-hub", "repo_id": model_name_or_path,
        "requested_revision": revision, "resolved_commit": resolved.lower(),
    }


def resolve_hf_dataset_revision(dataset_id: str, revision: str | None) -> str:
    requested = revision or "main"
    resolved = requested.lower() if _COMMIT_RE.fullmatch(requested) else _cached_hf_commit(
        dataset_id, requested, "datasets"
    )
    if resolved is None:
        try:
            from huggingface_hub import HfApi
            resolved = HfApi().dataset_info(dataset_id, revision=requested).sha
        except Exception as exc:
            raise ProvenanceResolutionError(
                f"Cannot resolve Hugging Face dataset {dataset_id!r} revision {requested!r} "
                "to an immutable commit. Connect once or pass --revision with a full commit SHA."
            ) from exc
    if not resolved or not _COMMIT_RE.fullmatch(resolved):
        raise ProvenanceResolutionError(f"Dataset revision is not immutable: {resolved!r}.")
    return resolved.lower()


def implementation_identity(target: Any) -> dict[str, Any]:
    """Bind a configured class/function to its defining module bytes and package version."""
    module = inspect.getmodule(target)
    module_name = getattr(module, "__name__", getattr(target, "__module__", None))
    source_path = inspect.getsourcefile(target) or (getattr(module, "__file__", None) if module else None)
    record: dict[str, Any] = {
        "qualified_name": f"{module_name}:{getattr(target, '__qualname__', getattr(target, '__name__', type(target).__name__))}",
    }
    if source_path and Path(source_path).is_file():
        record["source_file"] = Path(source_path).name
        record["source_digest"] = file_digest(Path(source_path))
    top_level = module_name.split(".", 1)[0] if module_name else None
    distributions = packages_distributions().get(top_level, []) if top_level else []
    if distributions:
        distribution = sorted(distributions)[0]
        try:
            record["distribution"] = distribution
            record["distribution_version"] = version(distribution)
        except PackageNotFoundError:
            pass
    if top_level and top_level != "choicebench":
        try:
            package = importlib.import_module(top_level)
            roots = [Path(item) for item in getattr(package, "__path__", [])]
            tree_records = digest_directory_tree(
                roots,
                exclude=lambda rel: "__pycache__" in rel.parts or rel.suffix in {".pyc", ".pyo"},
                hash_fn=file_digest,
                record_path=lambda root, path: f"{root.name}/{path.relative_to(root).as_posix()}",
            )
            if tree_records:
                record["package_tree_digest"] = integrity_digest(tree_records)
                record["package_file_count"] = len(tree_records)
        except (ImportError, OSError):
            pass
    return record
