"""Immutable experiment manifests and mutable run completion state."""

from __future__ import annotations

import json
import platform
import subprocess
from dataclasses import asdict
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Mapping

from choicebench import __version__
from choicebench.identity import CANONICALIZATION_VERSION, canonicalize, integrity_digest, short_id
from choicebench.infra.artifacts import FileLock, atomic_write_json

PROTOCOL_VERSION = "choicebench.protocol.v2"
MANIFEST_SCHEMA_VERSION = "choicebench.manifest.v2"
RUN_STATE_SCHEMA_VERSION = "choicebench.run-state.v2"
MANIFEST_FILENAME = "manifest.json"
RUN_STATE_FILENAME = "run_state.json"

PROTOCOL_V3_VERSION = "choicebench.protocol.v3"
MANIFEST_V3_SCHEMA_VERSION = "choicebench.manifest.v3"
RUN_STATE_V3_SCHEMA_VERSION = "choicebench.run-state.v3"
_MANIFEST_V3_PAYLOAD_KEYS = {
    "protocol_version",
    "canonicalization_version",
    "semantic_conditions",
    "realizations",
}


class ManifestCompatibilityError(RuntimeError):
    pass


def _sanitize_machine_paths(value: Any, key: str = "") -> Any:
    if isinstance(value, dict):
        return {name: _sanitize_machine_paths(item, name) for name, item in value.items()}
    if isinstance(value, list):
        return [_sanitize_machine_paths(item, key) for item in value]
    path_key = key.lower() in {"source", "path", "file", "directory", "dir"} or key.lower().endswith(
        ("_path", "_file", "_dir", "_directory")
    )
    if key == "model_name_or_path" and isinstance(value, str) and Path(value).is_absolute():
        return Path(value).name
    if path_key and isinstance(value, str) and value != "benchmark":
        return Path(value).name
    return value


def _dependency_versions() -> dict[str, str]:
    names = (
        "numpy", "pandas", "scipy", "pyyaml", "datasets", "openai", "anthropic",
        "google-genai", "groq", "torch", "transformers", "accelerate",
    )
    out: dict[str, str] = {}
    for name in names:
        try:
            out[name] = version(name)
        except PackageNotFoundError:
            out[name] = "not-installed"
    return out


def source_identity(cwd: Path | None = None) -> dict[str, Any]:
    if cwd is not None:
        root = Path(cwd)
        source_roots = (root / "src", root / "scripts")
    else:
        candidate = Path(__file__).resolve().parents[2]
        if (candidate / "pyproject.toml").exists():
            root = candidate
            source_roots = (root / "src", root / "scripts")
        else:
            root = Path(__file__).resolve().parent
            source_roots = (root,)
    source_files = sorted(
        [p for base in source_roots if base.exists()
         for p in base.rglob("*.py") if "__pycache__" not in p.parts]
        + ([root / "pyproject.toml"] if (root / "pyproject.toml").exists() else [])
    )
    tree_payload = [
        {"path": str(path.relative_to(root)), "sha256": integrity_digest(path.read_bytes().hex())}
        for path in source_files
    ]
    tree_digest = integrity_digest(tree_payload) if tree_payload else None
    is_checkout = (root / "pyproject.toml").exists() and (root / ".git").exists()
    if not is_checkout:
        return {
            "git_commit": None, "dirty": None, "dirty_tracked_digest": None,
            "source_tree_digest": tree_digest,
        }
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True,
            capture_output=True, check=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all", "--", "src", "scripts", "pyproject.toml"], cwd=root,
            text=True, capture_output=True, check=True,
        ).stdout
        diff = subprocess.run(
            ["git", "diff", "--binary", "HEAD", "--", "src", "scripts", "pyproject.toml"],
            cwd=root, capture_output=True, check=True,
        ).stdout
        return {
            "git_commit": commit,
            "dirty": bool(status.strip()),
            "dirty_tracked_digest": integrity_digest(diff.hex()) if diff else None,
            "source_tree_digest": tree_digest,
        }
    except (OSError, subprocess.CalledProcessError):
        return {"git_commit": None, "dirty": None, "dirty_tracked_digest": None, "source_tree_digest": tree_digest}


def environment_identity() -> dict[str, Any]:
    return {
        "choicebench_version": __version__,
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "dependencies": _dependency_versions(),
    }


def build_manifest_payload(
    *, config: Any, datasets: list[dict[str, Any]], prompts: dict[str, Any],
    models: list[dict[str, Any]], methods: list[dict[str, Any]],
    conditions: list[dict[str, Any]], calibrations: list[dict[str, Any]] | None = None,
    source: dict[str, Any] | None = None,
    environment: dict[str, Any] | None = None,
) -> dict[str, Any]:
    config_payload = asdict(config) if hasattr(config, "__dataclass_fields__") else config
    config_payload = canonicalize(_sanitize_machine_paths(config_payload))
    return canonicalize({
        "protocol_version": PROTOCOL_VERSION,
        "canonicalization_version": CANONICALIZATION_VERSION,
        "config": config_payload,
        "source": source if source is not None else source_identity(),
        "environment": environment if environment is not None else environment_identity(),
        "datasets": sorted(datasets, key=lambda item: item["selection_id"]),
        "calibrations": sorted(calibrations or [], key=lambda item: item["selection_id"]),
        "prompts": prompts,
        "models": sorted(models, key=lambda item: item["model_id"]),
        "methods": sorted(methods, key=lambda item: item["method_id"]),
        "conditions": sorted(conditions, key=lambda item: item["condition_id"]),
    })


def make_manifest(payload: dict[str, Any]) -> dict[str, Any]:
    payload = canonicalize(payload)
    payload.setdefault("protocol_version", PROTOCOL_VERSION)
    payload.setdefault("canonicalization_version", CANONICALIZATION_VERSION)
    identity_payload = {key: value for key, value in payload.items() if key != "config"}
    experiment_digest = integrity_digest(identity_payload)
    experiment_id = f"exp_{experiment_digest[:16]}"
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "experiment_id": experiment_id,
        "experiment_digest": experiment_digest,
        "payload_digest": integrity_digest(payload),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "payload": payload,
    }


def validate_manifest(manifest: dict[str, Any]) -> None:
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ManifestCompatibilityError("Unsupported or missing manifest schema version.")
    payload = manifest.get("payload")
    if not isinstance(payload, dict):
        raise ManifestCompatibilityError("Manifest payload is missing or invalid.")
    if payload.get("protocol_version") != PROTOCOL_VERSION:
        raise ManifestCompatibilityError(
            f"Unsupported protocol version {payload.get('protocol_version')!r}; expected {PROTOCOL_VERSION!r}."
        )
    if payload.get("canonicalization_version") != CANONICALIZATION_VERSION:
        raise ManifestCompatibilityError("Unsupported canonicalization version.")
    if manifest.get("payload_digest") != integrity_digest(payload):
        raise ManifestCompatibilityError("Manifest payload integrity check failed.")
    identity_payload = {key: value for key, value in payload.items() if key != "config"}
    digest = integrity_digest(identity_payload)
    experiment_id = f"exp_{digest[:16]}"
    if manifest.get("experiment_digest") != digest or manifest.get("experiment_id") != experiment_id:
        raise ManifestCompatibilityError("Manifest integrity check failed; payload or identity was modified.")
    conditions = payload.get("conditions", [])
    ids = [item.get("condition_id") for item in conditions]
    result_paths = [item.get("result_path") for item in conditions]
    if len(ids) != len(set(ids)) or len(result_paths) != len(set(result_paths)):
        raise ManifestCompatibilityError("Manifest contains duplicate condition IDs or result paths.")
    for item in conditions:
        condition_id = item.get("condition_id")
        expected = f"results/{condition_id}.csv"
        if (
            item.get("result_path") != expected
            or item.get("result_metadata_path") != f"results/{condition_id}.artifact.json"
            or item.get("checkpoint_path") != f"checkpoints/{condition_id}.json"
        ):
            raise ManifestCompatibilityError(f"Manifest contains an unsafe artifact path for {condition_id!r}.")
        if "gate_path" in item and item["gate_path"] != f"artifacts/{condition_id}/modal_k_gate.json":
            raise ManifestCompatibilityError(f"Manifest contains an unsafe gate path for {condition_id!r}.")
        prompt = payload.get("prompts")
        if prompt is not None and item.get("prompt_snapshot_path") != prompt.get("run_snapshot_path"):
            raise ManifestCompatibilityError(f"Manifest contains an unsafe prompt snapshot path for {condition_id!r}.")
    for section, prefix in (("datasets", "datasets"), ("calibrations", "calibrations")):
        for item in payload.get(section, []):
            expected = f"artifacts/{prefix}/{item.get('selection_id')}.csv"
            if item.get("run_snapshot_path") != expected:
                raise ManifestCompatibilityError(f"Manifest contains an unsafe {section} snapshot path.")
    prompt = payload.get("prompts")
    if prompt is not None:
        expected_prompt = f"artifacts/prompts/{prompt.get('prompt_id')}"
        if prompt.get("run_snapshot_path") != expected_prompt:
            raise ManifestCompatibilityError("Manifest contains an unsafe prompt snapshot path.")


def _differing_identity_sections(existing: dict[str, Any], candidate: dict[str, Any]) -> list[str]:
    """Name the identity-bearing payload sections that differ between two runs.

    Purely diagnostic: makes a resume refusal actionable ("environment"
    usually means a dependency/interpreter change; "source" a code change)
    instead of only reporting two opaque experiment digests.
    """
    keys = (set(existing) | set(candidate)) - {"config"}
    return sorted(
        key for key in keys
        if integrity_digest(existing.get(key)) != integrity_digest(candidate.get(key))
    )


def ensure_manifest(run_dir: Path, candidate: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    """Atomically create or verify a manifest. Returns (manifest, reused)."""
    run_dir = Path(run_dir)
    path = run_dir / MANIFEST_FILENAME
    lock = run_dir.parent / ".locks" / f"{run_dir.name}.manifest.lock"
    with FileLock(lock, f"create/verify manifest for run {run_dir.name}"):
        if path.exists():
            try:
                existing = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError) as exc:
                raise ManifestCompatibilityError(f"Existing manifest is unreadable: {path}: {exc}") from exc
            validate_manifest(existing)
            validate_manifest(candidate)
            if existing.get("experiment_digest") != candidate.get("experiment_digest"):
                changed = _differing_identity_sections(existing["payload"], candidate["payload"])
                raise ManifestCompatibilityError(
                    f"Run directory {run_dir} belongs to experiment "
                    f"{existing.get('experiment_id')}, not {candidate.get('experiment_id')}. "
                    f"Identity differs in: {', '.join(changed) or 'unknown section'}. "
                    "Use a new --run-id or pass --reset-run for an intentional replacement."
                )
            return existing, True
        meaningful = (
            [item for item in run_dir.iterdir() if not item.name.startswith(".manifest")]
            if run_dir.exists() else []
        )
        if meaningful:
            raise ManifestCompatibilityError(
                f"Legacy/nonempty run directory {run_dir} has no verified manifest. "
                "Use --reset-run or choose a new --run-id; unsafe legacy resume is refused."
            )
        run_dir.mkdir(parents=True, exist_ok=True)
        validate_manifest(candidate)
        atomic_write_json(path, candidate)
        return candidate, False


def initial_run_state(manifest: dict[str, Any]) -> dict[str, Any]:
    conditions = manifest["payload"]["conditions"]
    return {
        "schema_version": RUN_STATE_SCHEMA_VERSION,
        "experiment_id": manifest["experiment_id"],
        "conditions": {item["condition_id"]: {"status": "pending"} for item in conditions},
    }


def load_or_create_run_state(run_dir: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    path = Path(run_dir) / RUN_STATE_FILENAME
    if not path.exists():
        state = initial_run_state(manifest)
        write_run_state(run_dir, state)
        return state
    try:
        state = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ManifestCompatibilityError(f"Run state is unreadable: {path}: {exc}") from exc
    validate_run_state(state, manifest)
    return state


def load_run_state(run_dir: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    """Load and validate existing run state without creating mutable state."""
    path = Path(run_dir) / RUN_STATE_FILENAME
    if not path.is_file():
        raise ManifestCompatibilityError(f"Missing {RUN_STATE_FILENAME}.")
    try:
        state = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ManifestCompatibilityError(f"Unreadable {RUN_STATE_FILENAME}: {exc}") from exc
    validate_run_state(state, manifest)
    return state


def validate_run_state(state: dict[str, Any], manifest: dict[str, Any]) -> None:
    if state.get("schema_version") != RUN_STATE_SCHEMA_VERSION or state.get("experiment_id") != manifest["experiment_id"]:
        raise ManifestCompatibilityError(f"Run state does not belong to manifest {manifest['experiment_id']}.")
    expected = {c["condition_id"] for c in manifest["payload"]["conditions"]}
    if set(state.get("conditions", {})) != expected:
        raise ManifestCompatibilityError("Run state condition grid differs from the immutable manifest.")


def write_run_state(run_dir: Path, state: dict[str, Any]) -> None:
    path = Path(run_dir) / RUN_STATE_FILENAME
    atomic_write_json(path, canonicalize(state, redact_secrets=False))


def make_manifest_v3(
    payload: Mapping[str, Any], *, audit: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Build an immutable v3 manifest from semantic-condition/realization tables.

    Additive alongside make_manifest(); v2 manifests are unaffected. `payload`
    holds only identity-bearing content (no audit fields), so the experiment
    digest is simply the payload digest -- unlike v2 there is no "config" key
    to exclude.
    """
    if set(payload) != _MANIFEST_V3_PAYLOAD_KEYS:
        raise ManifestCompatibilityError("Manifest v3 payload fields are invalid.")
    payload = canonicalize(dict(payload))
    if payload["protocol_version"] != PROTOCOL_V3_VERSION:
        raise ManifestCompatibilityError(
            f"Unsupported protocol version {payload['protocol_version']!r}; "
            f"expected {PROTOCOL_V3_VERSION!r}."
        )
    if payload["canonicalization_version"] != CANONICALIZATION_VERSION:
        raise ManifestCompatibilityError("Unsupported canonicalization version.")
    experiment_digest = integrity_digest(payload)
    return {
        "schema_version": MANIFEST_V3_SCHEMA_VERSION,
        "experiment_id": f"exp_{experiment_digest[:16]}",
        "experiment_digest": experiment_digest,
        "payload_digest": integrity_digest(payload),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "payload": payload,
        "audit": canonicalize(dict(audit or {})),
    }


def validate_manifest_v3(manifest: Mapping[str, Any]) -> None:
    """Validate a v3 manifest, recomputing every child ID/digest from its
    own identity payload rather than trusting the stored value."""
    if manifest.get("schema_version") != MANIFEST_V3_SCHEMA_VERSION:
        raise ManifestCompatibilityError("Unsupported or missing manifest schema version.")
    payload = manifest.get("payload")
    if not isinstance(payload, dict) or set(payload) != _MANIFEST_V3_PAYLOAD_KEYS:
        raise ManifestCompatibilityError("Manifest v3 payload fields are invalid.")
    if payload.get("protocol_version") != PROTOCOL_V3_VERSION:
        raise ManifestCompatibilityError(
            f"Unsupported protocol version {payload.get('protocol_version')!r}; "
            f"expected {PROTOCOL_V3_VERSION!r}."
        )
    if payload.get("canonicalization_version") != CANONICALIZATION_VERSION:
        raise ManifestCompatibilityError("Unsupported canonicalization version.")
    if manifest.get("payload_digest") != integrity_digest(payload):
        raise ManifestCompatibilityError("Manifest payload integrity check failed.")
    experiment_digest = integrity_digest(payload)
    experiment_id = f"exp_{experiment_digest[:16]}"
    if (
        manifest.get("experiment_digest") != experiment_digest
        or manifest.get("experiment_id") != experiment_id
    ):
        raise ManifestCompatibilityError(
            "Manifest integrity check failed; payload or identity was modified."
        )

    semantic_conditions = payload["semantic_conditions"]
    if not isinstance(semantic_conditions, dict):
        raise ManifestCompatibilityError("Manifest semantic_conditions must be a mapping.")
    for condition_id, record in semantic_conditions.items():
        if not isinstance(record, dict):
            raise ManifestCompatibilityError("Manifest condition record is invalid.")
        recomputed_id = short_id("cond", record.get("identity"))
        recomputed_digest = integrity_digest(record.get("identity"))
        if (
            condition_id != record.get("condition_id")
            or condition_id != recomputed_id
            or record.get("condition_digest") != recomputed_digest
        ):
            raise ManifestCompatibilityError(
                f"Manifest condition {condition_id!r} ID/digest do not match its "
                "own identity payload."
            )

    realizations = payload["realizations"]
    if not isinstance(realizations, dict):
        raise ManifestCompatibilityError("Manifest realizations must be a mapping.")
    for realization_id, record in realizations.items():
        if not isinstance(record, dict):
            raise ManifestCompatibilityError("Manifest realization record is invalid.")
        recomputed_id = short_id("real", record.get("identity"))
        recomputed_digest = integrity_digest(record.get("identity"))
        if (
            realization_id != record.get("realization_id")
            or realization_id != recomputed_id
            or record.get("realization_digest") != recomputed_digest
        ):
            raise ManifestCompatibilityError(
                f"Manifest realization {realization_id!r} ID/digest do not match "
                "its own identity payload."
            )
        owning_condition = semantic_conditions.get(record.get("condition_id"))
        if (
            owning_condition is None
            or owning_condition.get("condition_digest") != record.get("condition_digest")
        ):
            raise ManifestCompatibilityError(
                f"Manifest realization {realization_id!r} condition binding is "
                "inconsistent with its declared semantic condition."
            )


def initial_run_state_v3(manifest: Mapping[str, Any]) -> dict[str, Any]:
    realizations = manifest["payload"]["realizations"]
    return {
        "schema_version": RUN_STATE_V3_SCHEMA_VERSION,
        "experiment_id": manifest["experiment_id"],
        "realizations": {realization_id: {"status": "pending"} for realization_id in realizations},
    }


def validate_run_state_v3(state: Mapping[str, Any], manifest: Mapping[str, Any]) -> None:
    if (
        state.get("schema_version") != RUN_STATE_V3_SCHEMA_VERSION
        or state.get("experiment_id") != manifest["experiment_id"]
    ):
        raise ManifestCompatibilityError(
            f"Run state does not belong to manifest {manifest['experiment_id']}."
        )
    expected = set(manifest["payload"]["realizations"])
    if set(state.get("realizations", {})) != expected:
        raise ManifestCompatibilityError(
            "Run state realization grid differs from the immutable manifest."
        )


def load_run_state_v3(run_dir: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Load and validate existing v3 run state without creating mutable state."""
    path = Path(run_dir) / RUN_STATE_FILENAME
    if not path.is_file():
        raise ManifestCompatibilityError(f"Missing {RUN_STATE_FILENAME}.")
    try:
        state = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ManifestCompatibilityError(f"Unreadable {RUN_STATE_FILENAME}: {exc}") from exc
    validate_run_state_v3(state, manifest)
    return state
