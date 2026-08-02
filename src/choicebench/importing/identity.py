"""Layered semantic, realization, lineage, and prediction-origin identities."""

from __future__ import annotations

from dataclasses import dataclass
import functools
import inspect
import math
from pathlib import PurePosixPath
import re
from typing import Any, Callable, Literal, Mapping, Sequence

from choicebench import __version__
from choicebench.datasets import dataset_content_digest, dataset_sample_identities
from choicebench.identity import (
    canonicalize,
    integrity_digest,
    is_credential_key,
    short_id,
)
from choicebench.importing.dataset_reference import ExpectedDataset
from choicebench.importing.schema import (
    CsvDialectSpec,
    ImportConditionSpec,
    ImportSpecError,
    ImportMethodSpec,
    ImportModelSpec,
    ImportPromptSpec,
    validate_csv_dialect_identity,
    validate_implementation_identity_record,
    validate_native_method_payload,
    validate_native_model_payload,
    validate_numeric_columns_identity,
    validate_option_mapping_identity,
)
from choicebench.provenance import implementation_identity
from choicebench.registry import METHOD_REGISTRY


class ImportIdentityError(ValueError):
    """Raised when an importer identity is cyclic, unsafe, or inconsistent."""


@dataclass(frozen=True)
class ImportSemanticRecords:
    dataset_artifact: Mapping[str, Any]
    selection: Mapping[str, Any]
    model: Mapping[str, Any]
    method: Mapping[str, Any]
    prompt: Mapping[str, Any]
    condition: Mapping[str, Any]


_CONDITION_KEYS = {
    "benchmark",
    "preflight",
    "model_id",
    "method_id",
    "prompt_id",
    "prompt_snapshot_path",
    "seed",
}
_DERIVATION_ORIGINS = {
    "native_execution",
    "external_import",
    "repair_overlay",
    "offline_transformation",
}
_PREDICTION_ORIGINS = {
    "native_inference",
    "external_historical_inference",
    "external_repair_inference",
}
_REALIZATION_KEYS = {
    "import_spec_digest",
    "sources",
    "expected_dataset",
    "importer_implementation",
    "parsing_policy",
    "validation",
    "evidence",
    "lineage_components",
    "result_origin",
    "parent_digests",
    "authorization_digest",
    "overlay",
}
_SOURCE_KEYS = {
    "source_id",
    "logical_path",
    "classification",
    "format",
    "format_version",
    "sha256",
    "provenance",
}
_SOURCE_PROVENANCE_KEYS = {
    "source_run_id",
    "source_repository",
    "source_commit",
    "notes_digest",
    "evidence_digest",
    "unknown_reasons",
}
_SOURCE_PROVENANCE_UNKNOWN_FIELDS = {
    "source_run_id",
    "source_repository",
    "source_commit",
}
_PARSING_POLICY_KEYS = {
    "dialect",
    "mapping",
    "null_values",
    "numeric_columns",
    "option_mapping",
    "extra_field_policy",
}
_DIALECT_KEYS = set(CsvDialectSpec.__dataclass_fields__)
_EVIDENCE_STATUSES = {
    "complete",
    "qualified",
    "partial",
    "malformed",
    "recoverable",
    "failed",
}
_SCOPE_DISPOSITIONS = {
    "included",
    "excluded_from_paper_matrix",
    "held",
    "superseded",
}
_LINEAGE_ID_RE = re.compile(r"^lin_[0-9a-f]{16}$")
_FORBIDDEN_IDENTITY_KEYS = {
    "audit_path",
    "absolute_path",
    "choicebench_home",
    "experiment_id",
    "final_csv_sha256",
    "final_result_digest",
    "host",
    "hostname",
    "import_date",
    "import_timestamp",
    "output_path",
    "output_root",
    "realization_id",
    "realization_digest",
    "report_path",
    "result_artifact_id",
    "result_artifact_digest",
    "result_sha256",
    "source_path",
    "temporary_path",
    "timestamp",
    "machine_id",
    "machine",
    "node",
    "operator",
    "platform",
    "recorded_at",
    "executed_at",
    "run_date",
    "cwd",
    "child_lineage_id",
}
_PROTOCOL_SETTING_KEYS = {"pride_modal_k_threshold"}
_NATIVE_DATASET_SOURCE_KEYS = {
    "generator",
    "seed",
    "hf_path",
    "hf_subset",
    "split",
    "requested_revision",
    "resolved_revision",
    "hf_fingerprint",
    "hf_dataset_info",
    "revision",
}
_METHOD_RUNTIME_PARAMETERS = {
    "self",
    "args",
    "kwargs",
    "backend",
    "method_name",
    "split_name",
    "prompt_version",
    "prompts_dir",
    "run_id",
    "temperature",
    "max_tokens",
    "seed",
    "perturbation_name",
    "model_label",
    "preflight_questions",
    "calibration_questions",
    "calibration_runs_dir",
    "modal_k",
    "gate_summary",
    "condition_id",
    "calibration_identity",
}
_DATASET_DERIVATION_KEYS = {
    "schema_version",
    "dataset_id",
    "reference_kind",
    "trust_label",
    "source_chain",
    "selection_source_id",
    "columns",
    "revision",
    "fingerprint",
    "declared_derivation",
    "selected_question_ids",
    "selection_semantics",
    "selection_unknown_reasons",
    "identity_mode",
    "limitations",
}


def _unknown(reason: str | None, field: str) -> dict[str, Any]:
    if not isinstance(reason, str) or not reason.strip():
        raise ImportIdentityError(f"Missing {field} requires an explicit unknown reason.")
    return {"value": None, "reason": reason}


def _nullable_semantic(value: Any, reason: str | None, field: str) -> Any:
    if value is not None:
        return value
    if isinstance(reason, str) and reason.strip().casefold() == "not applicable":
        return None
    return _unknown(reason, field)


def _validate_unknown_reason_contract(
    values: Mapping[str, Any], reasons: Mapping[str, str], where: str
) -> None:
    if not isinstance(reasons, Mapping):
        raise ImportIdentityError(f"{where} unknown reasons must be a mapping.")
    expected = {field for field, value in values.items() if value is None}
    if set(reasons) != expected or not all(
        isinstance(reason, str) and reason.strip() for reason in reasons.values()
    ):
        raise ImportIdentityError(
            f"{where} unknown reasons contradict known and missing values."
        )


def _validate_direct_declarations(
    condition: ImportConditionSpec,
    model: ImportModelSpec,
    method: ImportMethodSpec,
    prompt: ImportPromptSpec,
) -> None:
    _validate_stable_values(
        {
            "model_key": model.model_key,
            "display_name": model.display_name,
            "backend": model.backend,
            "provider": model.provider,
            "revision": model.revision,
            "method_key": method.method_key,
            "method_name": method.name,
            "prompt_key": prompt.prompt_key,
            "template_identity": prompt.template_identity,
        },
        "Semantic child declarations",
    )
    _validate_stable_parameters(
        model.effective_parameters, "Model effective parameters"
    )
    _validate_stable_parameters(
        condition.generation_parameters, "Condition generation parameters"
    )
    _validate_stable_parameters(
        method.effective_parameters, "Method effective parameters"
    )
    if condition.preflight_identity is not None:
        _validate_stable_parameters(
            condition.preflight_identity, "Condition preflight identity"
        )
    if method.implementation is not None:
        try:
            validated_implementation = validate_implementation_identity_record(
                method.implementation
            )
        except ImportSpecError as exc:
            raise ImportIdentityError(f"Invalid method implementation: {exc}") from exc
        if canonicalize(validated_implementation) != canonicalize(
            method.implementation
        ):
            raise ImportIdentityError(
                "Method implementation does not match the closed code identity schema."
            )
        _validate_stable_values(validated_implementation, "Method implementation")
    _validate_unknown_reason_contract(
        {
            "backend": model.backend,
            "provider": model.provider,
            "revision": model.revision,
        },
        model.unknown_reasons,
        "Model declaration",
    )
    _validate_unknown_reason_contract(
        {"implementation": method.implementation},
        method.unknown_reasons,
        "Method declaration",
    )
    _validate_unknown_reason_contract(
        {
            "seed": condition.seed,
            "calibration_identity": condition.calibration_identity,
            "preflight_identity": condition.preflight_identity,
        },
        condition.unknown_reasons,
        "Condition declaration",
    )
    prompt_values = (
        prompt.template_identity,
        prompt.template_digest,
        prompt.template_contents,
    )
    has_unknown_prompt = any(value is None for value in prompt_values)
    if has_unknown_prompt != (prompt.unknown_reason is not None) or (
        prompt.unknown_reason is not None
        and (not isinstance(prompt.unknown_reason, str) or not prompt.unknown_reason.strip())
    ):
        raise ImportIdentityError(
            "Prompt declaration unknown reason contradicts known and missing values."
        )
    if prompt.template_digest is not None:
        _validate_digest(prompt.template_digest, "prompt.template_digest")
    if (
        prompt.native_compatibility_identity is None
        and prompt.template_digest is not None
        and prompt.template_contents is not None
        and prompt.template_digest != integrity_digest(prompt.template_contents)
    ):
        raise ImportIdentityError(
            "Fallback prompt digest does not own the exact template contents."
        )


def _identity_record(prefix: str, payload: Mapping[str, Any]) -> tuple[str, str, dict]:
    identity = canonicalize(payload)
    return short_id(prefix, identity), integrity_digest(identity), identity


def _verify_claim(
    *, prefix: str, payload: Mapping[str, Any], claimed_id: Any, claimed_digest: Any
) -> tuple[str, str, dict]:
    if not isinstance(payload, Mapping):
        raise ImportIdentityError(f"Invalid validated native {prefix} identity payload.")
    identity_id, digest, identity = _identity_record(prefix, payload)
    if claimed_id != identity_id or claimed_digest != digest:
        raise ImportIdentityError(f"Invalid validated native {prefix} identity claim.")
    return identity_id, digest, identity


def _reject_forbidden(
    value: Any, where: str, *, allowed_path_keys: frozenset[str] = frozenset()
) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ImportIdentityError(f"{where} contains a non-string field name.")
            normalized = key.casefold()
            unsafe_alias = (
                normalized in _FORBIDDEN_IDENTITY_KEYS
                or (normalized not in allowed_path_keys and normalized.endswith("_path"))
                or normalized in {"path", "created_at", "updated_at", "import_state"}
                or "timestamp" in normalized
                or normalized.endswith(
                    ("realization_id", "result_artifact_id", "experiment_id")
                )
                or ("csv" in normalized and normalized.endswith(("sha256", "digest")))
            )
            if unsafe_alias:
                raise ImportIdentityError(f"{where} contains forbidden field {key!r}.")
            _reject_forbidden(item, where, allowed_path_keys=allowed_path_keys)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _reject_forbidden(item, where, allowed_path_keys=allowed_path_keys)


def _is_machine_path(value: str) -> bool:
    return bool(
        value.startswith(("/", "./", "../", "~/", "~\\", "\\\\"))
        or re.match(r"^[A-Za-z]:[\\/]", value)
    )


def _validate_stable_values(value: Any, where: str) -> None:
    _reject_forbidden(value, where)
    if isinstance(value, Mapping):
        for item in value.values():
            _validate_stable_values(item, where)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _validate_stable_values(item, where)
    elif isinstance(value, str) and _is_machine_path(value):
        raise ImportIdentityError(f"{where} contains a machine-local path value.")


def _validate_no_machine_path_values(value: Any, where: str) -> None:
    if isinstance(value, Mapping):
        for item in value.values():
            _validate_no_machine_path_values(item, where)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _validate_no_machine_path_values(item, where)
    elif isinstance(value, str) and _is_machine_path(value):
        raise ImportIdentityError(f"{where} contains a machine-local path value.")


def _validate_stable_parameters(value: Any, where: str) -> None:
    _reject_forbidden(value, where)
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str) or not key:
                raise ImportIdentityError(f"{where} keys must be non-empty strings.")
            normalized = key.casefold().replace("-", "_")
            if (
                is_credential_key(key)
                or normalized.endswith(
                    (
                        "_id",
                        "_ids",
                        "_digest",
                        "_digests",
                        "_checksum",
                        "_hash",
                        "_sha256",
                    )
                )
            ):
                raise ImportIdentityError(
                    f"{where} contains forbidden metadata field {key!r}."
                )
            _validate_stable_parameters(item, where)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _validate_stable_parameters(item, where)
    elif isinstance(value, str) and (
        _is_machine_path(value) or "/" in value or "\\" in value
    ):
        raise ImportIdentityError(f"{where} contains a path-like metadata value.")
    else:
        try:
            canonicalize(value)
        except (TypeError, ValueError) as exc:
            raise ImportIdentityError(f"{where} contains unsafe parameter data.") from exc


def _validate_condition_identity(value: Mapping[str, Any]) -> None:
    benchmark = _exact_mapping(
        value["benchmark"],
        {"name", "split", "selection_id", "artifact_id"},
        "Semantic condition benchmark",
    )
    for field in ("name", "split"):
        if not isinstance(benchmark[field], str) or not benchmark[field]:
            raise ImportIdentityError(
                f"Semantic condition benchmark {field} must be non-empty."
            )
    for field, prefix in (
        ("selection_id", "sel"),
        ("artifact_id", "ds"),
        ("model_id", "model"),
        ("method_id", "method"),
        ("prompt_id", "prompt"),
    ):
        identifier = benchmark[field] if field in benchmark else value[field]
        if not isinstance(identifier, str) or not re.fullmatch(
            rf"{prefix}_[0-9a-f]{{16}}", identifier
        ):
            raise ImportIdentityError(
                f"Semantic condition {field} is not a valid {prefix} identity."
            )
    preflight = value["preflight"]
    if preflight is not None:
        if not isinstance(preflight, Mapping):
            raise ImportIdentityError("Semantic condition preflight must be a mapping or null.")
        if set(preflight) == {"value", "reason"}:
            if preflight["value"] is not None or not isinstance(
                preflight["reason"], str
            ) or not preflight["reason"].strip():
                raise ImportIdentityError(
                    "Semantic condition unknown preflight must include a reason."
                )
        else:
            native = _exact_mapping(
                preflight,
                {"artifact_id", "selection_id", "split", "content_digest"},
                "Semantic condition preflight",
            )
            if not re.fullmatch(r"ds_[0-9a-f]{16}", str(native["artifact_id"])):
                raise ImportIdentityError("Semantic condition preflight artifact_id is invalid.")
            if not re.fullmatch(r"sel_[0-9a-f]{16}", str(native["selection_id"])):
                raise ImportIdentityError("Semantic condition preflight selection_id is invalid.")
            if not isinstance(native["split"], str) or not native["split"]:
                raise ImportIdentityError("Semantic condition preflight split is invalid.")
            _validate_digest(native["content_digest"], "preflight.content_digest")
    seed = value["seed"]
    if not (
        isinstance(seed, int)
        and not isinstance(seed, bool)
        or (
            isinstance(seed, Mapping)
            and set(seed) == {"value", "reason"}
            and seed.get("value") is None
            and isinstance(seed.get("reason"), str)
            and bool(seed["reason"].strip())
        )
    ):
        raise ImportIdentityError("Semantic condition seed is invalid.")
    if "protocol_settings" in value:
        if not isinstance(value["protocol_settings"], Mapping):
            raise ImportIdentityError("Semantic condition protocol_settings must be a mapping.")
        unexpected_protocol = sorted(
            set(value["protocol_settings"]) - _PROTOCOL_SETTING_KEYS
        )
        if unexpected_protocol:
            raise ImportIdentityError(
                "Semantic condition protocol_settings contains unsupported metadata "
                f"or protocol fields {unexpected_protocol}."
            )
        threshold = value["protocol_settings"].get("pride_modal_k_threshold")
        if threshold is not None and (
            isinstance(threshold, bool)
            or not isinstance(threshold, (int, float))
            or not math.isfinite(float(threshold))
        ):
            raise ImportIdentityError(
                "Semantic condition pride_modal_k_threshold must be finite numeric data."
            )
        _validate_stable_parameters(
            value["protocol_settings"], "Semantic condition protocol_settings"
        )


def _exact_mapping(value: Any, keys: set[str], where: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ImportIdentityError(f"{where} must be a mapping.")
    missing = sorted(keys - set(value))
    unexpected = sorted(set(value) - keys)
    if missing or unexpected:
        raise ImportIdentityError(
            f"{where} fields are invalid; missing={missing}, unexpected={unexpected}."
        )
    return value


def _digest_sequence(value: Any, field: str) -> list[str]:
    if not isinstance(value, (list, tuple)):
        raise ImportIdentityError(f"{field} must be a digest sequence.")
    result = []
    for index, digest in enumerate(value):
        checked = _validate_digest(digest, f"{field}[{index}]")
        assert checked is not None
        result.append(checked)
    if len(result) != len(set(result)):
        raise ImportIdentityError(f"{field} contains duplicate digest edges.")
    return sorted(result)


def _validate_expected_dataset_contract(dataset: ExpectedDataset) -> None:
    artifact_payload = dataset.artifact_payload
    if dataset.identity_mode == "imported_semantic_fallback":
        artifact = _exact_mapping(
            artifact_payload,
            {"schema_version", "benchmark", "split", "content_digest"},
            "Expected dataset fallback artifact",
        )
        if artifact["schema_version"] != "choicebench.semantic-dataset.v1":
            raise ImportIdentityError("Expected dataset fallback schema is invalid.")
        if (
            artifact["benchmark"] != dataset.benchmark_name
            or artifact["split"] != dataset.split
        ):
            raise ImportIdentityError(
                "Expected dataset fallback artifact conflicts with its declaration."
            )
    else:
        artifact = _exact_mapping(
            artifact_payload,
            {"spec", "content_digest", "source"},
            "Expected dataset native artifact",
        )
        spec = _exact_mapping(
            artifact["spec"],
            {
                "benchmark",
                "split",
                "hf_path",
                "hf_subset",
                "source_revision",
                "normalization_version",
                "transforms",
                "output_name",
            },
            "Expected dataset native artifact spec",
        )
        if spec["benchmark"] != dataset.benchmark_name or spec["split"] != dataset.split:
            raise ImportIdentityError(
                "Expected dataset native artifact conflicts with its declaration."
            )
        _validate_no_machine_path_values(
            spec, "Expected dataset native artifact spec"
        )
        if not isinstance(artifact["source"], Mapping):
            raise ImportIdentityError("Expected dataset native source is invalid.")
        source = artifact["source"]
        unexpected_source = sorted(set(source) - _NATIVE_DATASET_SOURCE_KEYS)
        if unexpected_source:
            raise ImportIdentityError(
                "Expected dataset native source contains unsupported audit or "
                f"source fields {unexpected_source}."
            )
        if "generator" in source and (
            not isinstance(source["generator"], str) or not source["generator"]
        ):
            raise ImportIdentityError("Expected dataset native source generator is invalid.")
        if isinstance(source.get("generator"), str) and _is_machine_path(
            source["generator"]
        ):
            raise ImportIdentityError(
                "Expected dataset native source generator contains a machine-local path."
            )
        if "seed" in source and (
            not isinstance(source["seed"], int) or isinstance(source["seed"], bool)
        ):
            raise ImportIdentityError("Expected dataset native source seed is invalid.")
        for field in (
            "hf_path",
            "hf_subset",
            "split",
            "requested_revision",
            "resolved_revision",
            "hf_fingerprint",
            "revision",
        ):
            if field in source and source[field] is not None and not isinstance(
                source[field], str
            ):
                raise ImportIdentityError(
                    f"Expected dataset native source {field} is invalid."
                )
            if isinstance(source.get(field), str) and _is_machine_path(source[field]):
                raise ImportIdentityError(
                    f"Expected dataset native source {field} contains a "
                    "machine-local path."
                )
        if "hf_dataset_info" in source:
            info = _exact_mapping(
                source["hf_dataset_info"],
                {"builder_name", "config_name", "version"},
                "Expected dataset native source hf_dataset_info",
            )
            if any(
                item is not None and not isinstance(item, str)
                for item in info.values()
            ):
                raise ImportIdentityError(
                    "Expected dataset native source hf_dataset_info is invalid."
                )
            _validate_no_machine_path_values(
                info, "Expected dataset native source hf_dataset_info"
            )
    if artifact["content_digest"] != dataset_content_digest(dataset.artifact_frame):
        raise ImportIdentityError(
            "Expected dataset artifact content digest does not own its frame."
        )

    semantics = _exact_mapping(
        dataset.selection_semantics,
        {"seed", "n_samples", "subject_filter"},
        "Expected dataset selection semantics",
    )
    selection = _exact_mapping(
        dataset.selection_payload,
        {
            "artifact_id",
            "content_digest",
            "sample_identities",
            "seed",
            "n_samples",
            "subject_filter",
        },
        "Expected dataset selection",
    )
    expected_selection = canonicalize(
        {
            "artifact_id": dataset.artifact_id,
            "content_digest": dataset_content_digest(dataset.frame),
            "sample_identities": dataset_sample_identities(dataset.frame),
            "seed": semantics["seed"],
            "n_samples": semantics["n_samples"],
            "subject_filter": sorted(semantics["subject_filter"]),
        }
    )
    if canonicalize(selection) != expected_selection:
        raise ImportIdentityError(
            "Expected dataset selection does not own its selected frame."
        )
    frame_question_ids = tuple(dataset.frame["question_id"].astype(str))
    if dataset.selected_question_ids != frame_question_ids:
        raise ImportIdentityError(
            "Expected dataset selected question IDs do not match its frame."
        )
    artifact_question_ids = tuple(dataset.artifact_frame["question_id"].astype(str))
    if len(artifact_question_ids) != len(set(artifact_question_ids)):
        raise ImportIdentityError("Expected dataset artifact has duplicate question IDs.")
    artifact_samples = dict(
        zip(
            artifact_question_ids,
            dataset_sample_identities(dataset.artifact_frame),
            strict=True,
        )
    )
    selected_samples = dataset_sample_identities(dataset.frame)
    for question_id, sample_identity in zip(
        frame_question_ids, selected_samples, strict=True
    ):
        if artifact_samples.get(question_id) != sample_identity:
            raise ImportIdentityError(
                "Expected dataset selected content is not owned by its artifact."
            )
    expected_unknowns = {
        field
        for field, value in (
            ("selection_seed", semantics["seed"]),
            ("selection_n_samples", semantics["n_samples"]),
        )
        if value is None
    }
    if set(dataset.selection_unknown_reasons) != expected_unknowns or not all(
        isinstance(reason, str) and reason.strip()
        for reason in dataset.selection_unknown_reasons.values()
    ):
        raise ImportIdentityError(
            "Expected dataset selection unknown reasons are contradictory."
        )

    if dataset.reference_kind not in {
        "independent_input_snapshot",
        "profile_derived_reference_snapshot",
    }:
        raise ImportIdentityError("Expected dataset reference kind is invalid.")
    if not isinstance(dataset.trust_label, str) or not dataset.trust_label.strip():
        raise ImportIdentityError("Expected dataset trust label is invalid.")
    if dataset.question_set_digest != integrity_digest(list(frame_question_ids)):
        raise ImportIdentityError(
            "Expected dataset question-set digest does not own its selected IDs."
        )
    derivation = _exact_mapping(
        dataset.derivation,
        _DATASET_DERIVATION_KEYS,
        "Expected dataset derivation",
    )
    expected_derivation_fields = {
        "schema_version": "choicebench.dataset-reference.v1",
        "dataset_id": dataset.dataset_id,
        "reference_kind": dataset.reference_kind,
        "trust_label": dataset.trust_label,
        "selected_question_ids": list(frame_question_ids),
        "selection_semantics": semantics,
        "selection_unknown_reasons": dict(dataset.selection_unknown_reasons),
        "identity_mode": dataset.identity_mode,
        "limitations": list(dataset.limitations),
    }
    for field, expected_value in expected_derivation_fields.items():
        if canonicalize(derivation[field]) != canonicalize(expected_value):
            raise ImportIdentityError(
                f"Expected dataset derivation {field} conflicts with its snapshot."
            )
    source_chain = derivation["source_chain"]
    if not isinstance(source_chain, (list, tuple)) or not source_chain:
        raise ImportIdentityError("Expected dataset derivation source chain is invalid.")
    seen_source_ids: set[str] = set()
    for index, item in enumerate(source_chain):
        source = _exact_mapping(
            item,
            {"source_id", "logical_path", "sha256"},
            f"Expected dataset derivation source_chain[{index}]",
        )
        if (
            not isinstance(source["source_id"], str)
            or not source["source_id"]
            or source["source_id"] in seen_source_ids
        ):
            raise ImportIdentityError(
                "Expected dataset derivation source IDs must be non-empty and unique."
            )
        logical_path = source["logical_path"]
        if not isinstance(logical_path, str):
            raise ImportIdentityError(
                "Expected dataset derivation logical path is invalid."
            )
        pure_path = PurePosixPath(logical_path)
        if (
            pure_path.is_absolute()
            or ".." in pure_path.parts
            or not pure_path.parts
            or _is_machine_path(logical_path)
        ):
            raise ImportIdentityError(
                "Expected dataset derivation logical path is unsafe."
            )
        _validate_digest(
            source["sha256"],
            f"Expected dataset derivation source_chain[{index}].sha256",
        )
        seen_source_ids.add(source["source_id"])
    if (
        not isinstance(derivation["selection_source_id"], str)
        or derivation["selection_source_id"] not in seen_source_ids
    ):
        raise ImportIdentityError(
            "Expected dataset derivation selection source is not in its source chain."
        )
    if not isinstance(derivation["columns"], Mapping) or not all(
        isinstance(key, str) and isinstance(value, str)
        for key, value in derivation["columns"].items()
    ):
        raise ImportIdentityError("Expected dataset derivation columns are invalid.")
    _validate_stable_values(
        derivation["revision"], "Expected dataset derivation revision"
    )
    _validate_stable_values(
        derivation["fingerprint"], "Expected dataset derivation fingerprint"
    )
    _validate_stable_values(
        derivation["declared_derivation"],
        "Expected dataset declared derivation",
    )
    if dataset.reference_kind == "profile_derived_reference_snapshot":
        declared_derivation = derivation["declared_derivation"]
        if not isinstance(declared_derivation, Mapping):
            raise ImportIdentityError(
                "Profile-derived reference declared derivation is invalid."
            )
        raw_groups = declared_derivation.get("independent_source_groups")
        if not isinstance(raw_groups, (list, tuple)):
            raise ImportIdentityError(
                "Profile-derived reference requires at least two independent "
                "source groups."
            )
        groups: list[tuple[str, ...]] = []
        for item in raw_groups:
            if not isinstance(item, (list, tuple)) or not item:
                raise ImportIdentityError(
                    "Profile-derived reference requires at least two independent "
                    "source groups."
                )
            groups.append(tuple(str(source_id) for source_id in item))
        if len(groups) < 2:
            raise ImportIdentityError(
                "Profile-derived reference requires at least two independent "
                "source groups."
            )
        flattened = [source_id for group in groups for source_id in group]
        if (
            len(flattened) != len(set(flattened))
            or set(flattened) != seen_source_ids
        ):
            raise ImportIdentityError(
                "Profile-derived reference independent source groups must be "
                "disjoint and cover every declared source."
            )
    expected_derivation_digest = integrity_digest(derivation)
    if dataset.derivation_digest != expected_derivation_digest:
        raise ImportIdentityError("Expected dataset derivation digest is inconsistent.")
    expected_snapshot_digest = integrity_digest(
        {
            "artifact_digest": dataset.artifact_digest,
            "selection_digest": dataset.selection_digest,
            "question_set_digest": dataset.question_set_digest,
            "derivation_digest": expected_derivation_digest,
            "reference_kind": dataset.reference_kind,
            "trust_label": dataset.trust_label,
        }
    )
    if dataset.snapshot_digest != expected_snapshot_digest:
        raise ImportIdentityError("Expected dataset snapshot digest is inconsistent.")


def _semantic_child_records(
    *,
    condition: ImportConditionSpec,
    dataset: ExpectedDataset,
    model: ImportModelSpec,
    method: ImportMethodSpec,
    prompt: ImportPromptSpec,
) -> tuple[dict, dict, dict, dict, dict]:
    dataset_record = {
        "artifact_id": dataset.artifact_id,
        "artifact_digest": dataset.artifact_digest,
        "identity": canonicalize(dataset.artifact_payload),
        "identity_mode": dataset.identity_mode,
    }
    if dataset.artifact_digest != integrity_digest(dataset_record["identity"]):
        raise ImportIdentityError("Expected dataset artifact digest is inconsistent.")
    if dataset.artifact_id != short_id("ds", dataset_record["identity"]):
        raise ImportIdentityError("Expected dataset artifact ID is inconsistent.")
    selection_record = {
        "selection_id": dataset.selection_id,
        "selection_digest": dataset.selection_digest,
        "identity": canonicalize(dataset.selection_payload),
    }
    if dataset.selection_digest != integrity_digest(selection_record["identity"]):
        raise ImportIdentityError("Expected dataset selection digest is inconsistent.")
    if dataset.selection_id != short_id("sel", selection_record["identity"]):
        raise ImportIdentityError("Expected dataset selection ID is inconsistent.")

    if model.native_compatibility_identity is not None:
        native = model.native_compatibility_identity
        if not isinstance(native, Mapping) or set(native) != {
            "payload",
            "digest",
            "model_id",
        }:
            raise ImportIdentityError("Invalid validated native model identity fields.")
        model_id, model_digest, model_payload = _verify_claim(
            prefix="model",
            payload=native["payload"],
            claimed_id=native["model_id"],
            claimed_digest=native["digest"],
        )
        if model.backend is None:
            raise ImportIdentityError(
                "An unknown model backend cannot be upgraded by a native identity claim."
            )
        try:
            validated_model_payload = validate_native_model_payload(model_payload)
        except ImportSpecError as exc:
            raise ImportIdentityError(f"Invalid validated native model payload: {exc}") from exc
        if canonicalize(validated_model_payload) != canonicalize(model_payload):
            raise ImportIdentityError(
                "Validated native model payload does not match the current backend schema."
            )
        model_payload = validated_model_payload
        for nullable_field in ("provider", "revision"):
            value = getattr(model, nullable_field)
            reason = model.unknown_reasons.get(nullable_field)
            if value is None and (
                not isinstance(reason, str)
                or reason.strip().casefold() != "not applicable"
            ):
                raise ImportIdentityError(
                    f"Unknown model {nullable_field} cannot be upgraded by a native "
                    "identity claim."
                )
        if model.backend is not None and model_payload.get("backend") != model.backend:
            raise ImportIdentityError(
                "Validated native model backend conflicts with its declaration."
            )
        if model.provider is not None and model_payload.get("provider") != model.provider:
            raise ImportIdentityError(
                "Validated native model provider conflicts with its declaration."
            )
        if (
            "model_name_or_path" in model_payload
            and model_payload["model_name_or_path"] != model.display_name
        ):
            raise ImportIdentityError(
                "Validated native model name conflicts with its declaration."
            )
        if model.backend == "huggingface":
            resolved_model = model_payload["model"]
            declared_name = (
                resolved_model["repo_id"]
                if resolved_model["kind"] == "huggingface-hub"
                else resolved_model["logical_name"]
            )
            if declared_name != model.display_name:
                raise ImportIdentityError(
                    "Validated native model name conflicts with its declaration."
                )
            if resolved_model.get("requested_revision") != model.revision:
                raise ImportIdentityError(
                    "Validated native model revision conflicts with its declaration."
                )
        elif model.revision is not None:
            raise ImportIdentityError(
                "The current native model identity cannot represent a declared "
                "revision; use the imported semantic fallback."
            )
        expected_generation = dict(model.effective_parameters)
        for name, value in condition.generation_parameters.items():
            if name in expected_generation and expected_generation[name] != value:
                raise ImportIdentityError(
                    f"Generation parameter {name!r} conflicts with model effective parameters."
                )
            expected_generation[name] = value
        native_generation = model_payload.get("generation_kwargs", {})
        if canonicalize(native_generation) != canonicalize(expected_generation):
            raise ImportIdentityError(
                "Condition generation parameters are not bound by the validated "
                "native model identity."
            )
        model_mode = "native_compatibility"
    else:
        effective_parameters = dict(model.effective_parameters)
        for name, value in condition.generation_parameters.items():
            if name in effective_parameters and effective_parameters[name] != value:
                raise ImportIdentityError(
                    f"Generation parameter {name!r} conflicts with model effective parameters."
                )
            effective_parameters[name] = value
        model_payload = {
            "schema_version": "choicebench.semantic-model.v1",
            "backend": (
                model.backend
                if model.backend is not None
                else _unknown(model.unknown_reasons.get("backend"), "model backend")
            ),
            "provider": (
                model.provider
                if model.provider is not None
                else _unknown(model.unknown_reasons.get("provider"), "model provider")
            ),
            "model": model.display_name,
            "revision": (
                model.revision
                if model.revision is not None
                else _unknown(model.unknown_reasons.get("revision"), "model revision")
            ),
            "effective_parameters": effective_parameters,
        }
        model_id, model_digest, model_payload = _identity_record("model", model_payload)
        model_mode = "imported_semantic_fallback"
    model_record = {
        "model_id": model_id,
        "model_digest": model_digest,
        "identity": model_payload,
        "identity_mode": model_mode,
    }

    if method.native_compatibility_identity is not None:
        native = method.native_compatibility_identity
        if not isinstance(native, Mapping) or set(native) != {
            "payload",
            "digest",
            "method_id",
        }:
            raise ImportIdentityError("Invalid validated native method identity fields.")
        method_id, method_digest, method_payload = _verify_claim(
            prefix="method",
            payload=native["payload"],
            claimed_id=native["method_id"],
            claimed_digest=native["digest"],
        )
        try:
            validated_method_payload = validate_native_method_payload(method_payload)
        except ImportSpecError as exc:
            raise ImportIdentityError(
                f"Invalid validated native method payload: {exc}"
            ) from exc
        if canonicalize(validated_method_payload) != canonicalize(method_payload):
            raise ImportIdentityError(
                "Validated native method payload does not match the current schema."
            )
        method_payload = validated_method_payload
        runner_cls = METHOD_REGISTRY.get(method.name)
        if runner_cls is None:
            raise ImportIdentityError(
                "Validated native method does not name a registered current runner."
            )
        current_effective_params: dict[str, Any] = {}
        accepted_parameters: set[str] = set()
        for name, parameter in inspect.signature(runner_cls.__init__).parameters.items():
            if name in _METHOD_RUNTIME_PARAMETERS:
                continue
            accepted_parameters.add(name)
            if parameter.default is not inspect.Parameter.empty:
                current_effective_params[name] = parameter.default
        unexpected_effective = sorted(
            set(method.effective_parameters) - accepted_parameters
        )
        if unexpected_effective:
            raise ImportIdentityError(
                "Validated native method declares parameters not accepted by its "
                f"registered runner: {unexpected_effective}."
            )
        current_effective_params.update(method.effective_parameters)
        declared_preflight = _nullable_semantic(
            condition.preflight_identity,
            condition.unknown_reasons.get("preflight_identity"),
            "method preflight identity",
        )
        expected_current_payload = canonicalize(
            {
                "name": method.name,
                "effective_params": current_effective_params,
                "preflight": declared_preflight,
                "implementation": implementation_identity(runner_cls),
            }
        )
        if canonicalize(method_payload) != expected_current_payload:
            raise ImportIdentityError(
                "Validated native method does not match its registered current runner."
            )
        if method.implementation is None:
            raise ImportIdentityError(
                "An unknown method implementation cannot be upgraded by a native "
                "identity claim."
            )
        if (
            method_payload.get("name") != method.name
            or canonicalize(method_payload.get("effective_params"))
            != canonicalize(method.effective_parameters)
            or canonicalize(method_payload.get("preflight"))
            != canonicalize(declared_preflight)
            or (
                method.implementation is not None
                and canonicalize(method_payload.get("implementation"))
                != canonicalize(method.implementation)
            )
        ):
            raise ImportIdentityError(
                "Validated native method identity conflicts with its semantic declaration."
            )
        method_mode = "native_compatibility"
    else:
        method_payload = {
            "schema_version": "choicebench.semantic-method.v1",
            "name": method.name,
            "effective_params": dict(method.effective_parameters),
            "preflight": _nullable_semantic(
                condition.preflight_identity,
                condition.unknown_reasons.get("preflight_identity"),
                "method preflight identity",
            ),
            "implementation": (
                method.implementation
                if method.implementation is not None
                else _unknown(
                    method.unknown_reasons.get("implementation"),
                    "method implementation",
                )
            ),
        }
        method_id, method_digest, method_payload = _identity_record(
            "method", method_payload
        )
        method_mode = "imported_semantic_fallback"
    method_record = {
        "method_id": method_id,
        "method_digest": method_digest,
        "identity": method_payload,
        "identity_mode": method_mode,
    }

    if prompt.native_compatibility_identity is not None:
        native = prompt.native_compatibility_identity
        if not isinstance(native, Mapping) or set(native) != {
            "prompt_id",
            "version",
            "files",
        }:
            raise ImportIdentityError("Invalid validated native prompt identity fields.")
        prompt_payload = canonicalize(
            {"version": native["version"], "files": native["files"]},
            redact_secrets=False,
        )
        if any(
            value is None
            for value in (
                prompt.template_identity,
                prompt.template_digest,
                prompt.template_contents,
            )
        ):
            raise ImportIdentityError(
                "An unknown prompt cannot be upgraded by a native identity claim."
            )
        native_files_raw = _exact_mapping(
            prompt_payload.get("files"),
            {"direct_mcq", "free_text", "option_matching"},
            "Validated native prompt files",
        )
        for name, raw_item in native_files_raw.items():
            item = _exact_mapping(
                raw_item, {"sha256", "content"}, f"Validated native prompt file {name}"
            )
            if not isinstance(item["content"], str) or item["sha256"] != integrity_digest(
                item["content"]
            ):
                raise ImportIdentityError(
                    f"Validated native prompt file {name!r} has an invalid content hash."
                )
        prompt_id = f"prompt_{integrity_digest(prompt_payload)[:16]}"
        if native["prompt_id"] != prompt_id:
            raise ImportIdentityError("Invalid validated native prompt identity claim.")
        declared_contents = prompt.template_contents
        native_files = prompt_payload.get("files")
        native_contents = (
            {
                name: item.get("content")
                for name, item in native_files.items()
                if isinstance(item, Mapping)
            }
            if isinstance(native_files, Mapping)
            else None
        )
        if (
            declared_contents is not None
            and canonicalize(declared_contents, redact_secrets=False)
            != canonicalize(native_contents, redact_secrets=False)
        ):
            raise ImportIdentityError(
                "Validated native prompt contents conflict with their declaration."
            )
        if (
            prompt.template_identity is not None
            and prompt.template_identity != prompt_payload.get("version")
        ):
            raise ImportIdentityError(
                "Validated native prompt identity conflicts with its declaration."
            )
        if (
            prompt.template_digest is not None
            and prompt.template_digest != integrity_digest(native_files)
        ):
            raise ImportIdentityError(
                "Validated native prompt digest conflicts with its declaration."
            )
        prompt_digest = integrity_digest(prompt_payload)
        prompt_mode = "native_compatibility"
    else:
        unknown = lambda field: _unknown(prompt.unknown_reason, f"prompt {field}")
        prompt_payload = {
            "schema_version": "choicebench.semantic-prompt.v1",
            "template_identity": (
                prompt.template_identity
                if prompt.template_identity is not None
                else unknown("template identity")
            ),
            "template_digest": (
                prompt.template_digest
                if prompt.template_digest is not None
                else unknown("template digest")
            ),
            "template_contents": (
                prompt.template_contents
                if prompt.template_contents is not None
                else unknown("template contents")
            ),
        }
        prompt_payload = canonicalize(prompt_payload, redact_secrets=False)
        prompt_digest = integrity_digest(prompt_payload)
        prompt_id = f"prompt_{prompt_digest[:16]}"
        prompt_mode = "imported_semantic_fallback"
    prompt_record = {
        "prompt_id": prompt_id,
        "prompt_digest": prompt_digest,
        "identity": prompt_payload,
        "identity_mode": prompt_mode,
    }
    return dataset_record, selection_record, model_record, method_record, prompt_record


def make_semantic_condition(
    *, identity: Mapping[str, Any], fields: Mapping[str, Any]
) -> dict[str, Any]:
    """Create one scientific condition without realization provenance."""
    allowed = set(_CONDITION_KEYS)
    if "protocol_settings" in identity:
        allowed.add("protocol_settings")
    if set(identity) != allowed:
        raise ImportIdentityError("Semantic condition identity fields are invalid.")
    prompt_id = identity.get("prompt_id")
    expected_prompt_path = f"artifacts/prompts/{prompt_id}"
    if identity.get("prompt_snapshot_path") != expected_prompt_path:
        raise ImportIdentityError(
            "Semantic condition prompt snapshot path must be derived from prompt_id."
        )
    _validate_condition_identity(identity)
    _reject_forbidden(
        identity,
        "Semantic condition identity",
        allowed_path_keys=frozenset({"prompt_snapshot_path"}),
    )
    payload = canonicalize(identity)
    condition_id = short_id("cond", payload)
    condition_digest = integrity_digest(payload)
    if not isinstance(fields, Mapping) or not set(fields) <= {
        "condition_key",
        "expected_question_ids",
    }:
        raise ImportIdentityError(
            "Semantic condition fields may contain only condition_key and "
            "expected_question_ids."
        )
    extra = canonicalize(fields)
    return {
        "condition_id": condition_id,
        "condition_digest": condition_digest,
        "identity": payload,
        **extra,
    }


def build_import_semantic_identity(
    *,
    condition: ImportConditionSpec,
    dataset: ExpectedDataset,
    model: ImportModelSpec,
    method: ImportMethodSpec,
    prompt: ImportPromptSpec,
) -> ImportSemanticRecords:
    """Build semantic children and a current-shape scientific condition."""
    _validate_direct_declarations(condition, model, method, prompt)
    if dataset.identity_mode not in {
        "imported_semantic_fallback",
        "native_compatibility",
    }:
        raise ImportIdentityError(
            f"Unsupported expected dataset identity mode {dataset.identity_mode!r}."
        )
    _validate_expected_dataset_contract(dataset)
    references = (
        ("dataset_id", condition.dataset_id, dataset.dataset_id),
        ("model_key", condition.model_key, model.model_key),
        ("method_key", condition.method_key, method.method_key),
        ("prompt_key", condition.prompt_key, prompt.prompt_key),
    )
    for field, declared, supplied in references:
        if declared != supplied:
            raise ImportIdentityError(
                f"Condition {field}={declared!r} does not match supplied child {supplied!r}."
            )
    if tuple(condition.expected_question_ids) != dataset.selected_question_ids:
        raise ImportIdentityError("Condition question set does not match dataset selection.")
    artifact_identity = dataset.artifact_payload
    if dataset.identity_mode == "imported_semantic_fallback":
        if (
            artifact_identity.get("benchmark") != dataset.benchmark_name
            or artifact_identity.get("split") != dataset.split
        ):
            raise ImportIdentityError(
                "Expected dataset description conflicts with artifact identity."
            )
    else:
        artifact_spec = artifact_identity.get("spec")
        if (
            not isinstance(artifact_spec, Mapping)
            or artifact_spec.get("benchmark") != dataset.benchmark_name
            or artifact_spec.get("split") != dataset.split
        ):
            raise ImportIdentityError(
                "Expected dataset description conflicts with native artifact identity."
            )
    if dataset.selection_payload.get("artifact_id") != dataset.artifact_id:
        raise ImportIdentityError("Dataset selection points to a different artifact.")
    dataset_record, selection, model_record, method_record, prompt_record = (
        _semantic_child_records(
            condition=condition,
            dataset=dataset,
            model=model,
            method=method,
            prompt=prompt,
        )
    )
    seed = (
        condition.seed
        if condition.seed is not None
        else _unknown(condition.unknown_reasons.get("seed"), "condition seed")
    )
    preflight = _nullable_semantic(
        condition.calibration_identity,
        condition.unknown_reasons.get("calibration_identity"),
        "condition calibration identity",
    )
    condition_payload: dict[str, Any] = {
        "benchmark": {
            "name": dataset.benchmark_name,
            "split": dataset.split,
            "selection_id": selection["selection_id"],
            "artifact_id": dataset_record["artifact_id"],
        },
        "preflight": preflight,
        "model_id": model_record["model_id"],
        "method_id": method_record["method_id"],
        "prompt_id": prompt_record["prompt_id"],
        "prompt_snapshot_path": f"artifacts/prompts/{prompt_record['prompt_id']}",
        "seed": seed,
    }
    if condition.protocol_settings:
        condition_payload["protocol_settings"] = condition.protocol_settings
    condition_record = make_semantic_condition(
        identity=condition_payload,
        fields={
            "condition_key": condition.condition_key,
            "expected_question_ids": list(condition.expected_question_ids),
        },
    )
    return ImportSemanticRecords(
        dataset_artifact=dataset_record,
        selection=selection,
        model=model_record,
        method=method_record,
        prompt=prompt_record,
        condition=condition_record,
    )


def _validate_digest(value: Any, field: str, *, optional: bool = False) -> str | None:
    if optional and value is None:
        return None
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ImportIdentityError(f"{field} must be a lowercase SHA-256 digest.")
    return value


@functools.lru_cache(maxsize=None)
def _cached_global_implementation_identity(value: Any) -> dict[str, Any]:
    """Cache `implementation_identity` per referenced global object.

    `implementation_identity` walks and hashes a non-`choicebench` package's
    entire file tree; without caching, every callable that references the
    same third-party helper (e.g. a bare `from pandas import isna`) would
    repeat that walk on every identity computation. The referenced object's
    defining files do not change within a process, so caching by object
    identity (functions/classes hash by identity) is safe and keeps repeated
    identity computations cheap.
    """
    return validate_implementation_identity_record(implementation_identity(value))


def _resolved_global_bindings(target: Callable[..., Any], where: str) -> dict[str, Any]:
    """Bind behavior-affecting global values the callable's code resolves by name.

    Function/class globals (helpers) are bound by their own defining-file
    identity rather than skipped: `implementation_identity` only hashes a
    callable's own defining file, so a helper imported from another module is
    never covered by the calling callable's own source digest. Binding it
    shallowly (not expanding its own resolved globals) closes that gap without
    risking unbounded or cyclic expansion through mutually referencing
    helpers. Modules and dunder names are excluded as irrelevant runtime
    state; anything else that cannot be represented stably fails closed
    rather than being silently dropped.
    """
    code = getattr(target, "__code__", None)
    global_ns = getattr(target, "__globals__", None)
    if code is None or global_ns is None:
        return {}
    bindings: dict[str, Any] = {}
    for name in sorted(set(code.co_names)):
        if name.startswith("__") or name not in global_ns:
            continue
        value = global_ns[name]
        if value is target or inspect.ismodule(value):
            continue
        if inspect.isfunction(value) or inspect.isclass(value):
            try:
                bindings[name] = _cached_global_implementation_identity(value)
            except (ImportSpecError, OSError, TypeError, ValueError) as exc:
                raise ImportIdentityError(
                    f"{where} references global {name!r} with no inspectable "
                    "implementation identity."
                ) from exc
            continue
        if callable(value):
            raise ImportIdentityError(
                f"{where} references global {name!r} that is an uninspectable "
                "stateful callable."
            )
        try:
            bindings[name] = canonicalize(value)
        except (TypeError, ValueError) as exc:
            raise ImportIdentityError(
                f"{where} references global {name!r} with an unsupported value."
            ) from exc
    return bindings


def _runtime_callable_record(target: Callable[..., Any], where: str) -> dict[str, Any]:
    if inspect.ismethod(target) or not (
        inspect.isfunction(target) or inspect.isclass(target)
    ):
        raise ImportIdentityError(
            f"{where} must be a stateless module function or class, not a bound "
            "method or stateful callable instance."
        )
    if inspect.isfunction(target) and target.__closure__:
        raise ImportIdentityError(f"{where} cannot be a stateful closure.")
    qualified_name = getattr(target, "__qualname__", "")
    if "<lambda>" in qualified_name or "<locals>" in qualified_name:
        raise ImportIdentityError(
            f"{where} must be a uniquely addressable named module callable."
        )
    try:
        callable_source = inspect.getsource(target)
    except (OSError, TypeError) as exc:
        raise ImportIdentityError(
            f"{where} lacks uniquely inspectable callable source."
        ) from exc
    callable_payload = {
        "source": callable_source,
        "defaults": getattr(target, "__defaults__", None),
        "keyword_defaults": getattr(target, "__kwdefaults__", None),
        "resolved_globals": _resolved_global_bindings(target, where),
    }
    try:
        callable_digest = integrity_digest(callable_payload)
    except (TypeError, ValueError) as exc:
        raise ImportIdentityError(
            f"{where} has unsupported callable defaults."
        ) from exc
    try:
        record = validate_implementation_identity_record(
            implementation_identity(target)
        )
    except (ImportSpecError, OSError, TypeError, ValueError) as exc:
        raise ImportIdentityError(
            f"{where} lacks an inspectable runtime code identity."
        ) from exc
    if "source_digest" not in record:
        raise ImportIdentityError(f"{where} lacks an inspectable source digest.")
    record["callable_digest"] = callable_digest
    return record


def _runtime_parameter_schema(
    target: Callable[..., Any],
) -> tuple[set[str], set[str]]:
    try:
        signature = inspect.signature(target)
    except (TypeError, ValueError) as exc:
        raise ImportIdentityError(
            "Lineage runtime implementation has no inspectable parameter schema."
        ) from exc
    allowed = {
        name
        for name, parameter in signature.parameters.items()
        if parameter.kind is inspect.Parameter.KEYWORD_ONLY
    }
    required = {
        name
        for name, parameter in signature.parameters.items()
        if parameter.kind is inspect.Parameter.KEYWORD_ONLY
        and parameter.default is inspect.Parameter.empty
    }
    return allowed, required


def make_lineage_component(
    *,
    operation_type: str,
    question_id: str,
    parent_digests: Sequence[str],
    source_digests: Sequence[str],
    authorization_digest: str | None,
    implementation: Callable[..., Any] | Mapping[str, Any],
    implementation_mode: Literal["runtime_callable", "declared_external"] = (
        "runtime_callable"
    ),
    parameters: Mapping[str, Any],
    input_digest: str,
    preownership_output_digest: str,
    prediction_origin: str,
) -> dict[str, Any]:
    """Build a parent-derived lineage node without any child/self edge."""
    if not isinstance(operation_type, str) or not operation_type:
        raise ImportIdentityError("Lineage operation_type must be non-empty.")
    if not isinstance(question_id, str) or not question_id:
        raise ImportIdentityError("Lineage question_id must be non-empty.")
    if prediction_origin not in _PREDICTION_ORIGINS:
        raise ImportIdentityError(f"Invalid prediction origin {prediction_origin!r}.")
    for index, digest in enumerate(parent_digests):
        _validate_digest(digest, f"parent_digests[{index}]")
    for index, digest in enumerate(source_digests):
        _validate_digest(digest, f"source_digests[{index}]")
    if len(parent_digests) != len(set(parent_digests)):
        raise ImportIdentityError("Lineage parent_digests contains duplicate edges.")
    if len(source_digests) != len(set(source_digests)):
        raise ImportIdentityError("Lineage source_digests contains duplicate edges.")
    _validate_digest(authorization_digest, "authorization_digest", optional=True)
    _validate_digest(input_digest, "input_digest")
    _validate_digest(preownership_output_digest, "preownership_output_digest")
    if implementation_mode == "runtime_callable":
        if not callable(implementation):
            raise ImportIdentityError(
                "Lineage runtime implementation must be an inspectable callable."
            )
        validated_implementation = _runtime_callable_record(
            implementation, "Lineage runtime implementation"
        )
        allowed_parameters, required_parameters = _runtime_parameter_schema(
            implementation
        )
        unexpected_parameters = sorted(set(parameters) - allowed_parameters)
        if unexpected_parameters:
            raise ImportIdentityError(
                "Lineage parameters contain undeclared field(s) "
                f"{unexpected_parameters}."
            )
        missing_parameters = sorted(required_parameters - set(parameters))
        if missing_parameters:
            raise ImportIdentityError(
                "Lineage parameters are missing required field(s) "
                f"{missing_parameters}."
            )
    elif implementation_mode == "declared_external":
        try:
            validated_implementation = validate_implementation_identity_record(
                implementation
            )
        except ImportSpecError as exc:
            raise ImportIdentityError(
                f"Invalid declared external lineage implementation: {exc}"
            ) from exc
        source_digest = validated_implementation.get("source_digest")
        if source_digest is None or source_digest not in source_digests:
            raise ImportIdentityError(
                "Declared external lineage implementation source digest must be "
                "present in source_digests."
            )
        if parameters:
            raise ImportIdentityError(
                "Declared external lineage implementations cannot self-declare "
                "unverified parameters."
            )
    else:
        raise ImportIdentityError(
            f"Invalid lineage implementation_mode {implementation_mode!r}."
        )
    implementation_record = {
        "identity_mode": implementation_mode,
        "identity": validated_implementation,
    }
    _validate_stable_values(implementation_record, "Lineage implementation")
    _validate_stable_parameters(parameters, "Lineage parameters")
    payload = canonicalize(
        {
            "schema_version": "choicebench.lineage-component.v1",
            "operation_type": operation_type,
            "question_id": question_id,
            "parent_digests": sorted(parent_digests),
            "source_digests": sorted(source_digests),
            "authorization_digest": authorization_digest,
            "implementation": implementation_record,
            "parameters": parameters,
            "input_digest": input_digest,
            "preownership_output_digest": preownership_output_digest,
            "prediction_origin": prediction_origin,
        }
    )
    return {
        "lineage_id": short_id("lin", payload),
        "lineage_digest": integrity_digest(payload),
        "identity": payload,
    }


def make_result_origin(
    *,
    derivation_origin: Literal[
        "native_execution", "external_import", "repair_overlay", "offline_transformation"
    ],
    row_assignments: Sequence[tuple[str, str, str]],
) -> dict[str, Any]:
    """Record constituent and ordered per-row prediction origins."""
    if derivation_origin not in _DERIVATION_ORIGINS:
        raise ImportIdentityError(f"Invalid derivation origin {derivation_origin!r}.")
    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    counts: dict[str, int] = {}
    for index, assignment in enumerate(row_assignments):
        if not isinstance(assignment, (list, tuple)) or len(assignment) != 3:
            raise ImportIdentityError(
                f"Result origin row assignment {index} must contain exactly "
                "question_id, prediction_origin, and prediction_lineage_id."
            )
        question_id, prediction_origin, lineage_id = assignment
        if not isinstance(question_id, str) or not question_id or question_id in seen:
            raise ImportIdentityError("Result origin question IDs must be non-empty and unique.")
        if prediction_origin not in _PREDICTION_ORIGINS:
            raise ImportIdentityError(f"Invalid prediction origin {prediction_origin!r}.")
        if not isinstance(lineage_id, str) or not _LINEAGE_ID_RE.fullmatch(lineage_id):
            raise ImportIdentityError(
                "Result origin prediction_lineage_id must be a lineage-component ID."
            )
        seen.add(question_id)
        counts[prediction_origin] = counts.get(prediction_origin, 0) + 1
        rows.append(
            {
                "question_id": question_id,
                "prediction_origin": prediction_origin,
                "prediction_lineage_id": lineage_id,
            }
        )
    rows = canonicalize(rows)
    lineage_projection = [
        {
            "prediction_origin": row["prediction_origin"],
            "prediction_lineage_id": row["prediction_lineage_id"],
        }
        for row in rows
    ]
    identity = {
        "derivation_origin": derivation_origin,
        "prediction_origins": sorted(counts),
        "prediction_origin_counts": {key: counts[key] for key in sorted(counts)},
        "row_assignments": rows,
        "ordered_row_origin_digest": integrity_digest(rows),
        "lineage_component_origin_digest": integrity_digest(lineage_projection),
    }
    return {
        "origin_id": short_id("origin", identity),
        "origin_digest": integrity_digest(identity),
        **identity,
    }


def _validate_lineage_component_record(
    value: Any, *, runtime_callable: Callable[..., Any] | None
) -> dict[str, Any]:
    record = _exact_mapping(
        value,
        {"lineage_id", "lineage_digest", "identity"},
        "Realization lineage component",
    )
    identity = _exact_mapping(
        record["identity"],
        {
            "schema_version",
            "operation_type",
            "question_id",
            "parent_digests",
            "source_digests",
            "authorization_digest",
            "implementation",
            "parameters",
            "input_digest",
            "preownership_output_digest",
            "prediction_origin",
        },
        "Realization lineage component identity",
    )
    if identity["schema_version"] != "choicebench.lineage-component.v1":
        raise ImportIdentityError("Realization lineage component schema is invalid.")
    for field in ("operation_type", "question_id"):
        if not isinstance(identity[field], str) or not identity[field]:
            raise ImportIdentityError(
                f"Realization lineage component {field} must be non-empty."
            )
    normalized_parents = _digest_sequence(
        identity["parent_digests"], "lineage.parent_digests"
    )
    normalized_sources = _digest_sequence(
        identity["source_digests"], "lineage.source_digests"
    )
    authorization_digest = _validate_digest(
        identity["authorization_digest"],
        "lineage.authorization_digest",
        optional=True,
    )
    implementation = _exact_mapping(
        identity["implementation"],
        {"identity_mode", "identity"},
        "Realization lineage implementation",
    )
    if implementation["identity_mode"] not in {
        "runtime_callable",
        "declared_external",
    }:
        raise ImportIdentityError(
            "Realization lineage implementation identity mode is invalid."
        )
    try:
        implementation_identity_record = validate_implementation_identity_record(
            implementation["identity"]
        )
    except ImportSpecError as exc:
        raise ImportIdentityError(
            f"Realization lineage implementation is invalid: {exc}"
        ) from exc
    source_digest = implementation_identity_record.get("source_digest")
    if source_digest is None:
        raise ImportIdentityError(
            "Realization lineage implementation lacks a source digest."
        )
    if implementation["identity_mode"] == "runtime_callable":
        if runtime_callable is None:
            raise ImportIdentityError(
                "Realization runtime lineage requires its registered callable."
            )
        runtime_identity = _runtime_callable_record(
            runtime_callable, "Realization lineage runtime implementation"
        )
        if canonicalize(runtime_identity) != canonicalize(
            implementation_identity_record
        ):
            raise ImportIdentityError(
                "Realization runtime lineage implementation does not match its "
                "registered callable."
            )
    elif runtime_callable is not None:
        raise ImportIdentityError(
            "Declared external lineage cannot claim a runtime callable."
        )
    if (
        implementation["identity_mode"] == "declared_external"
        and source_digest not in normalized_sources
    ):
        raise ImportIdentityError(
            "Declared external lineage implementation is not owned by its source edge."
        )
    parameters = identity["parameters"]
    if not isinstance(parameters, Mapping):
        raise ImportIdentityError("Realization lineage parameters must be a mapping.")
    _validate_stable_parameters(parameters, "Realization lineage parameters")
    input_digest = _validate_digest(identity["input_digest"], "lineage.input_digest")
    output_digest = _validate_digest(
        identity["preownership_output_digest"],
        "lineage.preownership_output_digest",
    )
    prediction_origin = identity["prediction_origin"]
    if prediction_origin not in _PREDICTION_ORIGINS:
        raise ImportIdentityError("Realization lineage prediction origin is invalid.")
    normalized_identity = canonicalize(
        {
            "schema_version": "choicebench.lineage-component.v1",
            "operation_type": identity["operation_type"],
            "question_id": identity["question_id"],
            "parent_digests": normalized_parents,
            "source_digests": normalized_sources,
            "authorization_digest": authorization_digest,
            "implementation": {
                "identity_mode": implementation["identity_mode"],
                "identity": implementation_identity_record,
            },
            "parameters": parameters,
            "input_digest": input_digest,
            "preownership_output_digest": output_digest,
            "prediction_origin": prediction_origin,
        }
    )
    expected_digest = integrity_digest(normalized_identity)
    expected_id = short_id("lin", normalized_identity)
    if (
        record["lineage_digest"] != expected_digest
        or record["lineage_id"] != expected_id
        or canonicalize(record["identity"]) != normalized_identity
    ):
        raise ImportIdentityError(
            "Realization lineage component identity or digest is inconsistent."
        )
    return {
        "lineage_id": expected_id,
        "lineage_digest": expected_digest,
        "identity": normalized_identity,
    }


def importer_implementation_identity(
    *, adapter: Callable[..., Any], validator: Callable[..., Any]
) -> dict[str, Any]:
    """Bind installed ChoiceBench plus importer-core, adapter, and validator code."""
    records: dict[str, Mapping[str, Any]] = {}
    for role, target in (("adapter", adapter), ("validator", validator)):
        records[role] = _runtime_callable_record(
            target, f"Importer {role} implementation"
        )
    importer_record = _runtime_callable_record(
        _identity_record, "Importer core implementation"
    )
    return canonicalize(
        {
            "schema_version": "choicebench.importer-implementation.v1",
            "package": {"name": "choicebench", "version": __version__},
            "importer": importer_record,
            "adapter": records["adapter"],
            "validator": records["validator"],
        }
    )


def _validate_result_origin_record(value: Any) -> dict[str, Any]:
    record = _exact_mapping(
        value,
        {
            "origin_id",
            "origin_digest",
            "derivation_origin",
            "prediction_origins",
            "prediction_origin_counts",
            "row_assignments",
            "ordered_row_origin_digest",
            "lineage_component_origin_digest",
        },
        "Realization result_origin",
    )
    raw_rows = record["row_assignments"]
    if not isinstance(raw_rows, (list, tuple)):
        raise ImportIdentityError("Realization result_origin row_assignments must be a list.")
    assignments = []
    for index, raw_row in enumerate(raw_rows):
        row = _exact_mapping(
            raw_row,
            {"question_id", "prediction_origin", "prediction_lineage_id"},
            f"Realization result_origin row_assignments[{index}]",
        )
        assignments.append(
            (
                row["question_id"],
                row["prediction_origin"],
                row["prediction_lineage_id"],
            )
        )
    rebuilt = make_result_origin(
        derivation_origin=record["derivation_origin"],
        row_assignments=assignments,
    )
    if canonicalize(record) != canonicalize(rebuilt):
        raise ImportIdentityError("Realization result_origin is internally inconsistent.")
    return rebuilt


def _validate_realization_identity(
    value: Mapping[str, Any],
    *,
    runtime_importer: Mapping[str, Any],
    expected_dataset: ExpectedDataset,
    lineage_runtime_callables: Mapping[str, Callable[..., Any]],
) -> dict[str, Any]:
    raw = _exact_mapping(value, _REALIZATION_KEYS, "Realization identity")
    _validate_digest(raw["import_spec_digest"], "import_spec_digest")

    sources = raw["sources"]
    if not isinstance(sources, (list, tuple)) or not sources:
        raise ImportIdentityError("Realization sources must be a non-empty list.")
    normalized_sources = []
    seen_source_ids: set[str] = set()
    for index, raw_source in enumerate(sources):
        source = _exact_mapping(raw_source, _SOURCE_KEYS, f"Realization sources[{index}]")
        source_id = source["source_id"]
        logical_path = source["logical_path"]
        if (
            not isinstance(source_id, str)
            or not source_id
            or source_id in seen_source_ids
        ):
            raise ImportIdentityError("Realization source IDs must be non-empty and unique.")
        if not isinstance(logical_path, str):
            raise ImportIdentityError("Realization source logical_path must be a string.")
        pure_path = PurePosixPath(logical_path)
        if (
            pure_path.is_absolute()
            or ".." in pure_path.parts
            or not pure_path.parts
            or _is_machine_path(logical_path)
        ):
            raise ImportIdentityError("Realization source logical_path is unsafe.")
        if source["classification"] not in {
            "raw",
            "canonical",
            "derived",
            "repaired",
            "aggregate_only",
        }:
            raise ImportIdentityError("Realization source classification is invalid.")
        if source["format"] != "csv":
            raise ImportIdentityError("Realization source format is unsupported.")
        if not isinstance(source["format_version"], str) or not source["format_version"]:
            raise ImportIdentityError("Realization source format_version is invalid.")
        _validate_digest(source["sha256"], f"sources[{index}].sha256")
        provenance = _exact_mapping(
            source["provenance"],
            _SOURCE_PROVENANCE_KEYS,
            f"Realization sources[{index}].provenance",
        )
        for field in ("source_run_id", "source_repository", "source_commit"):
            value = provenance[field]
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ImportIdentityError(
                    f"Realization source provenance {field} must be null or non-empty."
                )
            if isinstance(value, str) and _is_machine_path(value):
                raise ImportIdentityError(
                    f"Realization source provenance {field} contains a machine-local path."
                )
        for field in ("notes_digest", "evidence_digest"):
            _validate_digest(
                provenance[field], f"sources[{index}].provenance.{field}", optional=True
            )
        _validate_unknown_reason_contract(
            {field: provenance[field] for field in _SOURCE_PROVENANCE_UNKNOWN_FIELDS},
            provenance["unknown_reasons"],
            f"Realization sources[{index}].provenance",
        )
        source = {**source, "provenance": canonicalize(provenance)}
        seen_source_ids.add(source_id)
        normalized_sources.append(canonicalize(source))

    expected = _exact_mapping(
        raw["expected_dataset"],
        {
            "snapshot_digest",
            "question_set_digest",
            "derivation_digest",
            "question_ids",
        },
        "Realization expected_dataset",
    )
    for field in ("snapshot_digest", "question_set_digest", "derivation_digest"):
        digest = expected[field]
        _validate_digest(digest, f"expected_dataset.{field}")
    expected_question_ids = expected["question_ids"]
    if not isinstance(expected_question_ids, (list, tuple)) or not all(
        isinstance(question_id, str) and question_id
        for question_id in expected_question_ids
    ):
        raise ImportIdentityError(
            "Realization expected dataset question_ids must be a non-empty string list."
        )
    if not expected_question_ids or len(expected_question_ids) != len(
        set(expected_question_ids)
    ):
        raise ImportIdentityError(
            "Realization expected dataset question_ids must be non-empty and unique."
        )
    if integrity_digest(list(expected_question_ids)) != expected["question_set_digest"]:
        raise ImportIdentityError(
            "Realization expected dataset question_ids do not match question_set_digest."
        )
    _validate_expected_dataset_contract(expected_dataset)
    authoritative_expected = canonicalize(
        {
            "snapshot_digest": expected_dataset.snapshot_digest,
            "question_set_digest": expected_dataset.question_set_digest,
            "derivation_digest": expected_dataset.derivation_digest,
            "question_ids": list(expected_dataset.selected_question_ids),
        }
    )
    if canonicalize(expected) != authoritative_expected:
        raise ImportIdentityError(
            "Realization expected dataset does not match the validated dataset snapshot."
        )

    raw_importer = raw["importer_implementation"]
    if canonicalize(raw_importer) != canonicalize(runtime_importer):
        raise ImportIdentityError(
            "Realization importer implementation does not match the supplied runtime "
            "adapter and validator callables."
        )
    importer = _exact_mapping(
        raw_importer,
        {"schema_version", "package", "importer", "adapter", "validator"},
        "Realization importer_implementation",
    )
    if importer["schema_version"] != "choicebench.importer-implementation.v1":
        raise ImportIdentityError("Realization importer implementation schema is invalid.")
    package = _exact_mapping(
        importer["package"], {"name", "version"}, "Realization importer package"
    )
    if package != {"name": "choicebench", "version": __version__}:
        raise ImportIdentityError("Realization importer package identity is invalid.")
    for role in ("importer", "adapter", "validator"):
        try:
            component = validate_implementation_identity_record(importer[role])
        except ImportSpecError as exc:
            raise ImportIdentityError(
                f"Realization importer {role} identity is invalid: {exc}"
            ) from exc
        if not isinstance(component.get("source_digest"), str):
            raise ImportIdentityError(
                f"Realization importer {role} lacks a source identity."
            )
        _validate_digest(component["source_digest"], f"importer.{role}.source_digest")

    parsing = _exact_mapping(
        raw["parsing_policy"], _PARSING_POLICY_KEYS, "Realization parsing_policy"
    )
    _exact_mapping(
        parsing["dialect"], _DIALECT_KEYS, "Realization parsing_policy.dialect"
    )
    try:
        dialect = validate_csv_dialect_identity(parsing["dialect"])
    except ImportSpecError as exc:
        raise ImportIdentityError(f"Invalid realization parsing dialect: {exc}") from exc
    mapping = parsing["mapping"]
    if not isinstance(mapping, Mapping):
        raise ImportIdentityError("Realization parsing mapping must be a mapping.")
    normalized_mapping: dict[str, str] = {}
    for semantic_field, source_column in mapping.items():
        if not isinstance(semantic_field, str) or not semantic_field.strip():
            raise ImportIdentityError(
                "Realization parsing mapping keys must be non-empty strings."
            )
        if not isinstance(source_column, str) or not source_column.strip():
            raise ImportIdentityError(
                "Realization parsing mapping values must be non-empty source columns."
            )
        normalized_mapping[semantic_field] = source_column
    if len(normalized_mapping.values()) != len(set(normalized_mapping.values())):
        raise ImportIdentityError("Realization parsing mapping reuses a source column.")
    null_values = parsing["null_values"]
    if not isinstance(null_values, (list, tuple)) or not all(
        isinstance(item, str) for item in null_values
    ):
        raise ImportIdentityError(
            "Realization parsing null_values must be a string list."
        )
    if len(null_values) != len(set(null_values)):
        raise ImportIdentityError("Realization parsing null_values contains duplicates.")
    try:
        numeric_columns = validate_numeric_columns_identity(
            parsing["numeric_columns"]
        )
        option_mapping = validate_option_mapping_identity(parsing["option_mapping"])
    except ImportSpecError as exc:
        raise ImportIdentityError(f"Invalid realization parsing policy: {exc}") from exc
    if parsing["extra_field_policy"] not in {
        "preserve_unmapped",
        "reject_unmapped",
    }:
        raise ImportIdentityError("Realization extra_field_policy is invalid.")

    validation = _exact_mapping(
        raw["validation"], {"findings_digest"}, "Realization validation"
    )
    _validate_digest(validation["findings_digest"], "validation.findings_digest")
    evidence = _exact_mapping(
        raw["evidence"],
        {
            "evidence_status",
            "qualification_digest",
            "limitation_digest",
            "defect_digest",
            "scope_disposition",
        },
        "Realization evidence",
    )
    if evidence["evidence_status"] not in _EVIDENCE_STATUSES:
        raise ImportIdentityError("Realization evidence_status is invalid.")
    if evidence["scope_disposition"] not in _SCOPE_DISPOSITIONS:
        raise ImportIdentityError("Realization scope_disposition is invalid.")
    for field in ("qualification_digest", "limitation_digest", "defect_digest"):
        _validate_digest(evidence[field], f"evidence.{field}")

    raw_lineage_components = raw["lineage_components"]
    if not isinstance(raw_lineage_components, (list, tuple)):
        raise ImportIdentityError(
            "Realization lineage_components must be a list."
        )
    if not isinstance(lineage_runtime_callables, Mapping) or not all(
        isinstance(lineage_id, str) and callable(target)
        for lineage_id, target in lineage_runtime_callables.items()
    ):
        raise ImportIdentityError(
            "Realization lineage runtime callable registry is invalid."
        )
    lineage_components = []
    for component in raw_lineage_components:
        claimed_lineage_id = (
            component.get("lineage_id") if isinstance(component, Mapping) else None
        )
        lineage_components.append(
            _validate_lineage_component_record(
                component,
                runtime_callable=lineage_runtime_callables.get(claimed_lineage_id),
            )
        )
    lineage_by_id: dict[str, dict[str, Any]] = {}
    lineage_digests: set[str] = set()
    for component in lineage_components:
        lineage_id = component["lineage_id"]
        lineage_digest = component["lineage_digest"]
        if lineage_id in lineage_by_id or lineage_digest in lineage_digests:
            raise ImportIdentityError(
                "Realization lineage components contain duplicate identities."
            )
        lineage_by_id[lineage_id] = component
        lineage_digests.add(lineage_digest)
    runtime_lineage_ids = {
        component["lineage_id"]
        for component in lineage_components
        if component["identity"]["implementation"]["identity_mode"]
        == "runtime_callable"
    }
    if set(lineage_runtime_callables) != runtime_lineage_ids:
        raise ImportIdentityError(
            "Realization lineage runtime callable registry does not exactly own "
            "the runtime lineage components."
        )

    result_origin = _validate_result_origin_record(raw["result_origin"])
    origin_question_ids = [
        row["question_id"] for row in result_origin["row_assignments"]
    ]
    evidence_status = evidence["evidence_status"]
    origin_question_id_set = set(origin_question_ids)
    unexpected_origin_ids = sorted(
        origin_question_id_set - set(expected_question_ids)
    )
    ordered_subset = [
        question_id
        for question_id in expected_question_ids
        if question_id in origin_question_id_set
    ]
    if unexpected_origin_ids or ordered_subset != origin_question_ids:
        raise ImportIdentityError(
            "Realization result origin question IDs are not an ordered subset of "
            "the expected dataset question set."
        )
    if (
        evidence_status in {"complete", "qualified"}
        and evidence["scope_disposition"] == "included"
        and origin_question_ids != list(expected_question_ids)
    ):
        raise ImportIdentityError(
            "Complete or qualified realization result origin question IDs do not "
            "match the full expected dataset question set."
        )
    for assignment in result_origin["row_assignments"]:
        lineage = lineage_by_id.get(assignment["prediction_lineage_id"])
        if lineage is None:
            raise ImportIdentityError(
                "Realization result origin references an unowned lineage component."
            )
        lineage_identity = lineage["identity"]
        if (
            lineage_identity["question_id"] != assignment["question_id"]
            or lineage_identity["prediction_origin"]
            != assignment["prediction_origin"]
        ):
            raise ImportIdentityError(
                "Realization result origin conflicts with its lineage component."
            )
    assigned_lineage_ids = {
        assignment["prediction_lineage_id"]
        for assignment in result_origin["row_assignments"]
    }
    unreachable_lineage_ids = sorted(set(lineage_by_id) - assigned_lineage_ids)
    if unreachable_lineage_ids:
        raise ImportIdentityError(
            f"Realization lineage component(s) {unreachable_lineage_ids} are not "
            "reachable from any result-origin row."
        )
    parents = _exact_mapping(
        raw["parent_digests"],
        {"realization_digests", "evidence_digests", "result_digests"},
        "Realization parent_digests",
    )
    normalized_parents = {
        field: _digest_sequence(items, f"parent_digests.{field}")
        for field, items in parents.items()
    }
    authorization_digest = _validate_digest(
        raw["authorization_digest"], "authorization_digest", optional=True
    )
    lineage_authorizations = {
        component["identity"]["authorization_digest"]
        for component in lineage_components
        if component["identity"]["authorization_digest"] is not None
    }
    if any(digest != authorization_digest for digest in lineage_authorizations):
        raise ImportIdentityError(
            "Realization lineage authorization is not owned by the realization."
        )
    declared_parent_edges = {
        digest
        for digests in normalized_parents.values()
        for digest in digests
    }
    lineage_parent_edges = {
        digest
        for component in lineage_components
        for digest in component["identity"]["parent_digests"]
    }
    if not lineage_parent_edges <= declared_parent_edges:
        raise ImportIdentityError(
            "Realization lineage parent edge is not owned by the realization."
        )
    overlay = raw["overlay"]
    normalized_overlay = None
    if overlay is not None:
        overlay = _exact_mapping(
            overlay,
            {
                "source_sha256",
                "replacement_digest",
                "transformation_input_digest",
                "preownership_output_digest",
                "implementation_digest",
            },
            "Realization overlay",
        )
        normalized_overlay = {}
        for field, digest in overlay.items():
            normalized_overlay[field] = _validate_digest(
                digest,
                f"overlay.{field}",
                optional=(field not in {"source_sha256", "replacement_digest"}),
            )
    declared_source_digests = {source["sha256"] for source in normalized_sources}
    if normalized_overlay is not None:
        declared_source_digests.add(normalized_overlay["source_sha256"])
    lineage_source_edges = {
        digest
        for component in lineage_components
        for digest in component["identity"]["source_digests"]
    }
    if not lineage_source_edges <= declared_source_digests:
        raise ImportIdentityError(
            "Realization lineage source edge is not owned by the realization."
        )
    unreachable_source_digests = (
        sorted(declared_source_digests - lineage_source_edges)
        if lineage_components
        else []
    )
    if unreachable_source_digests:
        raise ImportIdentityError(
            f"Realization source(s) {unreachable_source_digests} are not reachable "
            "from any lineage component."
        )

    derivation_origin = result_origin["derivation_origin"]
    derived = derivation_origin in {"repair_overlay", "offline_transformation"}
    if derived:
        if authorization_digest is None:
            raise ImportIdentityError(
                f"{derivation_origin} realization requires an authorization digest."
            )
        if not normalized_parents["realization_digests"]:
            raise ImportIdentityError(
                f"{derivation_origin} realization requires a parent realization digest."
            )
        if not normalized_parents["evidence_digests"]:
            raise ImportIdentityError(
                f"{derivation_origin} realization requires a parent evidence digest."
            )
        if normalized_overlay is None:
            raise ImportIdentityError(
                f"{derivation_origin} realization requires an overlay identity."
            )
        if authorization_digest not in lineage_authorizations:
            raise ImportIdentityError(
                f"{derivation_origin} realization requires an authorized lineage "
                "component."
            )
        if not lineage_parent_edges:
            raise ImportIdentityError(
                f"{derivation_origin} realization requires a parent-derived lineage "
                "component."
            )
        derived_components = [
            component
            for component in lineage_components
            if component["identity"]["operation_type"] == derivation_origin
        ]
        if not derived_components:
            raise ImportIdentityError(
                f"{derivation_origin} realization requires a matching derived "
                "lineage component."
            )
        unassigned_derived_ids = sorted(
            component["lineage_id"]
            for component in derived_components
            if component["lineage_id"] not in assigned_lineage_ids
        )
        if unassigned_derived_ids:
            raise ImportIdentityError(
                f"{derivation_origin} lineage component(s) {unassigned_derived_ids} "
                "do not own a result-origin row assignment."
            )
        for component in derived_components:
            component_identity = component["identity"]
            if component_identity["authorization_digest"] != authorization_digest:
                raise ImportIdentityError(
                    f"{derivation_origin} lineage component lacks the realization "
                    "authorization."
                )
            if not component_identity["parent_digests"]:
                raise ImportIdentityError(
                    f"{derivation_origin} lineage component lacks a parent edge."
                )
            if not (
                set(component_identity["parent_digests"])
                & set(normalized_parents["realization_digests"])
            ):
                raise ImportIdentityError(
                    f"{derivation_origin} lineage component lacks the declared base "
                    "realization edge."
                )
            if normalized_overlay["source_sha256"] not in component_identity[
                "source_digests"
            ]:
                raise ImportIdentityError(
                    f"{derivation_origin} lineage component lacks the overlay source "
                    "edge."
                )
        if derivation_origin == "repair_overlay":
            for assignment in result_origin["row_assignments"]:
                if assignment["prediction_origin"] in {
                    "native_inference",
                    "external_repair_inference",
                }:
                    component = lineage_by_id[assignment["prediction_lineage_id"]]
                    if component["identity"]["operation_type"] != "repair_overlay":
                        raise ImportIdentityError(
                            "Repair prediction row lacks repair_overlay lineage."
                        )
        if derivation_origin == "offline_transformation" and any(
            normalized_overlay[field] is None
            for field in (
                "transformation_input_digest",
                "preownership_output_digest",
                "implementation_digest",
            )
        ):
            raise ImportIdentityError(
                "offline_transformation overlay requires input, output, and "
                "implementation digests."
            )
        if derivation_origin == "offline_transformation":
            expected_input_digest = integrity_digest(
                [
                    component["identity"]["input_digest"]
                    for component in derived_components
                ]
            )
            expected_output_digest = integrity_digest(
                [
                    component["identity"]["preownership_output_digest"]
                    for component in derived_components
                ]
            )
            implementation_identities = {
                integrity_digest(component["identity"]["implementation"]): component[
                    "identity"
                ]["implementation"]
                for component in derived_components
            }
            if len(implementation_identities) != 1:
                raise ImportIdentityError(
                    "offline_transformation lineage components use conflicting "
                    "implementations."
                )
            expected_implementation_digest = next(iter(implementation_identities))
            if (
                normalized_overlay["transformation_input_digest"]
                != expected_input_digest
                or normalized_overlay["preownership_output_digest"]
                != expected_output_digest
                or normalized_overlay["implementation_digest"]
                != expected_implementation_digest
            ):
                raise ImportIdentityError(
                    "offline_transformation overlay input, output, or implementation "
                    "digest conflicts with its lineage components."
                )
        if derivation_origin == "repair_overlay" and not (
            {"native_inference", "external_repair_inference"}
            & set(result_origin["prediction_origins"])
        ):
            raise ImportIdentityError(
                "repair_overlay requires at least one repair prediction origin."
            )
    else:
        if authorization_digest is not None or normalized_overlay is not None:
            raise ImportIdentityError(
                f"{derivation_origin} realization cannot claim repair authorization "
                "or overlay."
            )
        incompatible_operations = sorted(
            {
                component["identity"]["operation_type"]
                for component in lineage_components
                if component["identity"]["operation_type"] != derivation_origin
            }
        )
        if incompatible_operations:
            raise ImportIdentityError(
                f"{derivation_origin} realization lineage components have "
                f"incompatible operation type(s) {incompatible_operations}."
            )

    normalized = {
        "import_spec_digest": raw["import_spec_digest"],
        "sources": normalized_sources,
        "expected_dataset": canonicalize(expected),
        "importer_implementation": canonicalize(importer),
        "parsing_policy": canonicalize(
            {
                "dialect": dialect,
                "mapping": normalized_mapping,
                "null_values": list(null_values),
                "numeric_columns": numeric_columns,
                "option_mapping": option_mapping,
                "extra_field_policy": parsing["extra_field_policy"],
            }
        ),
        "validation": canonicalize(validation),
        "evidence": canonicalize(evidence),
        "lineage_components": canonicalize(lineage_components),
        "result_origin": result_origin,
        "parent_digests": normalized_parents,
        "authorization_digest": authorization_digest,
        "overlay": normalized_overlay,
    }
    _reject_forbidden(
        normalized,
        "Realization identity",
        allowed_path_keys=frozenset({"logical_path"}),
    )
    return canonicalize(normalized)


def make_realization(
    *,
    condition_id: str,
    condition_digest: str,
    identity: Mapping[str, Any],
    fields: Mapping[str, Any],
    adapter: Callable[..., Any],
    validator: Callable[..., Any],
    expected_dataset: ExpectedDataset,
    semantic_condition: Mapping[str, Any],
    lineage_runtime_callables: Mapping[str, Callable[..., Any]],
) -> dict[str, Any]:
    """Create an immutable realization identity distinct from its condition."""
    if not isinstance(condition_id, str) or not re.fullmatch(r"cond_[0-9a-f]{16}", condition_id):
        raise ImportIdentityError("Realization condition_id is invalid.")
    _validate_digest(condition_digest, "condition_digest")
    if condition_id != f"cond_{condition_digest[:16]}":
        raise ImportIdentityError(
            "Realization condition_id does not match the condition_digest."
        )
    if not isinstance(identity, Mapping) or not identity:
        raise ImportIdentityError("Realization identity must be a non-empty mapping.")
    condition_record = _exact_mapping(
        semantic_condition,
        {
            "condition_id",
            "condition_digest",
            "identity",
            "condition_key",
            "expected_question_ids",
        },
        "Realization semantic condition",
    )
    rebuilt_condition = make_semantic_condition(
        identity=condition_record["identity"],
        fields={
            "condition_key": condition_record["condition_key"],
            "expected_question_ids": condition_record["expected_question_ids"],
        },
    )
    if canonicalize(rebuilt_condition) != canonicalize(condition_record):
        raise ImportIdentityError(
            "Realization semantic condition identity is inconsistent."
        )
    if (
        condition_record["condition_id"] != condition_id
        or condition_record["condition_digest"] != condition_digest
    ):
        raise ImportIdentityError(
            "Realization condition ID/digest do not match its semantic condition."
        )
    benchmark = condition_record["identity"]["benchmark"]
    if (
        benchmark["name"] != expected_dataset.benchmark_name
        or benchmark["split"] != expected_dataset.split
        or benchmark["artifact_id"] != expected_dataset.artifact_id
        or benchmark["selection_id"] != expected_dataset.selection_id
        or tuple(condition_record["expected_question_ids"])
        != expected_dataset.selected_question_ids
    ):
        raise ImportIdentityError(
            "Realization semantic condition is not owned by its expected dataset "
            "artifact and selection."
        )
    runtime_importer = importer_implementation_identity(
        adapter=adapter, validator=validator
    )
    validated_identity = _validate_realization_identity(
        identity,
        runtime_importer=runtime_importer,
        expected_dataset=expected_dataset,
        lineage_runtime_callables=lineage_runtime_callables,
    )
    payload = canonicalize(
        {
            "schema_version": "choicebench.realization.v1",
            "condition_id": condition_id,
            "condition_digest": condition_digest,
            "realization": validated_identity,
        }
    )
    if not isinstance(fields, Mapping) or not set(fields) <= {"audit", "realization_key"}:
        raise ImportIdentityError(
            "Realization fields may contain only audit and realization_key."
        )
    if "audit" in fields and not isinstance(fields["audit"], Mapping):
        raise ImportIdentityError("Realization audit field must be a mapping.")
    extra = canonicalize(fields)
    return {
        "realization_id": short_id("real", payload),
        "realization_digest": integrity_digest(payload),
        "condition_id": condition_id,
        "condition_digest": condition_digest,
        "identity": payload,
        **extra,
    }
