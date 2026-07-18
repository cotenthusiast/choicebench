"""Layered semantic, realization, lineage, and prediction-origin identities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal, Mapping, Sequence

from choicebench import __version__
from choicebench.identity import canonicalize, integrity_digest, short_id
from choicebench.importing.dataset_reference import ExpectedDataset
from choicebench.importing.schema import (
    ImportConditionSpec,
    ImportMethodSpec,
    ImportModelSpec,
    ImportPromptSpec,
)
from choicebench.provenance import implementation_identity


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
_FORBIDDEN_IDENTITY_KEYS = {
    "audit_path",
    "absolute_path",
    "choicebench_home",
    "experiment_id",
    "final_csv_sha256",
    "hostname",
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


def _reject_forbidden(value: Any, where: str) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if key in _FORBIDDEN_IDENTITY_KEYS:
                raise ImportIdentityError(f"{where} contains forbidden field {key!r}.")
            _reject_forbidden(item, where)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _reject_forbidden(item, where)


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
        if condition.generation_parameters:
            native_generation = model_payload.get("generation_kwargs")
            if not isinstance(native_generation, Mapping) or any(
                native_generation.get(name) != value
                for name, value in condition.generation_parameters.items()
            ):
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
        declared_preflight = _nullable_semantic(
            condition.preflight_identity,
            condition.unknown_reasons.get("preflight_identity"),
            "method preflight identity",
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
            {"version": native["version"], "files": native["files"]}
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
            and canonicalize(declared_contents) != canonicalize(native_contents)
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
        prompt_id, prompt_digest, prompt_payload = _identity_record(
            "prompt", prompt_payload
        )
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
    _reject_forbidden(identity, "Semantic condition identity")
    payload = canonicalize(identity)
    condition_id = short_id("cond", payload)
    condition_digest = integrity_digest(payload)
    extra = canonicalize(fields)
    reserved = {"condition_id", "condition_digest", "identity"}
    if not isinstance(extra, Mapping) or reserved & set(extra):
        raise ImportIdentityError("Semantic condition fields overwrite identity ownership.")
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
    if tuple(condition.expected_question_ids) != dataset.selected_question_ids:
        raise ImportIdentityError("Condition question set does not match dataset selection.")
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


def make_lineage_component(
    *,
    operation_type: str,
    question_id: str,
    parent_digests: Sequence[str],
    source_digests: Sequence[str],
    authorization_digest: str | None,
    implementation: Mapping[str, Any],
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
    _validate_digest(authorization_digest, "authorization_digest", optional=True)
    _validate_digest(input_digest, "input_digest")
    _validate_digest(preownership_output_digest, "preownership_output_digest")
    _reject_forbidden(implementation, "Lineage implementation")
    _reject_forbidden(parameters, "Lineage parameters")
    payload = canonicalize(
        {
            "schema_version": "choicebench.lineage-component.v1",
            "operation_type": operation_type,
            "question_id": question_id,
            "parent_digests": sorted(parent_digests),
            "source_digests": sorted(source_digests),
            "authorization_digest": authorization_digest,
            "implementation": implementation,
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
    if not row_assignments:
        raise ImportIdentityError("Result origin requires row assignments.")
    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    counts: dict[str, int] = {}
    for question_id, prediction_origin, lineage_id in row_assignments:
        if not isinstance(question_id, str) or not question_id or question_id in seen:
            raise ImportIdentityError("Result origin question IDs must be non-empty and unique.")
        if prediction_origin not in _PREDICTION_ORIGINS:
            raise ImportIdentityError(f"Invalid prediction origin {prediction_origin!r}.")
        if not isinstance(lineage_id, str) or not lineage_id:
            raise ImportIdentityError("Result origin lineage IDs must be non-empty.")
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
    return {
        "derivation_origin": derivation_origin,
        "prediction_origins": sorted(counts),
        "prediction_origin_counts": {key: counts[key] for key in sorted(counts)},
        "row_assignments": rows,
        "ordered_row_origin_digest": integrity_digest(rows),
        "lineage_component_origin_digest": integrity_digest(lineage_projection),
    }


def importer_implementation_identity(
    *, adapter: Callable[..., Any], validator: Callable[..., Any]
) -> dict[str, Any]:
    """Bind installed ChoiceBench plus registered adapter and validator code."""
    return canonicalize(
        {
            "schema_version": "choicebench.importer-implementation.v1",
            "package": {"name": "choicebench", "version": __version__},
            "adapter": implementation_identity(adapter),
            "validator": implementation_identity(validator),
        }
    )


def make_realization(
    *,
    condition_id: str,
    condition_digest: str,
    identity: Mapping[str, Any],
    fields: Mapping[str, Any],
) -> dict[str, Any]:
    """Create an immutable realization identity distinct from its condition."""
    if not isinstance(condition_id, str) or not condition_id.startswith("cond_"):
        raise ImportIdentityError("Realization condition_id is invalid.")
    _validate_digest(condition_digest, "condition_digest")
    if not isinstance(identity, Mapping) or not identity:
        raise ImportIdentityError("Realization identity must be a non-empty mapping.")
    _reject_forbidden(identity, "Realization identity")
    payload = canonicalize(
        {
            "schema_version": "choicebench.realization.v1",
            "condition_id": condition_id,
            "condition_digest": condition_digest,
            "realization": identity,
        }
    )
    extra = canonicalize(fields)
    reserved = {
        "realization_id",
        "realization_digest",
        "condition_id",
        "condition_digest",
        "identity",
    }
    if not isinstance(extra, Mapping) or reserved & set(extra):
        raise ImportIdentityError("Realization fields overwrite identity ownership.")
    return {
        "realization_id": short_id("real", payload),
        "realization_digest": integrity_digest(payload),
        "condition_id": condition_id,
        "condition_digest": condition_digest,
        "identity": payload,
        **extra,
    }
