"""Strict, paper-agnostic declarations for importing external results."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from numbers import Real
from pathlib import Path, PurePosixPath
import re
from typing import Any, Literal, Mapping, TypeAlias

import yaml

from choicebench.identity import canonicalize, integrity_digest, is_credential_key, short_id
from choicebench.metrics import BUILTIN_METRICS


ImportState: TypeAlias = Literal["validated", "imported", "failed"]
EvidenceStatus: TypeAlias = Literal[
    "complete", "qualified", "partial", "malformed", "recoverable", "failed"
]
ScopeDisposition: TypeAlias = Literal[
    "included", "excluded_from_paper_matrix", "held", "superseded"
]
PredictionOrigin: TypeAlias = Literal[
    "native_inference",
    "external_historical_inference",
    "external_repair_inference",
]
DerivationOrigin: TypeAlias = Literal[
    "native_execution", "external_import", "repair_overlay", "offline_transformation"
]


class ImportSpecError(ValueError):
    """Raised when an external import declaration is unsafe or inconsistent."""


@dataclass(frozen=True)
class CsvDialectSpec:
    encoding: Literal["utf-8"] = "utf-8"
    bom_policy: Literal["forbid", "strip_utf8_bom"] = "forbid"
    decoding_errors: Literal["strict"] = "strict"
    delimiter: str = ","
    quote_character: str = '"'
    escape_character: str | None = None
    double_quote: bool = True
    line_terminators: tuple[str, ...] = ("crlf", "lf", "cr")
    mixed_line_terminators: Literal["allow", "forbid"] = "allow"
    final_record_without_terminator: Literal["allow", "forbid"] = "allow"
    blank_record_policy: Literal["reject"] = "reject"
    skip_initial_space: bool = False
    header: Literal["first_logical_record"] = "first_logical_record"
    strict_syntax: bool = True


@dataclass(frozen=True)
class NumericColumnSpec:
    source_column: str
    value_type: Literal["integer", "float"]
    null_allowed: bool
    finite_only: bool = True


@dataclass(frozen=True)
class OptionMappingSpec:
    mode: Literal["ordered_columns", "structured_json"]
    ordered_columns: tuple[str, ...]
    structured_column: str | None
    structured_label_key: str | None
    structured_text_key: str | None


@dataclass(frozen=True)
class SourceArtifactSpec:
    source_id: str
    path: Path
    logical_path: str
    expected_sha256: str
    format: Literal["csv"]
    format_version: str
    classification: Literal["raw", "canonical", "derived", "repaired", "aggregate_only"]
    dialect: CsvDialectSpec
    columns: Mapping[str, str]
    expected_columns: tuple[str, ...]
    ignored_columns: Mapping[str, str]
    null_values: tuple[str, ...]
    numeric_columns: tuple[NumericColumnSpec, ...]
    option_mapping: OptionMappingSpec
    extra_field_policy: Literal["preserve_unmapped", "reject_unmapped"]
    preserve_namespace: str
    source_run_id: str | None
    source_repository: str | None
    source_commit: str | None
    notes: Mapping[str, Any]


@dataclass(frozen=True)
class DatasetReferenceSpec:
    dataset_id: str
    benchmark_name: str
    split: str
    reference_kind: Literal[
        "independent_input_snapshot", "profile_derived_reference_snapshot"
    ]
    trust_label: str
    source_ids: tuple[str, ...]
    selection_source_id: str
    expected_question_ids: tuple[str, ...]
    selection_seed: int | None
    selection_n_samples: int | None
    subject_filter: tuple[str, ...]
    selection_unknown_reasons: Mapping[str, str]
    columns: Mapping[str, str]
    revision: str | None
    fingerprint: str | None
    derivation: Mapping[str, Any]
    limitations: tuple[str, ...]
    native_compatibility_identity: Mapping[str, Any] | None


@dataclass(frozen=True)
class ImportModelSpec:
    model_key: str
    display_name: str
    backend: str | None
    provider: str | None
    revision: str | None
    effective_parameters: Mapping[str, Any]
    unknown_reasons: Mapping[str, str]
    native_compatibility_identity: Mapping[str, Any] | None


@dataclass(frozen=True)
class ImportMethodSpec:
    method_key: str
    name: str
    effective_parameters: Mapping[str, Any]
    implementation: Mapping[str, Any] | None
    unknown_reasons: Mapping[str, str]
    native_compatibility_identity: Mapping[str, Any] | None


@dataclass(frozen=True)
class ImportPromptSpec:
    prompt_key: str
    template_identity: str | None
    template_digest: str | None
    template_contents: Mapping[str, str] | None
    unknown_reason: str | None
    native_compatibility_identity: Mapping[str, Any] | None


@dataclass(frozen=True)
class ResultOriginSpec:
    derivation_origin: DerivationOrigin
    default_prediction_origin: PredictionOrigin | None
    per_question_prediction_origins: Mapping[str, PredictionOrigin]


@dataclass(frozen=True)
class ImportConditionSpec:
    condition_key: str
    source_ids: tuple[str, ...]
    dataset_id: str
    model_key: str
    method_key: str
    prompt_key: str
    seed: int | None
    calibration_identity: Mapping[str, Any] | None
    preflight_identity: Mapping[str, Any] | None
    protocol_settings: Mapping[str, Any]
    generation_parameters: Mapping[str, Any]
    unknown_reasons: Mapping[str, str]
    expected_question_ids: tuple[str, ...]
    evidence_status: EvidenceStatus
    scope_disposition: ScopeDisposition
    executable: bool | None
    qualifications: tuple[Mapping[str, Any], ...]
    limitations: tuple[Mapping[str, Any], ...]
    damaged_question_ids: tuple[str, ...]
    recoverable_question_ids: tuple[str, ...]
    result_origin: ResultOriginSpec


@dataclass(frozen=True)
class AuthorizationSpec:
    authorization_id: str
    authorization_type: Literal["inference_repair", "offline_transformation"]
    source_id: str
    condition_question_reasons: Mapping[str, Mapping[str, str]]
    authority: str
    purpose: str
    executable: bool
    input_evidence_digests: Mapping[str, str]
    expected_snapshot_digests: Mapping[str, str]


@dataclass(frozen=True)
class OverlaySpec:
    overlay_id: str
    base_run_path: Path
    base_condition_digest: str
    base_realization_id: str
    base_realization_digest: str
    base_evidence_digests: Mapping[str, str]
    base_validation_artifact_sha256: str
    base_result_sha256: str | None
    source_id: str
    authorization_id: str
    replacement_reasons: Mapping[str, str]
    result_origin: ResultOriginSpec
    lineage_notes: Mapping[str, Any]
    implementation: Mapping[str, Any]
    input_digest: str
    preownership_output_digest: str
    expected_evidence_status: EvidenceStatus


@dataclass(frozen=True)
class ImportSpec:
    schema_version: Literal["choicebench.import-spec.v1"]
    import_name: str
    sources: tuple[SourceArtifactSpec, ...]
    datasets: tuple[DatasetReferenceSpec, ...]
    models: tuple[ImportModelSpec, ...]
    methods: tuple[ImportMethodSpec, ...]
    prompts: tuple[ImportPromptSpec, ...]
    conditions: tuple[ImportConditionSpec, ...]
    authorizations: tuple[AuthorizationSpec, ...]
    overlays: tuple[OverlaySpec, ...]
    metrics: tuple[str, ...]
    provenance: Mapping[str, Any]
    audit: Mapping[str, Any]


_TOP_LEVEL_KEYS = {
    "schema_version",
    "import_name",
    "sources",
    "datasets",
    "models",
    "methods",
    "prompts",
    "conditions",
    "authorizations",
    "overlays",
    "metrics",
    "provenance",
    "audit",
}
_SOURCE_KEYS = {
    "source_id",
    "path",
    "logical_path",
    "expected_sha256",
    "format",
    "format_version",
    "classification",
    "dialect",
    "columns",
    "expected_columns",
    "ignored_columns",
    "null_values",
    "numeric_columns",
    "option_mapping",
    "extra_field_policy",
    "preserve_namespace",
    "source_run_id",
    "source_repository",
    "source_commit",
    "notes",
}
_DIALECT_KEYS = {
    "encoding",
    "bom_policy",
    "decoding_errors",
    "delimiter",
    "quote_character",
    "escape_character",
    "double_quote",
    "line_terminators",
    "mixed_line_terminators",
    "final_record_without_terminator",
    "blank_record_policy",
    "skip_initial_space",
    "header",
    "strict_syntax",
}
_NUMERIC_KEYS = {"source_column", "value_type", "null_allowed", "finite_only"}
_OPTION_KEYS = {
    "mode",
    "ordered_columns",
    "structured_column",
    "structured_label_key",
    "structured_text_key",
}
_DATASET_KEYS = {
    "dataset_id",
    "benchmark_name",
    "split",
    "reference_kind",
    "trust_label",
    "source_ids",
    "selection_source_id",
    "expected_question_ids",
    "selection_seed",
    "selection_n_samples",
    "subject_filter",
    "selection_unknown_reasons",
    "columns",
    "revision",
    "fingerprint",
    "derivation",
    "limitations",
    "native_compatibility_identity",
}
_MODEL_KEYS = {
    "model_key",
    "display_name",
    "backend",
    "provider",
    "revision",
    "effective_parameters",
    "unknown_reasons",
    "native_compatibility_identity",
}
_METHOD_KEYS = {
    "method_key",
    "name",
    "effective_parameters",
    "implementation",
    "unknown_reasons",
    "native_compatibility_identity",
}
_PROMPT_KEYS = {
    "prompt_key",
    "template_identity",
    "template_digest",
    "template_contents",
    "unknown_reason",
    "native_compatibility_identity",
}
_CONDITION_KEYS = {
    "condition_key",
    "source_ids",
    "dataset_id",
    "model_key",
    "method_key",
    "prompt_key",
    "seed",
    "calibration_identity",
    "preflight_identity",
    "protocol_settings",
    "generation_parameters",
    "unknown_reasons",
    "expected_question_ids",
    "evidence_status",
    "scope_disposition",
    "executable",
    "qualifications",
    "limitations",
    "damaged_question_ids",
    "recoverable_question_ids",
    "result_origin",
}
_PROTOCOL_SETTING_KEYS = {"pride_modal_k_threshold"}
_RESULT_ORIGIN_KEYS = {
    "derivation_origin",
    "default_prediction_origin",
    "per_question_prediction_origins",
}
_AUTHORIZATION_KEYS = {
    "authorization_id",
    "authorization_type",
    "source_id",
    "condition_question_reasons",
    "authority",
    "purpose",
    "executable",
    "input_evidence_digests",
    "expected_snapshot_digests",
}
_OVERLAY_KEYS = {
    "overlay_id",
    "base_run_path",
    "base_condition_digest",
    "base_realization_id",
    "base_realization_digest",
    "base_evidence_digests",
    "base_validation_artifact_sha256",
    "base_result_sha256",
    "source_id",
    "authorization_id",
    "replacement_reasons",
    "result_origin",
    "lineage_notes",
    "implementation",
    "input_digest",
    "preownership_output_digest",
    "expected_evidence_status",
}

_EVIDENCE_STATUSES = {
    "complete", "qualified", "partial", "malformed", "recoverable", "failed"
}
_SCOPE_DISPOSITIONS = {
    "included", "excluded_from_paper_matrix", "held", "superseded"
}
_PREDICTION_ORIGINS = {
    "native_inference", "external_historical_inference", "external_repair_inference"
}
_DERIVATION_ORIGINS = {
    "native_execution", "external_import", "repair_overlay", "offline_transformation"
}
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_WINDOWS_ABSOLUTE_RE = re.compile(r"^[A-Za-z]:[\\/]")


def _mapping(value: Any, where: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ImportSpecError(f"{where} must be a YAML mapping; got {value!r}.")
    bad_key = next((key for key in value if not isinstance(key, str)), None)
    if bad_key is not None:
        raise ImportSpecError(
            f"{where} mapping keys must be strings; got {bad_key!r}."
        )
    return value


def _exact_keys(value: Any, allowed: set[str], where: str) -> Mapping[str, Any]:
    raw = _mapping(value, where)
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ImportSpecError(f"Unknown field(s) in {where}: {unknown}.")
    missing = sorted(allowed - set(raw))
    if missing:
        raise ImportSpecError(f"Missing required field(s) in {where}: {missing}.")
    return raw


def _allowed_keys(value: Any, allowed: set[str], where: str) -> Mapping[str, Any]:
    raw = _mapping(value, where)
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ImportSpecError(f"Unknown field(s) in {where}: {unknown}.")
    return raw


def _nonempty(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ImportSpecError(f"{where} must be a non-empty string; got {value!r}.")
    return value.strip()


def _optional_string(value: Any, where: str) -> str | None:
    if value is None:
        return None
    return _nonempty(value, where)


def _strict_bool(value: Any, where: str) -> bool:
    if not isinstance(value, bool):
        raise ImportSpecError(f"{where} must be true or false; got {value!r}.")
    return value


def _optional_bool(value: Any, where: str) -> bool | None:
    if value is None:
        return None
    return _strict_bool(value, where)


def _strict_int(value: Any, where: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ImportSpecError(f"{where} must be an integer; got {value!r}.")
    if minimum is not None and value < minimum:
        raise ImportSpecError(f"{where} must be >= {minimum}; got {value!r}.")
    return value


def _strict_number(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ImportSpecError(f"{where} must be a finite number; got {value!r}.")
    result = float(value)
    if not math.isfinite(result):
        raise ImportSpecError(f"{where} must be a finite number; got {value!r}.")
    return result


def _optional_int(value: Any, where: str, *, minimum: int | None = None) -> int | None:
    if value is None:
        return None
    return _strict_int(value, where, minimum=minimum)


def _enum(value: Any, values: set[str], where: str) -> str:
    result = _nonempty(value, where)
    if result not in values:
        raise ImportSpecError(
            f"{where} must be one of {sorted(values)}; got {result!r}."
        )
    return result


def _sha256(value: Any, where: str) -> str:
    result = _nonempty(value, where)
    if not _SHA256_RE.fullmatch(result):
        raise ImportSpecError(f"{where} must be a lowercase SHA-256 digest.")
    return result


def _optional_sha256(value: Any, where: str) -> str | None:
    if value is None:
        return None
    return _sha256(value, where)


def _sequence(value: Any, where: str) -> list[Any]:
    if not isinstance(value, list):
        raise ImportSpecError(f"{where} must be a YAML list; got {value!r}.")
    return value


def _strings(
    value: Any,
    where: str,
    *,
    nonempty: bool = False,
    unique: bool = False,
    allow_empty_items: bool = False,
) -> tuple[str, ...]:
    items = _sequence(value, where)
    if allow_empty_items:
        if not all(isinstance(item, str) for item in items):
            bad = next(item for item in items if not isinstance(item, str))
            raise ImportSpecError(f"{where} entries must be strings; got {bad!r}.")
        result = tuple(items)
    else:
        result = tuple(
            _nonempty(item, f"{where}[{index}]")
            for index, item in enumerate(items)
        )
    if nonempty and not result:
        raise ImportSpecError(f"{where} must be a non-empty list.")
    if unique and len(result) != len(set(result)):
        raise ImportSpecError(f"{where} contains duplicate values.")
    return result


def _string_mapping(value: Any, where: str) -> dict[str, str]:
    raw = _mapping(value, where)
    return {
        _nonempty(key, f"{where} key"): _nonempty(item, f"{where}.{key}")
        for key, item in raw.items()
    }


def _validate_unknown_reasons(
    values: Mapping[str, Any], reasons: Mapping[str, str], where: str
) -> None:
    unknown = sorted(set(reasons) - set(values))
    if unknown:
        raise ImportSpecError(f"{where} contains unknown reason key(s) {unknown}.")
    missing = sorted(key for key, value in values.items() if value is None and key not in reasons)
    if missing:
        raise ImportSpecError(
            f"{where} must document every unknown nullable field; missing {missing}."
        )
    contradictory = sorted(
        key for key, value in values.items() if value is not None and key in reasons
    )
    if contradictory:
        raise ImportSpecError(
            f"{where} gives unknown reasons for known field(s) {contradictory}."
        )


def _sha_mapping(value: Any, where: str) -> dict[str, str]:
    raw = _mapping(value, where)
    return {
        _nonempty(key, f"{where} key"): _sha256(item, f"{where}.{key}")
        for key, item in raw.items()
    }


def _credential_keys_in(value: Any) -> set[str]:
    found: set[str] = set()
    if isinstance(value, Mapping):
        for key, item in value.items():
            if isinstance(key, str) and is_credential_key(key):
                found.add(key)
            found |= _credential_keys_in(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            found |= _credential_keys_in(item)
    return found


def _canonical_mapping(value: Any, where: str) -> dict[str, Any]:
    raw = _mapping(value, where)
    try:
        result = canonicalize(raw)
    except (TypeError, ValueError) as exc:
        raise ImportSpecError(
            f"{where} must be finite, safe canonical data: {exc}"
        ) from exc
    return dict(result)


def _canonical_mapping_or_none(value: Any, where: str) -> dict[str, Any] | None:
    if value is None:
        return None
    return _canonical_mapping(value, where)


def _mapping_sequence(value: Any, where: str) -> tuple[Mapping[str, Any], ...]:
    return tuple(
        _canonical_mapping(item, f"{where}[{index}]")
        for index, item in enumerate(_sequence(value, where))
    )


def _is_path_like(value: str) -> bool:
    return (
        value.startswith(("/", "./", "../", "~/", "~\\"))
        or _WINDOWS_ABSOLUTE_RE.match(value) is not None
    )


def _contains_path_like(value: Any) -> bool:
    if isinstance(value, str):
        return _is_path_like(value)
    if isinstance(value, Mapping):
        return any(_contains_path_like(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_path_like(item) for item in value)
    return False


def _build_provenance(value: Any) -> dict[str, Any]:
    raw = _mapping(value, "provenance")
    result: dict[str, Any] = {}
    for key, item in raw.items():
        where = f"provenance.{key}"
        record = _exact_keys(item, {"value", "reason"}, where)
        reason = record["reason"]
        if record["value"] is None:
            reason = _nonempty(reason, f"{where}.reason")
        elif reason is not None:
            reason = _nonempty(reason, f"{where}.reason")
        if _contains_path_like(record["value"]):
            raise ImportSpecError(
                f"{where}.value contains a machine-local path; put it in audit instead."
            )
        canonical = _canonical_mapping(
            {"value": record["value"], "reason": reason}, where
        )
        result[_nonempty(key, "provenance key")] = canonical
    return result


def _ascii_byte(value: Any, where: str, *, optional: bool = False) -> str | None:
    if value is None and optional:
        return None
    if not isinstance(value, str) or len(value.encode("utf-8")) != 1:
        raise ImportSpecError(f"{where} must be one ASCII byte.")
    if ord(value) >= 128 or value in {"\x00", "\r", "\n"}:
        raise ImportSpecError(f"{where} must be one usable ASCII byte.")
    return value


def _build_dialect(value: Any, where: str) -> CsvDialectSpec:
    raw = _allowed_keys(value, _DIALECT_KEYS, where)
    encoding = _enum(raw.get("encoding", "utf-8"), {"utf-8"}, f"{where}.encoding")
    decoding_errors = _enum(
        raw.get("decoding_errors", "strict"), {"strict"}, f"{where}.decoding_errors"
    )
    delimiter = _ascii_byte(raw.get("delimiter", ","), f"{where}.delimiter")
    quote = _ascii_byte(raw.get("quote_character", '"'), f"{where}.quote_character")
    escape = _ascii_byte(
        raw.get("escape_character"), f"{where}.escape_character", optional=True
    )
    characters = [item for item in (delimiter, quote, escape) if item is not None]
    if len(characters) != len(set(characters)):
        raise ImportSpecError(
            f"{where} delimiter, quote_character, and escape_character must be distinct."
        )
    line_terminators = _strings(
        raw.get("line_terminators", ["crlf", "lf", "cr"]),
        f"{where}.line_terminators",
        nonempty=True,
        unique=True,
    )
    if not set(line_terminators) <= {"crlf", "lf", "cr"}:
        raise ImportSpecError(
            f"{where}.line_terminators may contain only crlf, lf, and cr."
        )
    return CsvDialectSpec(
        encoding=encoding,
        bom_policy=_enum(
            raw.get("bom_policy", "forbid"),
            {"forbid", "strip_utf8_bom"},
            f"{where}.bom_policy",
        ),
        decoding_errors=decoding_errors,
        delimiter=delimiter,
        quote_character=quote,
        escape_character=escape,
        double_quote=_strict_bool(raw.get("double_quote", True), f"{where}.double_quote"),
        line_terminators=line_terminators,
        mixed_line_terminators=_enum(
            raw.get("mixed_line_terminators", "allow"),
            {"allow", "forbid"},
            f"{where}.mixed_line_terminators",
        ),
        final_record_without_terminator=_enum(
            raw.get("final_record_without_terminator", "allow"),
            {"allow", "forbid"},
            f"{where}.final_record_without_terminator",
        ),
        blank_record_policy=_enum(
            raw.get("blank_record_policy", "reject"),
            {"reject"},
            f"{where}.blank_record_policy",
        ),
        skip_initial_space=_strict_bool(
            raw.get("skip_initial_space", False), f"{where}.skip_initial_space"
        ),
        header=_enum(
            raw.get("header", "first_logical_record"),
            {"first_logical_record"},
            f"{where}.header",
        ),
        strict_syntax=_strict_bool(
            raw.get("strict_syntax", True), f"{where}.strict_syntax"
        ),
    )


def _build_numeric(value: Any, where: str) -> NumericColumnSpec:
    raw = _allowed_keys(value, _NUMERIC_KEYS, where)
    missing = sorted({"source_column", "value_type", "null_allowed"} - set(raw))
    if missing:
        raise ImportSpecError(f"Missing required field(s) in {where}: {missing}.")
    return NumericColumnSpec(
        source_column=_nonempty(raw["source_column"], f"{where}.source_column"),
        value_type=_enum(
            raw["value_type"], {"integer", "float"}, f"{where}.value_type"
        ),
        null_allowed=_strict_bool(raw["null_allowed"], f"{where}.null_allowed"),
        finite_only=_strict_bool(raw.get("finite_only", True), f"{where}.finite_only"),
    )


def _build_option(value: Any, where: str) -> OptionMappingSpec:
    raw = _exact_keys(value, _OPTION_KEYS, where)
    mode = _enum(raw["mode"], {"ordered_columns", "structured_json"}, f"{where}.mode")
    ordered = _strings(
        raw["ordered_columns"], f"{where}.ordered_columns", unique=True
    )
    structured_column = _optional_string(
        raw["structured_column"], f"{where}.structured_column"
    )
    label_key = _optional_string(
        raw["structured_label_key"], f"{where}.structured_label_key"
    )
    text_key = _optional_string(
        raw["structured_text_key"], f"{where}.structured_text_key"
    )
    if mode == "ordered_columns":
        if not ordered or any(
            item is not None for item in (structured_column, label_key, text_key)
        ):
            raise ImportSpecError(
                f"{where} ordered_columns mode requires ordered columns and null structured fields."
            )
    elif ordered or any(item is None for item in (structured_column, label_key, text_key)):
        raise ImportSpecError(
            f"{where} structured_json mode requires an empty ordered list and all structured fields."
        )
    return OptionMappingSpec(mode, ordered, structured_column, label_key, text_key)


def _build_source(value: Any, index: int) -> SourceArtifactSpec:
    where = f"sources[{index}]"
    raw = _exact_keys(value, _SOURCE_KEYS, where)
    logical_path = _nonempty(raw["logical_path"], f"{where}.logical_path")
    pure_path = PurePosixPath(logical_path)
    if pure_path.is_absolute() or ".." in pure_path.parts:
        raise ImportSpecError(f"{where}.logical_path must be a safe relative logical path.")
    expected_columns = _strings(
        raw["expected_columns"], f"{where}.expected_columns", nonempty=True, unique=True
    )
    numeric_columns = tuple(
        _build_numeric(item, f"{where}.numeric_columns[{item_index}]")
        for item_index, item in enumerate(_sequence(raw["numeric_columns"], f"{where}.numeric_columns"))
    )
    numeric_names = [item.source_column for item in numeric_columns]
    if len(numeric_names) != len(set(numeric_names)):
        raise ImportSpecError(f"Duplicate numeric-column rules in {where}.numeric_columns.")
    missing_numeric = sorted(set(numeric_names) - set(expected_columns))
    if missing_numeric:
        raise ImportSpecError(
            f"{where}.numeric_columns reference unknown expected columns {missing_numeric}."
        )
    option_mapping = _build_option(raw["option_mapping"], f"{where}.option_mapping")
    option_columns = (
        set(option_mapping.ordered_columns)
        if option_mapping.mode == "ordered_columns"
        else {option_mapping.structured_column}
    )
    missing_options = sorted(item for item in option_columns if item not in expected_columns)
    if missing_options:
        raise ImportSpecError(
            f"{where}.option_mapping references unknown expected columns {missing_options}."
        )
    columns = _string_mapping(raw["columns"], f"{where}.columns")
    missing_mappings = sorted(set(columns.values()) - set(expected_columns))
    if missing_mappings:
        raise ImportSpecError(
            f"{where}.columns references unknown expected columns {missing_mappings}."
        )
    ignored_columns = _string_mapping(raw["ignored_columns"], f"{where}.ignored_columns")
    missing_ignored = sorted(set(ignored_columns) - set(expected_columns))
    if missing_ignored:
        raise ImportSpecError(
            f"{where}.ignored_columns references unknown expected columns {missing_ignored}."
        )
    mapped_values = list(columns.values())
    if len(mapped_values) != len(set(mapped_values)):
        raise ImportSpecError(
            f"{where}.columns assigns one source column to multiple mapped dispositions."
        )
    mapped_columns = set(mapped_values)
    ignored_set = set(ignored_columns)
    conflicts = sorted(
        (mapped_columns & option_columns)
        | (mapped_columns & ignored_set)
        | (option_columns & ignored_set)
    )
    if conflicts:
        raise ImportSpecError(
            f"{where} source column disposition conflicts for {conflicts}; mapped, "
            "option, and ignored columns must be pairwise disjoint."
        )
    extra_field_policy = _enum(
        raw["extra_field_policy"],
        {"preserve_unmapped", "reject_unmapped"},
        f"{where}.extra_field_policy",
    )
    disposed_columns = mapped_columns | option_columns | ignored_set
    if extra_field_policy == "reject_unmapped":
        undisposed = sorted(set(expected_columns) - disposed_columns)
        if undisposed:
            raise ImportSpecError(
                f"{where} reject_unmapped requires one explicit disposition per source "
                f"column; missing {undisposed}."
            )
    path_text = _nonempty(raw["path"], f"{where}.path")
    return SourceArtifactSpec(
        source_id=_nonempty(raw["source_id"], f"{where}.source_id"),
        path=Path(path_text),
        logical_path=logical_path,
        expected_sha256=_sha256(raw["expected_sha256"], f"{where}.expected_sha256"),
        format=_enum(raw["format"], {"csv"}, f"{where}.format"),
        format_version=_nonempty(raw["format_version"], f"{where}.format_version"),
        classification=_enum(
            raw["classification"],
            {"raw", "canonical", "derived", "repaired", "aggregate_only"},
            f"{where}.classification",
        ),
        dialect=_build_dialect(raw["dialect"], f"{where}.dialect"),
        columns=columns,
        expected_columns=expected_columns,
        ignored_columns=ignored_columns,
        null_values=_strings(
            raw["null_values"],
            f"{where}.null_values",
            unique=True,
            allow_empty_items=True,
        ),
        numeric_columns=numeric_columns,
        option_mapping=option_mapping,
        extra_field_policy=extra_field_policy,
        preserve_namespace=_nonempty(
            raw["preserve_namespace"], f"{where}.preserve_namespace"
        ),
        source_run_id=_optional_string(raw["source_run_id"], f"{where}.source_run_id"),
        source_repository=_optional_string(
            raw["source_repository"], f"{where}.source_repository"
        ),
        source_commit=_optional_string(raw["source_commit"], f"{where}.source_commit"),
        notes=_canonical_mapping(raw["notes"], f"{where}.notes"),
    )


def _validate_identity_claim(
    raw: Mapping[str, Any], *, payload_key: str, digest_key: str, id_key: str,
    prefix: str, where: str
) -> None:
    payload = raw[payload_key]
    if raw[digest_key] != integrity_digest(payload):
        raise ImportSpecError(f"{where}.{digest_key} does not match its payload.")
    if raw[id_key] != short_id(prefix, payload):
        raise ImportSpecError(f"{where}.{id_key} does not match its payload.")


def _validate_dataset_native(value: Any, where: str) -> dict[str, Any]:
    keys = {
        "artifact_payload", "artifact_digest", "artifact_id",
        "selection_payload", "selection_digest", "selection_id",
    }
    raw = _exact_keys(value, keys, where)
    artifact = _exact_keys(
        raw["artifact_payload"], {"spec", "content_digest", "source"},
        f"{where}.artifact_payload",
    )
    spec = _exact_keys(
        artifact["spec"],
        {
            "benchmark", "split", "hf_path", "hf_subset", "source_revision",
            "normalization_version", "transforms", "output_name",
        },
        f"{where}.artifact_payload.spec",
    )
    normalized_artifact = {
        "spec": {
            "benchmark": _nonempty(spec["benchmark"], f"{where}.artifact_payload.spec.benchmark"),
            "split": _nonempty(spec["split"], f"{where}.artifact_payload.spec.split"),
            "hf_path": _optional_string(spec["hf_path"], f"{where}.artifact_payload.spec.hf_path"),
            "hf_subset": _optional_string(spec["hf_subset"], f"{where}.artifact_payload.spec.hf_subset"),
            "source_revision": _optional_string(spec["source_revision"], f"{where}.artifact_payload.spec.source_revision"),
            "normalization_version": _nonempty(spec["normalization_version"], f"{where}.artifact_payload.spec.normalization_version"),
            "transforms": list(_strings(spec["transforms"], f"{where}.artifact_payload.spec.transforms", unique=True)),
            "output_name": _optional_string(spec["output_name"], f"{where}.artifact_payload.spec.output_name"),
        },
        "content_digest": _sha256(artifact["content_digest"], f"{where}.artifact_payload.content_digest"),
        "source": _canonical_mapping(artifact["source"], f"{where}.artifact_payload.source"),
    }
    selection = _exact_keys(
        raw["selection_payload"],
        {"artifact_id", "content_digest", "sample_identities", "seed", "n_samples", "subject_filter"},
        f"{where}.selection_payload",
    )
    normalized_selection = {
        "artifact_id": _nonempty(selection["artifact_id"], f"{where}.selection_payload.artifact_id"),
        "content_digest": _sha256(selection["content_digest"], f"{where}.selection_payload.content_digest"),
        "sample_identities": list(_strings(selection["sample_identities"], f"{where}.selection_payload.sample_identities", unique=False)),
        "seed": _strict_int(selection["seed"], f"{where}.selection_payload.seed"),
        "n_samples": _optional_int(selection["n_samples"], f"{where}.selection_payload.n_samples", minimum=1),
        "subject_filter": list(_strings(selection["subject_filter"], f"{where}.selection_payload.subject_filter", unique=True)),
    }
    for index, digest in enumerate(normalized_selection["sample_identities"]):
        _sha256(digest, f"{where}.selection_payload.sample_identities[{index}]")
    if normalized_selection["subject_filter"] != sorted(normalized_selection["subject_filter"]):
        raise ImportSpecError(f"{where}.selection_payload.subject_filter must be sorted.")
    normalized = {
        "artifact_payload": normalized_artifact,
        "artifact_digest": _sha256(raw["artifact_digest"], f"{where}.artifact_digest"),
        "artifact_id": _nonempty(raw["artifact_id"], f"{where}.artifact_id"),
        "selection_payload": normalized_selection,
        "selection_digest": _sha256(raw["selection_digest"], f"{where}.selection_digest"),
        "selection_id": _nonempty(raw["selection_id"], f"{where}.selection_id"),
    }
    _validate_identity_claim(
        normalized, payload_key="artifact_payload", digest_key="artifact_digest",
        id_key="artifact_id", prefix="ds", where=where,
    )
    if normalized_selection["artifact_id"] != normalized["artifact_id"]:
        raise ImportSpecError(f"{where}.selection_payload.artifact_id is inconsistent.")
    _validate_identity_claim(
        normalized, payload_key="selection_payload", digest_key="selection_digest",
        id_key="selection_id", prefix="sel", where=where,
    )
    return normalized


def _validate_model_payload(value: Any, where: str) -> dict[str, Any]:
    raw = _mapping(value, where)
    backend = _enum(raw.get("backend"), {"dummy", "api", "huggingface"}, f"{where}.backend")
    allowed = {
        "dummy": {"backend", "model_name_or_path"},
        "api": {"backend", "provider", "model_name_or_path", "base_url", "generation_kwargs"},
        "huggingface": {"backend", "model", "device", "add_bos_token", "generation_kwargs", "loader"},
    }[backend]
    _exact_keys(raw, allowed, where)
    if backend == "dummy":
        return {
            "backend": backend,
            "model_name_or_path": _nonempty(
                raw["model_name_or_path"], f"{where}.model_name_or_path"
            ),
        }
    generation_keys = (
        {"max_new_tokens", "temperature"}
        if backend == "api"
        else {"max_new_tokens", "temperature", "do_sample"}
    )
    generation = _exact_keys(
        raw["generation_kwargs"], generation_keys, f"{where}.generation_kwargs"
    )
    normalized_generation: dict[str, Any] = {
        "max_new_tokens": _strict_int(
            generation["max_new_tokens"],
            f"{where}.generation_kwargs.max_new_tokens",
            minimum=1,
        ),
        "temperature": _strict_number(
            generation["temperature"], f"{where}.generation_kwargs.temperature"
        ),
    }
    if backend == "api":
        return {
            "backend": backend,
            "provider": _nonempty(raw["provider"], f"{where}.provider"),
            "model_name_or_path": _nonempty(
                raw["model_name_or_path"], f"{where}.model_name_or_path"
            ),
            "base_url": _optional_string(raw["base_url"], f"{where}.base_url"),
            "generation_kwargs": normalized_generation,
        }
    normalized_generation["do_sample"] = _strict_bool(
        generation["do_sample"], f"{where}.generation_kwargs.do_sample"
    )
    resolved = _mapping(raw["model"], f"{where}.model")
    kind = _enum(
        resolved.get("kind"), {"local", "huggingface-hub"}, f"{where}.model.kind"
    )
    if kind == "local":
        resolved = _exact_keys(
            resolved,
            {"kind", "logical_name", "content_digest", "file_count", "total_bytes"},
            f"{where}.model",
        )
        normalized_model = {
            "kind": kind,
            "logical_name": _nonempty(
                resolved["logical_name"], f"{where}.model.logical_name"
            ),
            "content_digest": _sha256(
                resolved["content_digest"], f"{where}.model.content_digest"
            ),
            "file_count": _strict_int(
                resolved["file_count"], f"{where}.model.file_count", minimum=1
            ),
            "total_bytes": _strict_int(
                resolved["total_bytes"], f"{where}.model.total_bytes", minimum=0
            ),
        }
    else:
        resolved = _exact_keys(
            resolved,
            {"kind", "repo_id", "requested_revision", "resolved_commit"},
            f"{where}.model",
        )
        resolved_commit = _nonempty(
            resolved["resolved_commit"], f"{where}.model.resolved_commit"
        )
        if not re.fullmatch(r"[0-9a-f]{40,64}", resolved_commit):
            raise ImportSpecError(
                f"{where}.model.resolved_commit must be a lowercase immutable commit."
            )
        normalized_model = {
            "kind": kind,
            "repo_id": _nonempty(resolved["repo_id"], f"{where}.model.repo_id"),
            "requested_revision": _optional_string(
                resolved["requested_revision"], f"{where}.model.requested_revision"
            ),
            "resolved_commit": resolved_commit,
        }
    loader = _exact_keys(
        raw["loader"], {"trust_remote_code", "torch_dtype"}, f"{where}.loader"
    )
    return {
        "backend": backend,
        "model": normalized_model,
        "device": _nonempty(raw["device"], f"{where}.device"),
        "add_bos_token": _strict_bool(raw["add_bos_token"], f"{where}.add_bos_token"),
        "generation_kwargs": normalized_generation,
        "loader": {
            "trust_remote_code": _strict_bool(
                loader["trust_remote_code"], f"{where}.loader.trust_remote_code"
            ),
            "torch_dtype": _enum(
                loader["torch_dtype"],
                {"float16", "float32"},
                f"{where}.loader.torch_dtype",
            ),
        },
    }


_IMPLEMENTATION_KEYS = {
    "qualified_name", "source_file", "source_digest", "distribution",
    "distribution_version", "package_tree_digest", "package_file_count",
}


def _validate_implementation(value: Any, where: str) -> dict[str, Any]:
    raw = _allowed_keys(value, _IMPLEMENTATION_KEYS, where)
    if "qualified_name" not in raw:
        raise ImportSpecError(f"Missing required field 'qualified_name' in {where}.")
    result = _canonical_mapping(raw, where)
    _nonempty(result["qualified_name"], f"{where}.qualified_name")
    for key in ("source_file", "distribution", "distribution_version"):
        if key in result:
            _nonempty(result[key], f"{where}.{key}")
    for key in ("source_digest", "package_tree_digest"):
        if key in result:
            _sha256(result[key], f"{where}.{key}")
    has_package_digest = "package_tree_digest" in result
    has_package_count = "package_file_count" in result
    if has_package_digest != has_package_count:
        raise ImportSpecError(
            f"{where}.package_tree_digest and {where}.package_file_count must appear together."
        )
    if has_package_count:
        _strict_int(result["package_file_count"], f"{where}.package_file_count", minimum=1)
    return result


def _validate_method_payload(value: Any, where: str) -> dict[str, Any]:
    raw = _exact_keys(
        value, {"name", "effective_params", "preflight", "implementation"}, where
    )
    preflight = raw["preflight"]
    if preflight is not None:
        preflight = _exact_keys(
            preflight, {"source", "split", "n"}, f"{where}.preflight"
        )
        preflight = {
            "source": _nonempty(preflight["source"], f"{where}.preflight.source"),
            "split": _nonempty(preflight["split"], f"{where}.preflight.split"),
            "n": _strict_int(preflight["n"], f"{where}.preflight.n", minimum=1),
        }
    return {
        "name": _nonempty(raw["name"], f"{where}.name"),
        "effective_params": _canonical_mapping(raw["effective_params"], f"{where}.effective_params"),
        "preflight": preflight,
        "implementation": _validate_implementation(raw["implementation"], f"{where}.implementation"),
    }


def _validate_prompt_payload(value: Any, where: str) -> dict[str, Any]:
    raw = _exact_keys(value, {"version", "files"}, where)
    files_raw = _mapping(raw["files"], f"{where}.files")
    template_names = {"direct_mcq", "free_text", "option_matching"}
    if set(files_raw) != template_names:
        raise ImportSpecError(
            f"{where}.files must contain exactly the current template names "
            f"{sorted(template_names)}."
        )
    files: dict[str, Any] = {}
    for name in ("direct_mcq", "free_text", "option_matching"):
        item = files_raw[name]
        record = _exact_keys(item, {"sha256", "content"}, f"{where}.files.{name}")
        content = record["content"]
        if not isinstance(content, str):
            raise ImportSpecError(f"{where}.files.{name}.content must be a string.")
        digest = _sha256(record["sha256"], f"{where}.files.{name}.sha256")
        if digest != integrity_digest(content):
            raise ImportSpecError(f"{where}.files.{name}.sha256 does not match content.")
        files[name] = {"sha256": digest, "content": content}
    return {"version": _nonempty(raw["version"], f"{where}.version"), "files": files}


def _validate_prompt_native(value: Any, where: str) -> dict[str, Any]:
    raw = _exact_keys(value, {"prompt_id", "version", "files"}, where)
    payload = _validate_prompt_payload(
        {"version": raw["version"], "files": raw["files"]}, where
    )
    prompt_id = _nonempty(raw["prompt_id"], f"{where}.prompt_id")
    expected_id = f"prompt_{integrity_digest(payload)[:16]}"
    if prompt_id != expected_id:
        raise ImportSpecError(
            f"{where}.prompt_id does not match prompt_bundle_identity semantics."
        )
    return {"prompt_id": prompt_id, **payload}


def validate_csv_dialect_identity(value: Any) -> dict[str, Any]:
    """Validate and normalize an identity-bearing CSV dialect declaration."""
    prepared = dict(value) if isinstance(value, Mapping) else value
    if isinstance(prepared, dict) and isinstance(
        prepared.get("line_terminators"), tuple
    ):
        prepared["line_terminators"] = list(prepared["line_terminators"])
    return asdict(_build_dialect(prepared, "parsing_policy.dialect"))


def validate_numeric_columns_identity(value: Any) -> list[dict[str, Any]]:
    """Validate identity-bearing numeric-column declarations."""
    if not isinstance(value, (list, tuple)):
        raise ImportSpecError("parsing_policy.numeric_columns must be a list.")
    records = [
        asdict(_build_numeric(item, f"parsing_policy.numeric_columns[{index}]"))
        for index, item in enumerate(value)
    ]
    columns = [record["source_column"] for record in records]
    if len(columns) != len(set(columns)):
        raise ImportSpecError(
            "parsing_policy.numeric_columns contains duplicate source columns."
        )
    return records


def validate_option_mapping_identity(value: Any) -> dict[str, Any]:
    """Validate and normalize an identity-bearing option mapping."""
    prepared = dict(value) if isinstance(value, Mapping) else value
    if isinstance(prepared, dict) and isinstance(
        prepared.get("ordered_columns"), tuple
    ):
        prepared["ordered_columns"] = list(prepared["ordered_columns"])
    return asdict(_build_option(prepared, "parsing_policy.option_mapping"))


def validate_native_model_payload(value: Any) -> dict[str, Any]:
    """Validate a claimed current-native model identity payload."""
    return _validate_model_payload(value, "native_model.payload")


def validate_native_method_payload(value: Any) -> dict[str, Any]:
    """Validate a claimed current-native method identity payload."""
    return _validate_method_payload(value, "native_method.payload")


def validate_implementation_identity_record(value: Any) -> dict[str, Any]:
    """Validate the closed runtime implementation-identity record shape."""
    return _validate_implementation(value, "implementation")


def _validate_simple_native(
    value: Any, where: str, prefix: str, payload_validator
) -> dict[str, Any]:
    id_key = f"{prefix}_id"
    raw = _exact_keys(value, {"payload", "digest", id_key}, where)
    normalized = {
        "payload": payload_validator(raw["payload"], f"{where}.payload"),
        "digest": _sha256(raw["digest"], f"{where}.digest"),
        id_key: _nonempty(raw[id_key], f"{where}.{id_key}"),
    }
    _validate_identity_claim(
        normalized, payload_key="payload", digest_key="digest", id_key=id_key,
        prefix=prefix, where=where,
    )
    return normalized


def _build_dataset(value: Any, index: int) -> DatasetReferenceSpec:
    where = f"datasets[{index}]"
    raw = _exact_keys(value, _DATASET_KEYS, where)
    native = raw["native_compatibility_identity"]
    selection_seed = _optional_int(raw["selection_seed"], f"{where}.selection_seed")
    selection_n_samples = _optional_int(
        raw["selection_n_samples"], f"{where}.selection_n_samples", minimum=1
    )
    selection_unknown_reasons = _string_mapping(
        raw["selection_unknown_reasons"], f"{where}.selection_unknown_reasons"
    )
    _validate_unknown_reasons(
        {
            "selection_seed": selection_seed,
            "selection_n_samples": selection_n_samples,
        },
        selection_unknown_reasons,
        f"{where}.selection_unknown_reasons",
    )
    return DatasetReferenceSpec(
        dataset_id=_nonempty(raw["dataset_id"], f"{where}.dataset_id"),
        benchmark_name=_nonempty(raw["benchmark_name"], f"{where}.benchmark_name"),
        split=_nonempty(raw["split"], f"{where}.split"),
        reference_kind=_enum(
            raw["reference_kind"],
            {"independent_input_snapshot", "profile_derived_reference_snapshot"},
            f"{where}.reference_kind",
        ),
        trust_label=_nonempty(raw["trust_label"], f"{where}.trust_label"),
        source_ids=_strings(raw["source_ids"], f"{where}.source_ids", nonempty=True, unique=True),
        selection_source_id=_nonempty(raw["selection_source_id"], f"{where}.selection_source_id"),
        expected_question_ids=_strings(raw["expected_question_ids"], f"{where}.expected_question_ids", nonempty=True, unique=True),
        selection_seed=selection_seed,
        selection_n_samples=selection_n_samples,
        subject_filter=_strings(raw["subject_filter"], f"{where}.subject_filter", unique=True),
        selection_unknown_reasons=selection_unknown_reasons,
        columns=_string_mapping(raw["columns"], f"{where}.columns"),
        revision=_optional_string(raw["revision"], f"{where}.revision"),
        fingerprint=_optional_string(raw["fingerprint"], f"{where}.fingerprint"),
        derivation=_canonical_mapping(raw["derivation"], f"{where}.derivation"),
        limitations=_strings(raw["limitations"], f"{where}.limitations"),
        native_compatibility_identity=(
            None if native is None else _validate_dataset_native(native, f"{where}.native_compatibility_identity")
        ),
    )


def _build_model(value: Any, index: int) -> ImportModelSpec:
    where = f"models[{index}]"
    raw = _exact_keys(value, _MODEL_KEYS, where)
    native = raw["native_compatibility_identity"]
    backend = _optional_string(raw["backend"], f"{where}.backend")
    provider = _optional_string(raw["provider"], f"{where}.provider")
    revision = _optional_string(raw["revision"], f"{where}.revision")
    unknown_reasons = _string_mapping(raw["unknown_reasons"], f"{where}.unknown_reasons")
    _validate_unknown_reasons(
        {"backend": backend, "provider": provider, "revision": revision},
        unknown_reasons,
        f"{where}.unknown_reasons",
    )
    return ImportModelSpec(
        model_key=_nonempty(raw["model_key"], f"{where}.model_key"),
        display_name=_nonempty(raw["display_name"], f"{where}.display_name"),
        backend=backend,
        provider=provider,
        revision=revision,
        effective_parameters=_canonical_mapping(raw["effective_parameters"], f"{where}.effective_parameters"),
        unknown_reasons=unknown_reasons,
        native_compatibility_identity=(
            None if native is None else _validate_simple_native(
                native, f"{where}.native_compatibility_identity", "model", _validate_model_payload
            )
        ),
    )


def _build_method(value: Any, index: int) -> ImportMethodSpec:
    where = f"methods[{index}]"
    raw = _exact_keys(value, _METHOD_KEYS, where)
    implementation = raw["implementation"]
    native = raw["native_compatibility_identity"]
    implementation = (
        None
        if implementation is None
        else _validate_implementation(implementation, f"{where}.implementation")
    )
    unknown_reasons = _string_mapping(raw["unknown_reasons"], f"{where}.unknown_reasons")
    _validate_unknown_reasons(
        {"implementation": implementation},
        unknown_reasons,
        f"{where}.unknown_reasons",
    )
    return ImportMethodSpec(
        method_key=_nonempty(raw["method_key"], f"{where}.method_key"),
        name=_nonempty(raw["name"], f"{where}.name"),
        effective_parameters=_canonical_mapping(raw["effective_parameters"], f"{where}.effective_parameters"),
        implementation=implementation,
        unknown_reasons=unknown_reasons,
        native_compatibility_identity=(
            None if native is None else _validate_simple_native(
                native, f"{where}.native_compatibility_identity", "method", _validate_method_payload
            )
        ),
    )


def _build_prompt(value: Any, index: int) -> ImportPromptSpec:
    where = f"prompts[{index}]"
    raw = _exact_keys(value, _PROMPT_KEYS, where)
    contents = raw["template_contents"]
    native = raw["native_compatibility_identity"]
    template_identity = _optional_string(
        raw["template_identity"], f"{where}.template_identity"
    )
    template_digest = _optional_sha256(
        raw["template_digest"], f"{where}.template_digest"
    )
    template_contents = (
        None
        if contents is None
        else _string_mapping(contents, f"{where}.template_contents")
    )
    unknown_reason = _optional_string(raw["unknown_reason"], f"{where}.unknown_reason")
    has_unknown = any(
        item is None for item in (template_identity, template_digest, template_contents)
    )
    if has_unknown and unknown_reason is None:
        raise ImportSpecError(
            f"{where}.unknown_reason must explain unrecoverable prompt fields."
        )
    if not has_unknown and unknown_reason is not None:
        raise ImportSpecError(
            f"{where}.unknown_reason contradicts fully known prompt fields."
        )
    return ImportPromptSpec(
        prompt_key=_nonempty(raw["prompt_key"], f"{where}.prompt_key"),
        template_identity=template_identity,
        template_digest=template_digest,
        template_contents=template_contents,
        unknown_reason=unknown_reason,
        native_compatibility_identity=(
            None
            if native is None
            else _validate_prompt_native(native, f"{where}.native_compatibility_identity")
        ),
    )


def _build_result_origin(value: Any, where: str) -> ResultOriginSpec:
    raw = _exact_keys(value, _RESULT_ORIGIN_KEYS, where)
    default = raw["default_prediction_origin"]
    per_question_raw = _mapping(
        raw["per_question_prediction_origins"],
        f"{where}.per_question_prediction_origins",
    )
    return ResultOriginSpec(
        derivation_origin=_enum(raw["derivation_origin"], _DERIVATION_ORIGINS, f"{where}.derivation_origin"),
        default_prediction_origin=(None if default is None else _enum(default, _PREDICTION_ORIGINS, f"{where}.default_prediction_origin")),
        per_question_prediction_origins={
            _nonempty(key, f"{where}.per_question_prediction_origins key"): _enum(
                item, _PREDICTION_ORIGINS, f"{where}.per_question_prediction_origins.{key}"
            )
            for key, item in per_question_raw.items()
        },
    )


def _build_condition(value: Any, index: int) -> ImportConditionSpec:
    where = f"conditions[{index}]"
    raw = _exact_keys(value, _CONDITION_KEYS, where)
    seed = _optional_int(raw["seed"], f"{where}.seed")
    calibration_identity = _canonical_mapping_or_none(
        raw["calibration_identity"], f"{where}.calibration_identity"
    )
    preflight_identity = _canonical_mapping_or_none(
        raw["preflight_identity"], f"{where}.preflight_identity"
    )
    unknown_reasons = _string_mapping(raw["unknown_reasons"], f"{where}.unknown_reasons")
    protocol_settings = _canonical_mapping(
        raw["protocol_settings"], f"{where}.protocol_settings"
    )
    unsupported_protocol = sorted(
        set(protocol_settings) - _PROTOCOL_SETTING_KEYS
    )
    if unsupported_protocol:
        raise ImportSpecError(
            f"{where}.protocol_settings contains unsupported field(s) "
            f"{unsupported_protocol}."
        )
    if "pride_modal_k_threshold" in protocol_settings:
        _strict_number(
            protocol_settings["pride_modal_k_threshold"],
            f"{where}.protocol_settings.pride_modal_k_threshold",
        )
    _validate_unknown_reasons(
        {
            "seed": seed,
            "calibration_identity": calibration_identity,
            "preflight_identity": preflight_identity,
        },
        unknown_reasons,
        f"{where}.unknown_reasons",
    )
    return ImportConditionSpec(
        condition_key=_nonempty(raw["condition_key"], f"{where}.condition_key"),
        source_ids=_strings(raw["source_ids"], f"{where}.source_ids", nonempty=True, unique=True),
        dataset_id=_nonempty(raw["dataset_id"], f"{where}.dataset_id"),
        model_key=_nonempty(raw["model_key"], f"{where}.model_key"),
        method_key=_nonempty(raw["method_key"], f"{where}.method_key"),
        prompt_key=_nonempty(raw["prompt_key"], f"{where}.prompt_key"),
        seed=seed,
        calibration_identity=calibration_identity,
        preflight_identity=preflight_identity,
        protocol_settings=protocol_settings,
        generation_parameters=_canonical_mapping(raw["generation_parameters"], f"{where}.generation_parameters"),
        unknown_reasons=unknown_reasons,
        expected_question_ids=_strings(raw["expected_question_ids"], f"{where}.expected_question_ids", nonempty=True, unique=True),
        evidence_status=_enum(raw["evidence_status"], _EVIDENCE_STATUSES, f"{where}.evidence_status"),
        scope_disposition=_enum(raw["scope_disposition"], _SCOPE_DISPOSITIONS, f"{where}.scope_disposition"),
        executable=_optional_bool(raw["executable"], f"{where}.executable"),
        qualifications=_mapping_sequence(raw["qualifications"], f"{where}.qualifications"),
        limitations=_mapping_sequence(raw["limitations"], f"{where}.limitations"),
        damaged_question_ids=_strings(raw["damaged_question_ids"], f"{where}.damaged_question_ids", unique=True),
        recoverable_question_ids=_strings(raw["recoverable_question_ids"], f"{where}.recoverable_question_ids", unique=True),
        result_origin=_build_result_origin(raw["result_origin"], f"{where}.result_origin"),
    )


def _build_authorization(value: Any, index: int) -> AuthorizationSpec:
    where = f"authorizations[{index}]"
    raw = _exact_keys(value, _AUTHORIZATION_KEYS, where)
    reasons_raw = _mapping(raw["condition_question_reasons"], f"{where}.condition_question_reasons")
    reasons = {
        _nonempty(condition, f"{where}.condition_question_reasons key"): _string_mapping(
            question_reasons, f"{where}.condition_question_reasons.{condition}"
        )
        for condition, question_reasons in reasons_raw.items()
    }
    authorization_type = _enum(
        raw["authorization_type"],
        {"inference_repair", "offline_transformation"},
        f"{where}.authorization_type",
    )
    executable = _strict_bool(raw["executable"], f"{where}.executable")
    expected_executable = authorization_type == "inference_repair"
    if executable is not expected_executable:
        raise ImportSpecError(
            f"{where}.authorization_type={authorization_type!r} requires "
            f"executable={expected_executable!r}."
        )
    return AuthorizationSpec(
        authorization_id=_nonempty(raw["authorization_id"], f"{where}.authorization_id"),
        authorization_type=authorization_type,
        source_id=_nonempty(raw["source_id"], f"{where}.source_id"),
        condition_question_reasons=reasons,
        authority=_nonempty(raw["authority"], f"{where}.authority"),
        purpose=_nonempty(raw["purpose"], f"{where}.purpose"),
        executable=executable,
        input_evidence_digests=_sha_mapping(raw["input_evidence_digests"], f"{where}.input_evidence_digests"),
        expected_snapshot_digests=_sha_mapping(raw["expected_snapshot_digests"], f"{where}.expected_snapshot_digests"),
    )


def _build_overlay(value: Any, index: int) -> OverlaySpec:
    where = f"overlays[{index}]"
    raw = _exact_keys(value, _OVERLAY_KEYS, where)
    return OverlaySpec(
        overlay_id=_nonempty(raw["overlay_id"], f"{where}.overlay_id"),
        base_run_path=Path(_nonempty(raw["base_run_path"], f"{where}.base_run_path")),
        base_condition_digest=_sha256(raw["base_condition_digest"], f"{where}.base_condition_digest"),
        base_realization_id=_nonempty(raw["base_realization_id"], f"{where}.base_realization_id"),
        base_realization_digest=_sha256(raw["base_realization_digest"], f"{where}.base_realization_digest"),
        base_evidence_digests=_sha_mapping(raw["base_evidence_digests"], f"{where}.base_evidence_digests"),
        base_validation_artifact_sha256=_sha256(raw["base_validation_artifact_sha256"], f"{where}.base_validation_artifact_sha256"),
        base_result_sha256=_optional_sha256(raw["base_result_sha256"], f"{where}.base_result_sha256"),
        source_id=_nonempty(raw["source_id"], f"{where}.source_id"),
        authorization_id=_nonempty(raw["authorization_id"], f"{where}.authorization_id"),
        replacement_reasons=_string_mapping(raw["replacement_reasons"], f"{where}.replacement_reasons"),
        result_origin=_build_result_origin(raw["result_origin"], f"{where}.result_origin"),
        lineage_notes=_canonical_mapping(raw["lineage_notes"], f"{where}.lineage_notes"),
        implementation=_canonical_mapping(raw["implementation"], f"{where}.implementation"),
        input_digest=_sha256(raw["input_digest"], f"{where}.input_digest"),
        preownership_output_digest=_sha256(raw["preownership_output_digest"], f"{where}.preownership_output_digest"),
        expected_evidence_status=_enum(raw["expected_evidence_status"], _EVIDENCE_STATUSES, f"{where}.expected_evidence_status"),
    )


def _unique(items: tuple[Any, ...], attribute: str, where: str) -> set[str]:
    values = [getattr(item, attribute) for item in items]
    if len(values) != len(set(values)):
        raise ImportSpecError(f"Duplicate {attribute} in {where}.")
    return set(values)


def _require_references(spec: ImportSpec) -> None:
    source_ids = _unique(spec.sources, "source_id", "sources")
    dataset_ids = _unique(spec.datasets, "dataset_id", "datasets")
    model_keys = _unique(spec.models, "model_key", "models")
    method_keys = _unique(spec.methods, "method_key", "methods")
    prompt_keys = _unique(spec.prompts, "prompt_key", "prompts")
    condition_keys = _unique(spec.conditions, "condition_key", "conditions")
    authorization_ids = _unique(spec.authorizations, "authorization_id", "authorizations")
    _unique(spec.overlays, "overlay_id", "overlays")

    for dataset in spec.datasets:
        missing = sorted(set(dataset.source_ids) - source_ids)
        if missing:
            raise ImportSpecError(
                f"Dataset {dataset.dataset_id!r} has unknown source reference(s) {missing}."
            )
        if dataset.selection_source_id not in dataset.source_ids:
            raise ImportSpecError(
                f"Dataset {dataset.dataset_id!r} selection_source_id must reference one of its source_ids."
            )
    dataset_by_id = {item.dataset_id: item for item in spec.datasets}
    conditions_by_key = {item.condition_key: item for item in spec.conditions}
    for condition in spec.conditions:
        references = (
            (condition.dataset_id, dataset_ids, "dataset"),
            (condition.model_key, model_keys, "model"),
            (condition.method_key, method_keys, "method"),
            (condition.prompt_key, prompt_keys, "prompt"),
        )
        for reference, known, label in references:
            if reference not in known:
                raise ImportSpecError(
                    f"Condition {condition.condition_key!r} has unknown {label} reference {reference!r}."
                )
        missing_sources = sorted(set(condition.source_ids) - source_ids)
        if missing_sources:
            raise ImportSpecError(
                f"Condition {condition.condition_key!r} has unknown source reference(s) {missing_sources}."
            )
        dataset_questions = set(dataset_by_id[condition.dataset_id].expected_question_ids)
        expected = set(condition.expected_question_ids)
        if not expected <= dataset_questions:
            raise ImportSpecError(
                f"Condition {condition.condition_key!r} expects question IDs absent from its dataset."
            )
        damaged = set(condition.damaged_question_ids)
        recoverable = set(condition.recoverable_question_ids)
        if not damaged <= expected or not recoverable <= expected:
            raise ImportSpecError(
                f"Condition {condition.condition_key!r} damage references unknown question IDs."
            )
        if not recoverable <= damaged:
            raise ImportSpecError(
                f"Condition {condition.condition_key!r} recoverable questions must also be damaged."
            )
        origin_questions = set(condition.result_origin.per_question_prediction_origins)
        if not origin_questions <= expected:
            raise ImportSpecError(
                f"Condition {condition.condition_key!r} has origins for unknown question IDs."
            )
        evaluable = (
            condition.evidence_status in {"complete", "qualified"}
            and condition.scope_disposition == "included"
        )
        if (
            evaluable
            and condition.result_origin.default_prediction_origin is None
            and origin_questions != expected
        ):
            missing_origins = sorted(expected - origin_questions)
            raise ImportSpecError(
                f"Condition {condition.condition_key!r} has evaluable rows without a "
                f"prediction origin: {missing_origins}."
            )

    for authorization in spec.authorizations:
        if authorization.source_id not in source_ids:
            raise ImportSpecError(
                f"Authorization {authorization.authorization_id!r} has an unknown source reference."
            )
        for condition_key, reasons in authorization.condition_question_reasons.items():
            if condition_key not in condition_keys:
                raise ImportSpecError(
                    f"Authorization {authorization.authorization_id!r} has an unknown condition reference."
                )
            allowed_questions = set(conditions_by_key[condition_key].expected_question_ids)
            if not set(reasons) <= allowed_questions:
                raise ImportSpecError(
                    f"Authorization {authorization.authorization_id!r} references unknown question IDs."
                )
    for overlay in spec.overlays:
        if overlay.source_id not in source_ids:
            raise ImportSpecError(f"Overlay {overlay.overlay_id!r} has an unknown source reference.")
        if overlay.authorization_id not in authorization_ids:
            raise ImportSpecError(
                f"Overlay {overlay.overlay_id!r} has an unknown authorization reference."
            )


def _build_import_spec(raw_value: Any) -> ImportSpec:
    raw = _exact_keys(raw_value, _TOP_LEVEL_KEYS, "top level")
    credentials = sorted(_credential_keys_in(raw))
    if credentials:
        raise ImportSpecError(
            f"Import specification contains credential-named field(s) {credentials}; "
            "credentials must never be stored in import declarations."
        )
    sources = tuple(
        _build_source(item, index)
        for index, item in enumerate(_sequence(raw["sources"], "sources"))
    )
    datasets = tuple(
        _build_dataset(item, index)
        for index, item in enumerate(_sequence(raw["datasets"], "datasets"))
    )
    models = tuple(
        _build_model(item, index)
        for index, item in enumerate(_sequence(raw["models"], "models"))
    )
    methods = tuple(
        _build_method(item, index)
        for index, item in enumerate(_sequence(raw["methods"], "methods"))
    )
    prompts = tuple(
        _build_prompt(item, index)
        for index, item in enumerate(_sequence(raw["prompts"], "prompts"))
    )
    conditions = tuple(
        _build_condition(item, index)
        for index, item in enumerate(_sequence(raw["conditions"], "conditions"))
    )
    authorizations = tuple(
        _build_authorization(item, index)
        for index, item in enumerate(_sequence(raw["authorizations"], "authorizations"))
    )
    overlays = tuple(
        _build_overlay(item, index)
        for index, item in enumerate(_sequence(raw["overlays"], "overlays"))
    )
    for name, values in (
        ("sources", sources), ("datasets", datasets), ("models", models),
        ("methods", methods), ("prompts", prompts), ("conditions", conditions),
    ):
        if not values:
            raise ImportSpecError(f"{name} must be a non-empty list.")
    metrics = _strings(raw["metrics"], "metrics", nonempty=True, unique=True)
    invalid_metrics = sorted(set(metrics) - set(BUILTIN_METRICS))
    if invalid_metrics:
        raise ImportSpecError(
            f"Import metrics must use the closed built-in registry; unknown entries: {invalid_metrics}. "
            f"Built-ins: {sorted(BUILTIN_METRICS)}."
        )
    spec = ImportSpec(
        schema_version=_enum(
            raw["schema_version"], {"choicebench.import-spec.v1"}, "schema_version"
        ),
        import_name=_nonempty(raw["import_name"], "import_name"),
        sources=sources,
        datasets=datasets,
        models=models,
        methods=methods,
        prompts=prompts,
        conditions=conditions,
        authorizations=authorizations,
        overlays=overlays,
        metrics=metrics,
        provenance=_build_provenance(raw["provenance"]),
        audit=_canonical_mapping(raw["audit"], "audit"),
    )
    _require_references(spec)
    return spec


def load_import_spec(path: Path) -> ImportSpec:
    """Load a YAML import declaration using a closed, non-coercing schema."""
    source_path = Path(path)
    try:
        raw = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ImportSpecError(f"Could not parse {source_path} as safe YAML: {exc}") from exc
    except (OSError, UnicodeError) as exc:
        raise ImportSpecError(f"Could not read import specification {source_path}: {exc}") from exc
    if not isinstance(raw, Mapping):
        raise ImportSpecError(
            f"{source_path} must contain a YAML mapping at the top level."
        )
    return _build_import_spec(raw)


def stable_import_projection(spec: ImportSpec) -> dict[str, Any]:
    """Return the identity-bearing declaration without machine-local locations."""
    if not isinstance(spec, ImportSpec):
        raise TypeError(f"spec must be an ImportSpec; got {type(spec).__name__}.")
    projection = asdict(spec)
    projection.pop("audit", None)
    for source in projection["sources"]:
        source.pop("path", None)
    for overlay in projection["overlays"]:
        overlay.pop("base_run_path", None)
    return canonicalize(projection)


def import_spec_digest(spec: ImportSpec) -> str:
    """Return the stable full SHA-256 identity of an import declaration."""
    return integrity_digest(stable_import_projection(spec))
