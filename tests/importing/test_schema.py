from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from choicebench.identity import integrity_digest, short_id
from choicebench.importing.schema import (
    ImportSpecError,
    import_spec_digest,
    load_import_spec,
    stable_import_projection,
)
from choicebench.metrics import BUILTIN_METRICS


SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64


def _native_identity(prefix: str, payload: dict) -> dict:
    return {
        "payload": payload,
        "digest": integrity_digest(payload),
        f"{prefix}_id": short_id(prefix, payload),
    }


@pytest.fixture
def minimal_raw(tmp_path: Path) -> dict:
    return {
        "schema_version": "choicebench.import-spec.v1",
        "import_name": "minimal historical results",
        "sources": [
            {
                "source_id": "results",
                "path": str(tmp_path / "results.csv"),
                "logical_path": "freeze/results.csv",
                "expected_sha256": SHA_A,
                "format": "csv",
                "format_version": "producer-v1",
                "classification": "raw",
                "dialect": {
                    "encoding": "utf-8",
                    "bom_policy": "forbid",
                    "decoding_errors": "strict",
                    "delimiter": ",",
                    "quote_character": '"',
                    "escape_character": None,
                    "double_quote": True,
                    "line_terminators": ["crlf", "lf", "cr"],
                    "mixed_line_terminators": "allow",
                    "final_record_without_terminator": "allow",
                    "blank_record_policy": "reject",
                    "skip_initial_space": False,
                    "header": "first_logical_record",
                    "strict_syntax": True,
                },
                "columns": {
                    "question_id": "qid",
                    "prediction": "answer",
                    "gold": "gold",
                },
                "expected_columns": [
                    "qid",
                    "question",
                    "choice_a",
                    "choice_b",
                    "answer",
                    "gold",
                    "score",
                ],
                "ignored_columns": {"score": "producer aggregate only"},
                "null_values": ["", "NA"],
                "numeric_columns": [
                    {
                        "source_column": "score",
                        "value_type": "float",
                        "null_allowed": True,
                        "finite_only": True,
                    }
                ],
                "option_mapping": {
                    "mode": "ordered_columns",
                    "ordered_columns": ["choice_a", "choice_b"],
                    "structured_column": None,
                    "structured_label_key": None,
                    "structured_text_key": None,
                },
                "extra_field_policy": "preserve_unmapped",
                "preserve_namespace": "producer",
                "source_run_id": None,
                "source_repository": None,
                "source_commit": None,
                "notes": {},
            }
        ],
        "datasets": [
            {
                "dataset_id": "dataset",
                "benchmark_name": "historical-benchmark",
                "split": "test",
                "reference_kind": "independent_input_snapshot",
                "trust_label": "producer-supplied",
                "source_ids": ["results"],
                "selection_source_id": "results",
                "expected_question_ids": ["q1", "q2"],
                "selection_seed": None,
                "selection_n_samples": None,
                "subject_filter": [],
                "selection_unknown_reasons": {
                    "selection_seed": "not recorded by producer",
                    "selection_n_samples": "not recorded by producer",
                },
                "columns": {
                    "question_id": "qid",
                    "question_text": "question",
                    "correct_option": "gold",
                },
                "revision": None,
                "fingerprint": None,
                "derivation": {},
                "limitations": ["publisher revision was not recorded"],
                "native_compatibility_identity": None,
            }
        ],
        "models": [
            {
                "model_key": "model",
                "display_name": "historical-model",
                "backend": None,
                "provider": None,
                "revision": None,
                "effective_parameters": {},
                "unknown_reasons": {
                    "backend": "not recorded by producer",
                    "provider": "not recorded by producer",
                    "revision": "not recorded by producer",
                },
                "native_compatibility_identity": None,
            }
        ],
        "methods": [
            {
                "method_key": "method",
                "name": "historical-direct",
                "effective_parameters": {},
                "implementation": None,
                "unknown_reasons": {
                    "implementation": "not recorded by producer"
                },
                "native_compatibility_identity": None,
            }
        ],
        "prompts": [
            {
                "prompt_key": "prompt",
                "template_identity": None,
                "template_digest": None,
                "template_contents": None,
                "unknown_reason": "not recorded by producer",
                "native_compatibility_identity": None,
            }
        ],
        "conditions": [
            {
                "condition_key": "condition",
                "source_ids": ["results"],
                "dataset_id": "dataset",
                "model_key": "model",
                "method_key": "method",
                "prompt_key": "prompt",
                "seed": None,
                "calibration_identity": None,
                "preflight_identity": None,
                "protocol_settings": {},
                "generation_parameters": {},
                "unknown_reasons": {"seed": "not recorded by producer"},
                "expected_question_ids": ["q1", "q2"],
                "evidence_status": "complete",
                "scope_disposition": "included",
                "executable": None,
                "qualifications": [],
                "limitations": [],
                "damaged_question_ids": [],
                "recoverable_question_ids": [],
                "result_origin": {
                    "derivation_origin": "external_import",
                    "default_prediction_origin": "external_historical_inference",
                    "per_question_prediction_origins": {},
                },
            }
        ],
        "authorizations": [],
        "overlays": [],
        "metrics": ["accuracy"],
        "provenance": {
            "producer_request_id": {
                "value": None,
                "reason": "not recorded by producer",
            }
        },
        "audit": {
            "source_path": str(tmp_path),
            "imported_at": "2026-07-18T00:00:00Z",
        },
    }


def _load(tmp_path: Path, raw: object):
    path = tmp_path / "import.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return load_import_spec(path)


@pytest.fixture
def minimal_spec(tmp_path: Path, minimal_raw: dict):
    return _load(tmp_path, minimal_raw)


@pytest.fixture
def minimal_overlay_spec(tmp_path: Path, minimal_raw: dict):
    raw = deepcopy(minimal_raw)
    raw["authorizations"] = [
        {
            "authorization_id": "auth",
            "authorization_type": "inference_repair",
            "source_id": "results",
            "condition_question_reasons": {
                "condition": {"q2": "repair explicitly approved"}
            },
            "authority": "benchmark owner",
            "purpose": "repair damaged prediction",
            "executable": True,
            "input_evidence_digests": {"results": SHA_A},
            "expected_snapshot_digests": {"q2": SHA_B},
        }
    ]
    raw["overlays"] = [
        {
            "overlay_id": "overlay",
            "base_run_path": str(tmp_path / "base-run"),
            "base_condition_digest": SHA_A,
            "base_realization_id": "realization_base",
            "base_realization_digest": SHA_B,
            "base_evidence_digests": {"results": SHA_A},
            "base_validation_artifact_sha256": SHA_C,
            "base_result_sha256": None,
            "source_id": "results",
            "authorization_id": "auth",
            "replacement_reasons": {"q2": "malformed producer row"},
            "result_origin": {
                "derivation_origin": "repair_overlay",
                "default_prediction_origin": None,
                "per_question_prediction_origins": {
                    "q2": "external_repair_inference"
                },
            },
            "lineage_notes": {},
            "implementation": {"name": "approved repair"},
            "input_digest": SHA_A,
            "preownership_output_digest": SHA_B,
            "expected_evidence_status": "qualified",
        }
    ]
    return _load(tmp_path, raw)


def test_loads_valid_minimal_yaml(minimal_spec):
    assert minimal_spec.schema_version == "choicebench.import-spec.v1"
    assert minimal_spec.sources[0].path.is_absolute()
    assert minimal_spec.sources[0].numeric_columns[0].value_type == "float"
    assert minimal_spec.sources[0].option_mapping.mode == "ordered_columns"
    assert minimal_spec.metrics == ("accuracy",)


def test_safe_loader_rejects_non_mapping_and_python_object_tag(tmp_path: Path):
    path = tmp_path / "bad.yaml"
    path.write_text("- not\n- a\n- mapping\n", encoding="utf-8")
    with pytest.raises(ImportSpecError, match="top level"):
        load_import_spec(path)

    path.write_text("!!python/object/apply:os.system [['echo unsafe']]\n", encoding="utf-8")
    with pytest.raises(ImportSpecError, match="YAML"):
        load_import_spec(path)


@pytest.mark.parametrize(
    ("section", "unknown_key"),
    [
        (None, "output_root"),
        ("sources", "output_path"),
        ("dialect", "sniff"),
        ("numeric_columns", "coerce"),
        ("option_mapping", "labels"),
        ("datasets", "paper_name"),
        ("models", "endpoint"),
        ("methods", "entry_point"),
        ("prompts", "prompt_path"),
        ("conditions", "import_state"),
        ("result_origin", "producer"),
        ("authorizations", "approved_at"),
        ("overlays", "output_path"),
    ],
)
def test_rejects_unknown_keys_at_every_schema_layer(
    tmp_path: Path, minimal_raw: dict, section: str | None, unknown_key: str
):
    raw = deepcopy(minimal_raw)
    if section is None:
        raw[unknown_key] = "unsafe"
    elif section == "dialect":
        raw["sources"][0]["dialect"][unknown_key] = True
    elif section == "numeric_columns":
        raw["sources"][0]["numeric_columns"][0][unknown_key] = True
    elif section == "option_mapping":
        raw["sources"][0]["option_mapping"][unknown_key] = []
    elif section == "result_origin":
        raw["conditions"][0]["result_origin"][unknown_key] = "historical"
    elif section == "authorizations":
        raw["authorizations"] = deepcopy(
            minimal_overlay_raw(raw)["authorizations"]
        )
        raw["authorizations"][0][unknown_key] = "later"
    elif section == "overlays":
        overlay_raw = minimal_overlay_raw(raw)
        raw["authorizations"] = overlay_raw["authorizations"]
        raw["overlays"] = overlay_raw["overlays"]
        raw["overlays"][0][unknown_key] = "unsafe"
    else:
        raw[section][0][unknown_key] = "unsafe"
    with pytest.raises(ImportSpecError, match="Unknown field"):
        _load(tmp_path, raw)


def minimal_overlay_raw(raw: dict) -> dict:
    result = deepcopy(raw)
    result["authorizations"] = [
        {
            "authorization_id": "auth",
            "authorization_type": "inference_repair",
            "source_id": "results",
            "condition_question_reasons": {"condition": {"q2": "approved"}},
            "authority": "owner",
            "purpose": "repair",
            "executable": True,
            "input_evidence_digests": {"results": SHA_A},
            "expected_snapshot_digests": {"q2": SHA_B},
        }
    ]
    result["overlays"] = [
        {
            "overlay_id": "overlay",
            "base_run_path": "/audit/base",
            "base_condition_digest": SHA_A,
            "base_realization_id": "realization_base",
            "base_realization_digest": SHA_B,
            "base_evidence_digests": {"results": SHA_A},
            "base_validation_artifact_sha256": SHA_C,
            "base_result_sha256": None,
            "source_id": "results",
            "authorization_id": "auth",
            "replacement_reasons": {"q2": "damaged"},
            "result_origin": {
                "derivation_origin": "repair_overlay",
                "default_prediction_origin": None,
                "per_question_prediction_origins": {
                    "q2": "external_repair_inference"
                },
            },
            "lineage_notes": {},
            "implementation": {"name": "repair"},
            "input_digest": SHA_A,
            "preownership_output_digest": SHA_B,
            "expected_evidence_status": "qualified",
        }
    ]
    return result


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("sources", 0, "dialect", "double_quote"), 1),
        (("sources", 0, "numeric_columns", 0, "null_allowed"), "false"),
        (("conditions", 0, "seed"), True),
        (("conditions", 0, "executable"), 0),
        (("sources", 0, "notes", "temperature"), float("nan")),
        (("models", 0, "effective_parameters", "temperature"), float("inf")),
    ],
)
def test_rejects_coerced_booleans_integers_and_nonfinite_numbers(
    tmp_path: Path, minimal_raw: dict, path: tuple, value: object
):
    raw = deepcopy(minimal_raw)
    target = raw
    for part in path[:-1]:
        target = target[part]
    target[path[-1]] = value
    with pytest.raises(ImportSpecError, match="must be|finite"):
        _load(tmp_path, raw)


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("sources", 0, "expected_sha256"), "abc"),
        (("conditions", 0, "evidence_status"), "verified"),
        (("conditions", 0, "scope_disposition"), "paper"),
        (("conditions", 0, "result_origin", "derivation_origin"), "native"),
        (("sources", 0, "classification"), "trusted"),
    ],
)
def test_rejects_invalid_sha256_and_enums(
    tmp_path: Path, minimal_raw: dict, path: tuple, value: str
):
    raw = deepcopy(minimal_raw)
    target = raw
    for part in path[:-1]:
        target = target[part]
    target[path[-1]] = value
    with pytest.raises(ImportSpecError):
        _load(tmp_path, raw)


@pytest.mark.parametrize("credential_key", ["api_key", "nested_access_token", "password"])
def test_rejects_recursive_credential_named_keys(
    tmp_path: Path, minimal_raw: dict, credential_key: str
):
    raw = deepcopy(minimal_raw)
    raw["sources"][0]["notes"] = {"outer": [{"inner": {credential_key: "secret"}}]}
    with pytest.raises(ImportSpecError, match="credential-named"):
        _load(tmp_path, raw)


@pytest.mark.parametrize(
    "metric",
    [
        "package.metrics:UnsafeMetric",
        "package.metrics.UnsafeMetric",
        "entry-point://unsafe",
        "/tmp/metric.py:Metric",
        "./metric.py",
    ],
)
def test_import_metrics_are_closed_to_builtin_registry(
    tmp_path: Path, minimal_raw: dict, metric: str
):
    raw = deepcopy(minimal_raw)
    raw["metrics"] = [metric]
    with pytest.raises(ImportSpecError, match="built-in"):
        _load(tmp_path, raw)

    assert set(BUILTIN_METRICS)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("delimiter", "::"),
        ("delimiter", "é"),
        ("quote_character", ","),
        ("escape_character", '"'),
        ("encoding", "utf-8-sig"),
        ("decoding_errors", "replace"),
    ],
)
def test_csv_dialect_is_fixed_utf8_strict_and_uses_distinct_ascii_bytes(
    tmp_path: Path, minimal_raw: dict, field: str, value: object
):
    raw = deepcopy(minimal_raw)
    raw["sources"][0]["dialect"][field] = value
    with pytest.raises(ImportSpecError):
        _load(tmp_path, raw)


def test_csv_dialect_accepts_distinct_ascii_escape(tmp_path: Path, minimal_raw: dict):
    raw = deepcopy(minimal_raw)
    raw["sources"][0]["dialect"].update(
        {"delimiter": ";", "quote_character": "'", "escape_character": "\\"}
    )
    spec = _load(tmp_path, raw)
    assert spec.sources[0].dialect.escape_character == "\\"


def test_structured_json_options_and_reject_unmapped_policy(
    tmp_path: Path, minimal_raw: dict
):
    raw = deepcopy(minimal_raw)
    source = raw["sources"][0]
    source["expected_columns"].append("options_json")
    source["option_mapping"] = {
        "mode": "structured_json",
        "ordered_columns": [],
        "structured_column": "options_json",
        "structured_label_key": "label",
        "structured_text_key": "text",
    }
    source["extra_field_policy"] = "reject_unmapped"
    spec = _load(tmp_path, raw)
    assert spec.sources[0].option_mapping.structured_column == "options_json"
    assert spec.sources[0].extra_field_policy == "reject_unmapped"


@pytest.mark.parametrize(
    "option_mapping",
    [
        {
            "mode": "ordered_columns",
            "ordered_columns": [],
            "structured_column": None,
            "structured_label_key": None,
            "structured_text_key": None,
        },
        {
            "mode": "ordered_columns",
            "ordered_columns": ["choice_a"],
            "structured_column": "options_json",
            "structured_label_key": None,
            "structured_text_key": None,
        },
        {
            "mode": "structured_json",
            "ordered_columns": [],
            "structured_column": "options_json",
            "structured_label_key": None,
            "structured_text_key": "text",
        },
        {
            "mode": "structured_json",
            "ordered_columns": ["choice_a"],
            "structured_column": "options_json",
            "structured_label_key": "label",
            "structured_text_key": "text",
        },
    ],
)
def test_rejects_incomplete_or_contradictory_option_declarations(
    tmp_path: Path, minimal_raw: dict, option_mapping: dict
):
    raw = deepcopy(minimal_raw)
    raw["sources"][0]["option_mapping"] = option_mapping
    with pytest.raises(ImportSpecError, match="option_mapping"):
        _load(tmp_path, raw)


def test_rejects_duplicate_numeric_column_rules(tmp_path: Path, minimal_raw: dict):
    raw = deepcopy(minimal_raw)
    raw["sources"][0]["numeric_columns"].append(
        deepcopy(raw["sources"][0]["numeric_columns"][0])
    )
    with pytest.raises(ImportSpecError, match="Duplicate numeric"):
        _load(tmp_path, raw)


@pytest.mark.parametrize(
    ("section", "id_field"),
    [
        ("sources", "source_id"),
        ("datasets", "dataset_id"),
        ("models", "model_key"),
        ("methods", "method_key"),
        ("prompts", "prompt_key"),
        ("conditions", "condition_key"),
    ],
)
def test_rejects_duplicate_declared_ids(
    tmp_path: Path, minimal_raw: dict, section: str, id_field: str
):
    raw = deepcopy(minimal_raw)
    raw[section].append(deepcopy(raw[section][0]))
    assert raw[section][0][id_field] == raw[section][1][id_field]
    with pytest.raises(ImportSpecError, match="Duplicate"):
        _load(tmp_path, raw)


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("datasets", 0, "source_ids"), ["missing"]),
        (("datasets", 0, "selection_source_id"), "missing"),
        (("conditions", 0, "dataset_id"), "missing"),
        (("conditions", 0, "model_key"), "missing"),
        (("conditions", 0, "method_key"), "missing"),
        (("conditions", 0, "prompt_key"), "missing"),
    ],
)
def test_rejects_missing_references(
    tmp_path: Path, minimal_raw: dict, path: tuple, value: object
):
    raw = deepcopy(minimal_raw)
    target = raw
    for part in path[:-1]:
        target = target[part]
    target[path[-1]] = value
    with pytest.raises(ImportSpecError, match="reference|unknown"):
        _load(tmp_path, raw)


def test_unknown_provenance_is_explicit_null_with_reason(
    tmp_path: Path, minimal_raw: dict
):
    spec = _load(tmp_path, minimal_raw)
    assert spec.provenance["producer_request_id"] == {
        "value": None,
        "reason": "not recorded by producer",
    }

    for bad in (
        None,
        {"value": None},
        {"value": None, "reason": ""},
        {"value": None, "reason": "unknown", "guess": "request-1"},
    ):
        raw = deepcopy(minimal_raw)
        raw["provenance"]["producer_request_id"] = bad
        with pytest.raises(ImportSpecError, match="provenance"):
            _load(tmp_path, raw)


def test_rejects_paths_in_stable_provenance(tmp_path: Path, minimal_raw: dict):
    raw = deepcopy(minimal_raw)
    raw["provenance"]["producer_path"] = {
        "value": "/home/producer/results.csv",
        "reason": None,
    }
    with pytest.raises(ImportSpecError, match="path"):
        _load(tmp_path, raw)


def _native_compatibility_payloads() -> dict[str, dict]:
    artifact_payload = {
        "spec": {
            "benchmark": "toy",
            "split": "test",
            "hf_path": None,
            "hf_subset": None,
            "source_revision": None,
            "normalization_version": "2",
            "transforms": [],
            "output_name": "toy",
        },
        "content_digest": SHA_A,
        "source": {},
    }
    selection_payload = {
        "artifact_id": short_id("ds", artifact_payload),
        "content_digest": SHA_B,
        "sample_identities": [SHA_A, SHA_B],
        "seed": 7,
        "n_samples": None,
        "subject_filter": [],
    }
    dataset_identity = {
        "artifact_payload": artifact_payload,
        "artifact_digest": integrity_digest(artifact_payload),
        "artifact_id": short_id("ds", artifact_payload),
        "selection_payload": selection_payload,
        "selection_digest": integrity_digest(selection_payload),
        "selection_id": short_id("sel", selection_payload),
    }
    model_payload = {"backend": "dummy", "model_name_or_path": "dummy-model"}
    method_payload = {
        "name": "direct_mcq",
        "effective_params": {},
        "preflight": None,
        "implementation": {
            "qualified_name": "choicebench.methods.direct_mcq:DirectMCQRunner",
        },
    }
    prompt_payload = {
        "version": "v1",
        "files": {
            "direct_mcq": {
                "sha256": integrity_digest("prompt"),
                "content": "prompt",
            }
        },
    }
    return {
        "datasets": dataset_identity,
        "models": _native_identity("model", model_payload),
        "methods": _native_identity("method", method_payload),
        "prompts": _native_identity("prompt", prompt_payload),
    }


def test_validates_exact_native_compatibility_identities(
    tmp_path: Path, minimal_raw: dict
):
    raw = deepcopy(minimal_raw)
    identities = _native_compatibility_payloads()
    for section, identity in identities.items():
        raw[section][0]["native_compatibility_identity"] = identity
    spec = _load(tmp_path, raw)
    assert spec.models[0].native_compatibility_identity["model_id"].startswith(
        "model_"
    )


@pytest.mark.parametrize(
    "payload",
    [
        {
            "backend": "api",
            "provider": "openai",
            "model_name_or_path": "gpt-test",
            "base_url": None,
            "generation_kwargs": {
                "max_new_tokens": 32,
                "temperature": 0.0,
                "untrusted": True,
            },
        },
        {
            "backend": "huggingface",
            "model": {
                "kind": "huggingface-hub",
                "repo_id": "org/model",
                "requested_revision": None,
                "resolved_commit": "a" * 40,
                "untrusted": True,
            },
            "device": "cuda",
            "add_bos_token": True,
            "generation_kwargs": {
                "max_new_tokens": 32,
                "temperature": 0.0,
                "do_sample": False,
            },
            "loader": {"trust_remote_code": True, "torch_dtype": "float16"},
        },
        {
            "backend": "huggingface",
            "model": {
                "kind": "local",
                "logical_name": "model",
                "content_digest": SHA_A,
                "file_count": 1,
                "total_bytes": 10,
            },
            "device": "cuda",
            "add_bos_token": True,
            "generation_kwargs": {
                "max_new_tokens": 32,
                "temperature": 0.0,
                "do_sample": False,
            },
            "loader": {"trust_remote_code": 1, "torch_dtype": "float16"},
        },
    ],
)
def test_rejects_unknown_or_coerced_nested_native_model_fields(
    tmp_path: Path, minimal_raw: dict, payload: dict
):
    raw = deepcopy(minimal_raw)
    raw["models"][0]["native_compatibility_identity"] = _native_identity(
        "model", payload
    )
    with pytest.raises(ImportSpecError, match="native_compatibility_identity"):
        _load(tmp_path, raw)


def test_rejects_unknown_native_method_preflight_field(
    tmp_path: Path, minimal_raw: dict
):
    raw = deepcopy(minimal_raw)
    payload = deepcopy(_native_compatibility_payloads()["methods"]["payload"])
    payload["preflight"] = {
        "source": "benchmark",
        "split": "validation",
        "n": 100,
        "untrusted": True,
    }
    raw["methods"][0]["native_compatibility_identity"] = _native_identity(
        "method", payload
    )
    with pytest.raises(ImportSpecError, match="native_compatibility_identity"):
        _load(tmp_path, raw)


@pytest.mark.parametrize("section", ["datasets", "models", "methods", "prompts"])
@pytest.mark.parametrize("mutation", ["unknown", "partial", "mismatched_digest", "mismatched_id"])
def test_rejects_unverified_native_compatibility_claims(
    tmp_path: Path, minimal_raw: dict, section: str, mutation: str
):
    raw = deepcopy(minimal_raw)
    identity = deepcopy(_native_compatibility_payloads()[section])
    if mutation == "unknown":
        identity["claimed_native_id"] = "native"
    elif mutation == "partial":
        identity.pop(next(iter(identity)))
    elif mutation == "mismatched_digest":
        digest_key = "artifact_digest" if section == "datasets" else "digest"
        identity[digest_key] = SHA_C
    else:
        id_key = {
            "datasets": "selection_id",
            "models": "model_id",
            "methods": "method_id",
            "prompts": "prompt_id",
        }[section]
        identity[id_key] = "claimed_arbitrarily"
    raw[section][0]["native_compatibility_identity"] = identity
    with pytest.raises(ImportSpecError, match="native_compatibility_identity"):
        _load(tmp_path, raw)


def test_audit_locations_do_not_change_import_spec_digest(
    minimal_spec, minimal_overlay_spec
):
    moved = replace(
        minimal_spec,
        audit={"source_path": "/different/host", "imported_at": "later"},
    )
    assert import_spec_digest(moved) == import_spec_digest(minimal_spec)

    moved_source = replace(
        minimal_spec,
        sources=(
            replace(
                minimal_spec.sources[0], path=Path("/other/freeze/source.csv")
            ),
        ),
    )
    assert import_spec_digest(moved_source) == import_spec_digest(minimal_spec)

    moved_base = replace(
        minimal_overlay_spec,
        overlays=(
            replace(
                minimal_overlay_spec.overlays[0],
                base_run_path=Path("/other/home/runs/base"),
            ),
        ),
    )
    assert import_spec_digest(moved_base) == import_spec_digest(
        minimal_overlay_spec
    )


def mutate_identity_field(spec, change: str):
    source = spec.sources[0]
    condition = spec.conditions[0]
    if change == "mapping":
        return replace(spec, sources=(replace(source, columns={**source.columns, "x": "y"}),))
    if change == "dialect":
        return replace(
            spec,
            sources=(replace(source, dialect=replace(source.dialect, delimiter=";")),),
        )
    if change == "numeric_policy":
        numeric = source.numeric_columns[0]
        return replace(
            spec,
            sources=(
                replace(
                    source,
                    numeric_columns=(replace(numeric, null_allowed=False),),
                ),
            ),
        )
    if change == "option_policy":
        option = source.option_mapping
        return replace(
            spec,
            sources=(
                replace(
                    source,
                    option_mapping=replace(
                        option, ordered_columns=tuple(reversed(option.ordered_columns))
                    ),
                ),
            ),
        )
    if change == "extra_field_policy":
        return replace(
            spec,
            sources=(replace(source, extra_field_policy="reject_unmapped"),),
        )
    if change == "status":
        return replace(
            spec,
            conditions=(replace(condition, evidence_status="qualified"),),
        )
    if change == "scope":
        return replace(
            spec,
            conditions=(
                replace(condition, scope_disposition="excluded_from_paper_matrix"),
            ),
        )
    raise AssertionError(change)


@pytest.mark.parametrize(
    "change",
    [
        "mapping",
        "dialect",
        "numeric_policy",
        "option_policy",
        "extra_field_policy",
        "status",
        "scope",
    ],
)
def test_stable_mapping_fields_change_import_spec_digest(minimal_spec, change):
    changed = mutate_identity_field(minimal_spec, change)
    assert import_spec_digest(changed) != import_spec_digest(minimal_spec)


def test_stable_projection_is_json_native_and_has_no_audit_locations(
    minimal_overlay_spec,
):
    projection = stable_import_projection(minimal_overlay_spec)
    assert "audit" not in projection
    assert "path" not in projection["sources"][0]
    assert "base_run_path" not in projection["overlays"][0]
