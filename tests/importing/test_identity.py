from __future__ import annotations

from dataclasses import asdict, replace
from hashlib import sha256
import importlib
from pathlib import Path

import pandas as pd
import pytest

from choicebench.config.schema import (
    BenchmarkConfig,
    ExperimentConfig,
    MethodConfig,
    ModelConfig,
    RunConfig,
)
from choicebench.datasets import dataset_content_digest
from choicebench.identity import integrity_digest, short_id
from choicebench.importing.csv_adapter import OpenedSource
from choicebench.importing.dataset_reference import build_expected_dataset
from choicebench.importing.identity import (
    ImportIdentityError,
    build_import_semantic_identity,
    importer_implementation_identity,
    make_lineage_component,
    make_realization,
    make_result_origin,
    make_semantic_condition,
)
from choicebench.importing.schema import (
    DatasetReferenceSpec,
    CsvDialectSpec,
    ImportConditionSpec,
    ImportMethodSpec,
    ImportModelSpec,
    ImportPromptSpec,
    ResultOriginSpec,
)


def _adapter(value: str) -> str:
    return value


def _validator(value: str) -> bool:
    return bool(value)


def _dataset():
    frame = pd.DataFrame(
        [
            {"qid": "q1", "stem": "One?", "gold": "A", "a": "x", "b": "y"},
            {"qid": "q2", "stem": "Two?", "gold": "B", "a": "m", "b": "n"},
            {"qid": "q3", "stem": "Three?", "gold": "A", "a": "i", "b": "j"},
        ]
    )
    data = frame.to_csv(index=False, lineterminator="\n").encode()
    source = OpenedSource(
        source_id="benchmark",
        audit_path=Path("/machine-a/input.csv"),
        logical_path="inputs/benchmark.csv",
        data=data,
        sha256=sha256(data).hexdigest(),
    )
    declaration = DatasetReferenceSpec(
        dataset_id="dataset",
        benchmark_name="benchmark",
        split="test",
        reference_kind="independent_input_snapshot",
        trust_label="declared-independent-input",
        source_ids=("benchmark",),
        selection_source_id="benchmark",
        expected_question_ids=("q2", "q1"),
        selection_seed=None,
        selection_n_samples=2,
        subject_filter=(),
        selection_unknown_reasons={"selection_seed": "not recorded"},
        columns={
            "question_id": "qid",
            "question_text": "stem",
            "correct_option": "gold",
            "choice_a": "a",
            "choice_b": "b",
        },
        revision=None,
        fingerprint=None,
        derivation={"source_role": "benchmark_input"},
        limitations=("publisher revision unknown",),
        native_compatibility_identity=None,
    )
    return build_expected_dataset(declaration, {"benchmark": source})


def _model() -> ImportModelSpec:
    return ImportModelSpec(
        model_key="model",
        display_name="historical-model",
        backend=None,
        provider=None,
        revision=None,
        effective_parameters={"temperature": 0},
        unknown_reasons={
            "backend": "not recorded",
            "provider": "not recorded",
            "revision": "not recorded",
        },
        native_compatibility_identity=None,
    )


def _method() -> ImportMethodSpec:
    return ImportMethodSpec(
        method_key="method",
        name="semantic_matching_v1",
        effective_parameters={"matching": "semantic"},
        implementation=None,
        unknown_reasons={"implementation": "historical code unavailable"},
        native_compatibility_identity=None,
    )


def _prompt() -> ImportPromptSpec:
    return ImportPromptSpec(
        prompt_key="prompt",
        template_identity=None,
        template_digest=None,
        template_contents=None,
        unknown_reason="raw prompt was not preserved",
        native_compatibility_identity=None,
    )


def _condition() -> ImportConditionSpec:
    return ImportConditionSpec(
        condition_key="condition",
        source_ids=("results",),
        dataset_id="dataset",
        model_key="model",
        method_key="method",
        prompt_key="prompt",
        seed=None,
        calibration_identity=None,
        preflight_identity=None,
        protocol_settings={},
        generation_parameters={"max_tokens": 64},
        unknown_reasons={
            "seed": "not recorded",
            "calibration_identity": "not applicable",
            "preflight_identity": "not recorded",
        },
        expected_question_ids=("q2", "q1"),
        evidence_status="complete",
        scope_disposition="included",
        executable=None,
        qualifications=(),
        limitations=(),
        damaged_question_ids=(),
        recoverable_question_ids=(),
        result_origin=ResultOriginSpec(
            derivation_origin="external_import",
            default_prediction_origin="external_historical_inference",
            per_question_prediction_origins={},
        ),
    )


def _records(**condition_changes):
    condition = replace(_condition(), **condition_changes)
    return build_import_semantic_identity(
        condition=condition,
        dataset=_dataset(),
        model=_model(),
        method=_method(),
        prompt=_prompt(),
    )


def test_fallback_child_payloads_are_exact_and_unknowns_are_explicit():
    records = _records()

    assert records.dataset_artifact["identity"] == {
        "benchmark": "benchmark",
        "content_digest": records.dataset_artifact["identity"]["content_digest"],
        "schema_version": "choicebench.semantic-dataset.v1",
        "split": "test",
    }
    assert records.model["identity"] == {
        "backend": {"reason": "not recorded", "value": None},
        "effective_parameters": {"max_tokens": 64, "temperature": 0},
        "model": "historical-model",
        "provider": {"reason": "not recorded", "value": None},
        "revision": {"reason": "not recorded", "value": None},
        "schema_version": "choicebench.semantic-model.v1",
    }
    assert records.method["identity"] == {
        "effective_params": {"matching": "semantic"},
        "implementation": {
            "reason": "historical code unavailable",
            "value": None,
        },
        "name": "semantic_matching_v1",
        "preflight": {"reason": "not recorded", "value": None},
        "schema_version": "choicebench.semantic-method.v1",
    }
    unknown_prompt = {"reason": "raw prompt was not preserved", "value": None}
    assert records.prompt["identity"] == {
        "schema_version": "choicebench.semantic-prompt.v1",
        "template_contents": unknown_prompt,
        "template_digest": unknown_prompt,
        "template_identity": unknown_prompt,
    }


def test_child_records_have_full_digests_and_existing_short_prefixes():
    records = _records()
    for record, prefix, id_key, digest_key in (
        (records.dataset_artifact, "ds", "artifact_id", "artifact_digest"),
        (records.selection, "sel", "selection_id", "selection_digest"),
        (records.model, "model", "model_id", "model_digest"),
        (records.method, "method", "method_id", "method_digest"),
        (records.prompt, "prompt", "prompt_id", "prompt_digest"),
    ):
        assert record[digest_key] == integrity_digest(record["identity"])
        assert record[id_key] == short_id(prefix, record["identity"])


def test_condition_payload_matches_current_choicebench_keys_and_frozen_prompt_path():
    records = _records()
    identity = records.condition["identity"]

    assert set(identity) == {
        "benchmark",
        "preflight",
        "model_id",
        "method_id",
        "prompt_id",
        "prompt_snapshot_path",
        "seed",
    }
    assert identity["benchmark"] == {
        "name": "benchmark",
        "split": "test",
        "artifact_id": records.dataset_artifact["artifact_id"],
        "selection_id": records.selection["selection_id"],
    }
    assert identity["prompt_snapshot_path"] == (
        f"artifacts/prompts/{records.prompt['prompt_id']}"
    )
    assert records.condition["condition_digest"] == integrity_digest(identity)
    assert records.condition["condition_id"] == short_id("cond", identity)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("seed", 7),
        ("protocol_settings", {"threshold": 3}),
        ("generation_parameters", {"max_tokens": 65}),
    ],
)
def test_scientific_condition_changes_change_condition_identity(field, value):
    assert _records(**{field: value}).condition["condition_id"] != _records().condition[
        "condition_id"
    ]


@pytest.mark.parametrize(
    "field",
    ["selection_id", "artifact_id", "model_id", "method_id", "prompt_id"],
)
def test_semantic_child_change_changes_condition_identity(field):
    original = _records().condition
    identity = dict(original["identity"])
    if field in {"selection_id", "artifact_id"}:
        identity["benchmark"] = dict(identity["benchmark"])
        identity["benchmark"][field] = f"{field}_changed"
    else:
        identity[field] = f"{field}_changed"
        if field == "prompt_id":
            identity["prompt_snapshot_path"] = (
                f"artifacts/prompts/{identity['prompt_id']}"
            )
    changed = make_semantic_condition(identity=identity, fields={})
    assert changed["condition_id"] != original["condition_id"]


def test_make_semantic_condition_rejects_arbitrary_prompt_path_injection():
    identity = dict(_records().condition["identity"])
    identity["prompt_snapshot_path"] = "/tmp/attacker/prompt"

    with pytest.raises(ImportIdentityError, match="prompt.*path"):
        make_semantic_condition(identity=identity, fields={"condition_key": "x"})


def test_native_compatibility_children_are_preserved_without_fabrication():
    fallback = _records()
    model_payload = {"backend": "dummy", "model_name_or_path": "native"}
    method_payload = {
        "name": "direct",
        "effective_params": {},
        "preflight": None,
        "implementation": {"qualified_name": "choicebench.methods:Direct"},
    }
    native_model = {
        "payload": model_payload,
        "digest": integrity_digest(model_payload),
        "model_id": short_id("model", model_payload),
    }
    native_method = {
        "payload": method_payload,
        "digest": integrity_digest(method_payload),
        "method_id": short_id("method", method_payload),
    }
    prompt_payload = {
        "version": "v1",
        "files": {
            name: {"sha256": integrity_digest(name), "content": name}
            for name in ("direct_mcq", "free_text", "option_matching")
        },
    }
    native_prompt = {
        "prompt_id": short_id("prompt", prompt_payload),
        **prompt_payload,
    }
    records = build_import_semantic_identity(
        condition=replace(
            _condition(),
            generation_parameters={},
            unknown_reasons={
                **_condition().unknown_reasons,
                "preflight_identity": "not applicable",
            },
        ),
        dataset=_dataset(),
        model=replace(
            _model(),
            display_name="native",
            backend="dummy",
            effective_parameters={},
            unknown_reasons={"provider": "not applicable", "revision": "not applicable"},
            native_compatibility_identity=native_model,
        ),
        method=replace(
            _method(),
            name="direct",
            effective_parameters={},
            implementation=method_payload["implementation"],
            unknown_reasons={},
            native_compatibility_identity=native_method,
        ),
        prompt=replace(
            _prompt(),
            template_identity="v1",
            template_digest=integrity_digest(prompt_payload["files"]),
            template_contents={
                name: item["content"] for name, item in prompt_payload["files"].items()
            },
            unknown_reason=None,
            native_compatibility_identity=native_prompt,
        ),
    )

    assert records.model["identity"] == model_payload
    assert records.model["model_id"] == native_model["model_id"]
    assert records.method["identity"] == method_payload
    assert records.prompt["identity"] == prompt_payload
    assert records.condition["condition_id"] != fallback.condition["condition_id"]


def test_native_model_identity_refuses_unbound_generation_parameters():
    payload = {"backend": "dummy", "model_name_or_path": "historical-model"}
    native = {
        "payload": payload,
        "digest": integrity_digest(payload),
        "model_id": short_id("model", payload),
    }
    with pytest.raises(ImportIdentityError, match="generation"):
        build_import_semantic_identity(
            condition=_condition(),
            dataset=_dataset(),
            model=replace(_model(), native_compatibility_identity=native),
            method=_method(),
            prompt=_prompt(),
        )


def test_native_child_claims_must_match_their_semantic_declarations():
    model_payload = {"backend": "dummy", "model_name_or_path": "historical-model"}
    native_model = {
        "payload": model_payload,
        "digest": integrity_digest(model_payload),
        "model_id": short_id("model", model_payload),
    }
    with pytest.raises(ImportIdentityError, match="backend"):
        build_import_semantic_identity(
            condition=replace(_condition(), generation_parameters={}),
            dataset=_dataset(),
            model=replace(
                _model(), backend="api", native_compatibility_identity=native_model
            ),
            method=_method(),
            prompt=_prompt(),
        )

    method_payload = {
        "name": "direct_mcq",
        "effective_params": {},
        "preflight": None,
        "implementation": {"qualified_name": "choicebench.methods:Direct"},
    }
    native_method = {
        "payload": method_payload,
        "digest": integrity_digest(method_payload),
        "method_id": short_id("method", method_payload),
    }
    with pytest.raises(ImportIdentityError, match="method.*declaration"):
        build_import_semantic_identity(
            condition=replace(
                _condition(),
                preflight_identity=None,
                unknown_reasons={
                    **_condition().unknown_reasons,
                    "preflight_identity": "not applicable",
                },
            ),
            dataset=_dataset(),
            model=_model(),
            method=replace(_method(), native_compatibility_identity=native_method),
            prompt=_prompt(),
        )


def test_native_prompt_contents_must_match_the_semantic_declaration():
    payload = {
        "version": "v1",
        "files": {
            name: {"sha256": integrity_digest(name), "content": name}
            for name in ("direct_mcq", "free_text", "option_matching")
        },
    }
    native = {"prompt_id": short_id("prompt", payload), **payload}
    with pytest.raises(ImportIdentityError, match="prompt.*contents"):
        build_import_semantic_identity(
            condition=_condition(),
            dataset=_dataset(),
            model=_model(),
            method=_method(),
            prompt=replace(
                _prompt(),
                template_identity="v1",
                template_digest=integrity_digest(payload["files"]),
                template_contents={"direct_mcq": "different"},
                unknown_reason=None,
                native_compatibility_identity=native,
            ),
        )


def test_fully_validated_native_children_reproduce_build_execution_plan_condition():
    import choicebench.cli.run_experiment as run_experiment

    run_experiment = importlib.reload(run_experiment)
    config = ExperimentConfig(
        "native-compat",
        [ModelConfig("dummy", "same")],
        [BenchmarkConfig("toy", n_samples=2)],
        [MethodConfig("direct_mcq")],
        ["accuracy"],
        RunConfig(seed=42),
    )
    plan = run_experiment.build_execution_plan(config)
    native_condition = next(iter(plan.conditions.values()))
    native_selection = plan.selections[0]
    artifact = native_selection.artifact
    artifact_payload = {
        "spec": artifact.metadata["spec"],
        "content_digest": artifact.content_digest,
        "source": artifact.metadata["source"],
    }
    selection_payload = {
        "artifact_id": artifact.artifact_id,
        "content_digest": dataset_content_digest(native_selection.questions),
        "sample_identities": list(native_selection.sample_identities),
        "seed": 42,
        "n_samples": 2,
        "subject_filter": [],
    }
    native_dataset = {
        "artifact_payload": artifact_payload,
        "artifact_digest": integrity_digest(artifact_payload),
        "artifact_id": artifact.artifact_id,
        "selection_payload": selection_payload,
        "selection_digest": integrity_digest(selection_payload),
        "selection_id": native_selection.selection_id,
    }
    source_bytes = artifact.dataframe.to_csv(index=False, lineterminator="\n").encode()
    source = OpenedSource(
        source_id="native-input",
        audit_path=artifact.path,
        logical_path="prepared/toy.csv",
        data=source_bytes,
        sha256=sha256(source_bytes).hexdigest(),
    )
    dataset = build_expected_dataset(
        DatasetReferenceSpec(
            dataset_id="toy",
            benchmark_name="toy",
            split="test",
            reference_kind="independent_input_snapshot",
            trust_label="choicebench-verified",
            source_ids=("native-input",),
            selection_source_id="native-input",
            expected_question_ids=tuple(
                native_selection.questions["question_id"].astype(str)
            ),
            selection_seed=42,
            selection_n_samples=2,
            subject_filter=(),
            selection_unknown_reasons={},
            columns={column: column for column in artifact.dataframe.columns},
            revision=None,
            fingerprint=None,
            derivation={"kind": "native compatibility fixture"},
            limitations=(),
            native_compatibility_identity=native_dataset,
        ),
        {"native-input": source},
    )
    native_model_record = plan.manifest["payload"]["models"][0]
    native_model_payload = {
        "backend": "dummy",
        "model_name_or_path": "same",
    }
    native_method_record = plan.manifest["payload"]["methods"][0]
    native_method_payload = {
        "name": native_method_record["config"]["name"],
        "effective_params": native_method_record["config"]["params"],
        "preflight": native_method_record["config"]["preflight"],
        "implementation": native_method_record["implementation"],
    }
    native_prompt_record = plan.manifest["payload"]["prompts"]
    condition = replace(
        _condition(),
        dataset_id="toy",
        seed=42,
        generation_parameters={},
        unknown_reasons={
            "calibration_identity": "not applicable",
            "preflight_identity": "not applicable",
        },
        expected_question_ids=tuple(
            native_selection.questions["question_id"].astype(str)
        ),
    )
    records = build_import_semantic_identity(
        condition=condition,
        dataset=dataset,
        model=replace(
            _model(),
            display_name="same",
            backend="dummy",
            provider=None,
            revision=None,
            unknown_reasons={"provider": "not applicable", "revision": "not applicable"},
            effective_parameters={},
            native_compatibility_identity={
                "payload": native_model_payload,
                "digest": integrity_digest(native_model_payload),
                "model_id": native_model_record["model_id"],
            },
        ),
        method=replace(
            _method(),
            name="direct_mcq",
            effective_parameters=native_method_record["config"]["params"],
            implementation=native_method_record["implementation"],
            unknown_reasons={},
            native_compatibility_identity={
                "payload": native_method_payload,
                "digest": integrity_digest(native_method_payload),
                "method_id": native_method_record["method_id"],
            },
        ),
        prompt=replace(
            _prompt(),
            template_identity=native_prompt_record["version"],
            template_digest=integrity_digest(native_prompt_record["files"]),
            template_contents={
                name: item["content"]
                for name, item in native_prompt_record["files"].items()
            },
            unknown_reason=None,
            native_compatibility_identity={
                "prompt_id": native_prompt_record["prompt_id"],
                "version": native_prompt_record["version"],
                "files": native_prompt_record["files"],
            },
        ),
    )

    assert records.condition["identity"] == native_condition["identity"]
    assert records.condition["condition_id"] == native_condition["condition_id"]


def _realization_identity() -> dict:
    origin = make_result_origin(
        derivation_origin="external_import",
        row_assignments=(
            ("q2", "external_historical_inference", "lin_a"),
            ("q1", "external_historical_inference", "lin_b"),
        ),
    )
    return {
        "import_spec_digest": "1" * 64,
        "source_sha256": "2" * 64,
        "mapping": {"question_id": "qid"},
        "dialect": {"delimiter": ","},
        "evidence_status": "complete",
        "scope": "included",
        "authorization_digest": None,
        "overlay_sha256": None,
        "prediction_origins": origin,
    }


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("source_sha256", "3" * 64),
        ("mapping", {"question_id": "id"}),
        ("dialect", {"delimiter": ";"}),
        ("evidence_status", "qualified"),
        ("scope", "superseded"),
        ("authorization_digest", "4" * 64),
        ("overlay_sha256", "5" * 64),
        ("prediction_origins", {"ordered_row_origin_digest": "6" * 64}),
    ],
)
def test_realization_changes_leave_condition_fixed(field, value):
    condition = _records().condition
    original = make_realization(
        condition_id=condition["condition_id"],
        condition_digest=condition["condition_digest"],
        identity=_realization_identity(),
        fields={"audit": {"source_path": "/machine-a/results.csv"}},
    )
    changed_identity = _realization_identity()
    changed_identity[field] = value
    changed = make_realization(
        condition_id=condition["condition_id"],
        condition_digest=condition["condition_digest"],
        identity=changed_identity,
        fields={"audit": {"source_path": "/machine-b/results.csv"}},
    )

    assert changed["condition_id"] == original["condition_id"]
    assert changed["realization_id"] != original["realization_id"]


def test_audit_only_realization_fields_do_not_change_identity():
    condition = _records().condition
    first = make_realization(
        condition_id=condition["condition_id"],
        condition_digest=condition["condition_digest"],
        identity=_realization_identity(),
        fields={"audit": {"source_path": "/machine-a/a.csv", "timestamp": "now"}},
    )
    moved = make_realization(
        condition_id=condition["condition_id"],
        condition_digest=condition["condition_digest"],
        identity=_realization_identity(),
        fields={"audit": {"source_path": "/machine-b/a.csv", "timestamp": "later"}},
    )
    assert first["realization_id"] == moved["realization_id"]
    assert first["realization_digest"] == moved["realization_digest"]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("encoding", "utf-8-sig"),
        ("bom_policy", "strip_utf8_bom"),
        ("decoding_errors", "replace"),
        ("delimiter", ";"),
        ("quote_character", "'"),
        ("escape_character", "\\"),
        ("double_quote", False),
        ("line_terminators", ["lf"]),
        ("mixed_line_terminators", "forbid"),
        ("final_record_without_terminator", "forbid"),
        ("blank_record_policy", "allow"),
        ("skip_initial_space", True),
        ("header", "explicit"),
        ("strict_syntax", False),
    ],
)
def test_every_csv_decoding_and_dialect_field_changes_realization_not_condition(
    field, value
):
    condition = _records().condition
    base_dialect = asdict(CsvDialectSpec())
    first_identity = _realization_identity()
    first_identity["dialect"] = base_dialect
    changed_identity = _realization_identity()
    changed_identity["dialect"] = {**base_dialect, field: value}
    first = make_realization(
        condition_id=condition["condition_id"],
        condition_digest=condition["condition_digest"],
        identity=first_identity,
        fields={},
    )
    changed = make_realization(
        condition_id=condition["condition_id"],
        condition_digest=condition["condition_digest"],
        identity=changed_identity,
        fields={},
    )
    assert first["condition_id"] == changed["condition_id"]
    assert first["realization_id"] != changed["realization_id"]


def test_base_repair_and_transformation_are_distinct_realizations_of_one_condition():
    condition = _records().condition
    identities = []
    for derivation in ("external_import", "repair_overlay", "offline_transformation"):
        identity = _realization_identity()
        identity["derivation_origin"] = derivation
        identities.append(
            make_realization(
                condition_id=condition["condition_id"],
                condition_digest=condition["condition_digest"],
                identity=identity,
                fields={},
            )
        )
    assert {item["condition_id"] for item in identities} == {
        condition["condition_id"]
    }
    assert len({item["realization_id"] for item in identities}) == 3


def test_realization_refuses_post_publication_or_machine_local_identity_fields():
    condition = _records().condition
    for forbidden in (
        "result_artifact_id",
        "experiment_id",
        "output_path",
        "source_path",
        "timestamp",
    ):
        identity = _realization_identity()
        identity[forbidden] = "forbidden"
        with pytest.raises(ImportIdentityError, match=forbidden):
            make_realization(
                condition_id=condition["condition_id"],
                condition_digest=condition["condition_digest"],
                identity=identity,
                fields={},
            )


def test_lineage_component_is_parent_derived_and_has_no_child_edge():
    component = make_lineage_component(
        operation_type="offline_transformation",
        question_id="q1",
        parent_digests=("2" * 64, "1" * 64),
        source_digests=("4" * 64, "3" * 64),
        authorization_digest="5" * 64,
        implementation={"qualified_name": "choicebench.transforms:rematch"},
        parameters={"matcher": "v1"},
        input_digest="6" * 64,
        preownership_output_digest="7" * 64,
        prediction_origin="external_historical_inference",
    )

    assert component["lineage_digest"] == integrity_digest(component["identity"])
    assert component["lineage_id"] == short_id("lin", component["identity"])
    assert component["identity"]["parent_digests"] == ["1" * 64, "2" * 64]
    assert not ({"realization_id", "result_artifact_id", "experiment_id"} & set(component["identity"]))


def test_lineage_refuses_child_paths_timestamps_and_final_csv_hashes():
    for forbidden in ("realization_id", "output_path", "timestamp", "final_csv_sha256"):
        with pytest.raises(ImportIdentityError, match=forbidden):
            make_lineage_component(
                operation_type="offline_transformation",
                question_id="q1",
                parent_digests=(),
                source_digests=("1" * 64,),
                authorization_digest=None,
                implementation={"qualified_name": "x:y"},
                parameters={forbidden: "bad"},
                input_digest="2" * 64,
                preownership_output_digest="3" * 64,
                prediction_origin="external_historical_inference",
            )


def test_result_origin_records_constituents_counts_and_ordered_per_row_mapping():
    origin = make_result_origin(
        derivation_origin="repair_overlay",
        row_assignments=(
            ("q1", "external_historical_inference", "lin_base"),
            ("q2", "native_inference", "lin_repair"),
            ("q3", "native_inference", "lin_repair_2"),
        ),
    )

    assert origin["prediction_origins"] == [
        "external_historical_inference",
        "native_inference",
    ]
    assert origin["prediction_origin_counts"] == {
        "external_historical_inference": 1,
        "native_inference": 2,
    }
    assert [item["question_id"] for item in origin["row_assignments"]] == [
        "q1",
        "q2",
        "q3",
    ]
    assert origin["ordered_row_origin_digest"] == integrity_digest(
        origin["row_assignments"]
    )


def test_result_origin_rejects_bare_mixed_duplicate_rows_and_invalid_origins():
    for assignments in (
        (("q1", "mixed", "lin_a"),),
        (
            ("q1", "external_historical_inference", "lin_a"),
            ("q1", "native_inference", "lin_b"),
        ),
    ):
        with pytest.raises(ImportIdentityError):
            make_result_origin(
                derivation_origin="repair_overlay", row_assignments=assignments
            )


def test_offline_transformation_keeps_underlying_prediction_origin():
    origin = make_result_origin(
        derivation_origin="offline_transformation",
        row_assignments=(
            ("q1", "external_historical_inference", "lin_rematch"),
        ),
    )
    assert origin["derivation_origin"] == "offline_transformation"
    assert origin["prediction_origins"] == ["external_historical_inference"]


def test_importer_implementation_identity_binds_package_adapter_and_validator():
    identity = importer_implementation_identity(adapter=_adapter, validator=_validator)
    assert identity["schema_version"] == "choicebench.importer-implementation.v1"
    assert identity["package"]["name"] == "choicebench"
    assert identity["package"]["version"]
    assert identity["adapter"]["qualified_name"].endswith(":_adapter")
    assert identity["adapter"]["source_digest"]
    assert identity["validator"]["qualified_name"].endswith(":_validator")
    changed = importer_implementation_identity(adapter=_validator, validator=_adapter)
    assert integrity_digest(identity) != integrity_digest(changed)
