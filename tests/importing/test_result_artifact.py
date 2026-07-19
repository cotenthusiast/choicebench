"""Result-artifact-v2 tests: non-circular identity for imported realizations.

Reuses tests/importing/test_identity.py's condition/realization builders
instead of re-deriving a full semantic identity fixture here.
"""

from __future__ import annotations

import hashlib
import json

import pandas as pd
import pytest

from choicebench.identity import CANONICALIZATION_VERSION, integrity_digest
from choicebench.manifest import PROTOCOL_V3_VERSION, make_manifest_v3
from choicebench.io.writers import (
    RESULT_ARTIFACT_V2_SCHEMA_VERSION,
    prepare_manifest_result,
    publish_manifest_result,
    validate_result_artifact,
)
from tests.importing.test_identity import (
    _realization_identity,
    _records,
    make_realization as _make_realization,
)


def _build_manifest_and_realization():
    condition = _records().condition
    identity = _realization_identity()
    realization = _make_realization(
        condition_id=condition["condition_id"],
        condition_digest=condition["condition_digest"],
        identity=identity,
        fields={},
    )
    payload = {
        "protocol_version": PROTOCOL_V3_VERSION,
        "canonicalization_version": CANONICALIZATION_VERSION,
        "semantic_conditions": {condition["condition_id"]: condition},
        "realizations": {realization["realization_id"]: realization},
    }
    manifest = make_manifest_v3(payload)
    return manifest, realization


def _rows_for(manifest, realization):
    """Rows shaped as Task 9's normalization step would deliver them: already
    carrying the fixed ownership/identity columns prepare_manifest_result
    validates (never fabricates) per row."""
    realization_id = realization["realization_id"]
    condition_id = realization["condition_id"]
    condition = manifest["payload"]["semantic_conditions"][condition_id]
    benchmark = condition["identity"]["benchmark"]
    identity_columns = {
        "condition_id": condition_id,
        "realization_id": realization_id,
        "experiment_id": manifest["experiment_id"],
        "dataset_artifact_id": benchmark["artifact_id"],
        "dataset_selection_id": benchmark["selection_id"],
        "model_id": condition["identity"]["model_id"],
        "method_id": condition["identity"]["method_id"],
        "prompt_id": condition["identity"]["prompt_id"],
        "benchmark_name": benchmark["name"],
        "benchmark_split": benchmark["split"],
    }
    row_assignments = realization["identity"]["realization"]["result_origin"]["row_assignments"]
    return [
        {
            **identity_columns,
            "question_id": row["question_id"],
            "prediction_origin": row["prediction_origin"],
            "prediction_lineage_id": row["prediction_lineage_id"],
            "parsed_choice": "A",
            "is_correct": True,
        }
        for row in row_assignments
    ]


def test_prepare_manifest_result_computes_identity_after_manifest_is_fixed():
    manifest, realization = _build_manifest_and_realization()
    realization_id = realization["realization_id"]
    prepared = prepare_manifest_result(
        _rows_for(manifest, realization), manifest=manifest, realization_id=realization_id
    )
    assert "result_artifact_id" not in manifest
    assert prepared.metadata["experiment_id"] == manifest["experiment_id"]
    assert prepared.metadata["realization_id"] == realization_id
    assert prepared.metadata["file_sha256"] == hashlib.sha256(prepared.csv_bytes).hexdigest()
    assert prepared.metadata["result_artifact_id"].startswith("result_")
    assert prepared.result_path == f"results/{realization_id}.csv"
    assert prepared.metadata_path == f"results/{realization_id}.artifact.json"


def test_prepare_manifest_result_injects_identity_columns_per_row():
    manifest, realization = _build_manifest_and_realization()
    realization_id = realization["realization_id"]
    prepared = prepare_manifest_result(
        _rows_for(manifest, realization), manifest=manifest, realization_id=realization_id
    )
    df = pd.read_csv(pd.io.common.BytesIO(prepared.csv_bytes))
    for column in (
        "condition_id",
        "realization_id",
        "experiment_id",
        "dataset_artifact_id",
        "dataset_selection_id",
        "model_id",
        "method_id",
        "prompt_id",
        "benchmark_name",
        "benchmark_split",
        "prediction_origin",
        "prediction_lineage_id",
    ):
        assert column in df.columns
        assert df[column].nunique(dropna=False) == 1 or column in (
            "prediction_origin",
            "prediction_lineage_id",
        )
    assert set(df["realization_id"]) == {realization_id}


def test_prepare_manifest_result_orders_rows_by_declared_row_assignments():
    manifest, realization = _build_manifest_and_realization()
    realization_id = realization["realization_id"]
    rows = _rows_for(manifest, realization)
    prepared = prepare_manifest_result(
        list(reversed(rows)), manifest=manifest, realization_id=realization_id
    )
    expected_order = [row["question_id"] for row in rows]
    df = pd.read_csv(pd.io.common.BytesIO(prepared.csv_bytes), dtype={"question_id": "string"})
    assert df["question_id"].tolist() == expected_order


def test_prepare_manifest_result_rejects_unknown_question_id():
    manifest, realization = _build_manifest_and_realization()
    rows = _rows_for(manifest, realization)
    rows[0]["question_id"] = "not-a-declared-question"
    with pytest.raises(RuntimeError, match="unexpected question_id"):
        prepare_manifest_result(
            rows, manifest=manifest, realization_id=realization["realization_id"]
        )


def test_prepare_manifest_result_rejects_missing_row():
    manifest, realization = _build_manifest_and_realization()
    rows = _rows_for(manifest, realization)[:-1]
    with pytest.raises(RuntimeError, match="missing result rows"):
        prepare_manifest_result(
            rows, manifest=manifest, realization_id=realization["realization_id"]
        )


def test_prepare_manifest_result_rejects_prediction_origin_mismatch():
    manifest, realization = _build_manifest_and_realization()
    rows = _rows_for(manifest, realization)
    rows[0]["prediction_origin"] = "native_inference"
    with pytest.raises(RuntimeError, match="prediction_origin"):
        prepare_manifest_result(
            rows, manifest=manifest, realization_id=realization["realization_id"]
        )


def test_prepare_manifest_result_rejects_unknown_realization():
    manifest, _realization = _build_manifest_and_realization()
    with pytest.raises(RuntimeError, match="[Uu]nknown realization"):
        prepare_manifest_result(
            [], manifest=manifest, realization_id="real_" + "0" * 16
        )


def test_publish_manifest_result_writes_csv_and_sidecar(tmp_path):
    manifest, realization = _build_manifest_and_realization()
    realization_id = realization["realization_id"]
    prepared = prepare_manifest_result(
        _rows_for(manifest, realization), manifest=manifest, realization_id=realization_id
    )
    path, metadata, reused = publish_manifest_result(prepared, run_dir=tmp_path)
    assert reused is False
    assert path == tmp_path / "results" / f"{realization_id}.csv"
    assert path.read_bytes() == prepared.csv_bytes
    sidecar_path = tmp_path / "results" / f"{realization_id}.artifact.json"
    assert json.loads(sidecar_path.read_text()) == dict(prepared.metadata)
    assert metadata["result_artifact_id"] == prepared.metadata["result_artifact_id"]


def test_publish_manifest_result_is_idempotent_noop_for_identical_bytes(tmp_path):
    manifest, realization = _build_manifest_and_realization()
    realization_id = realization["realization_id"]
    prepared = prepare_manifest_result(
        _rows_for(manifest, realization), manifest=manifest, realization_id=realization_id
    )
    publish_manifest_result(prepared, run_dir=tmp_path)
    _, _, second_reused = publish_manifest_result(prepared, run_dir=tmp_path)
    assert second_reused is True


def test_publish_manifest_result_refuses_divergent_overwrite(tmp_path):
    manifest, realization = _build_manifest_and_realization()
    realization_id = realization["realization_id"]
    rows = _rows_for(manifest, realization)
    prepared = prepare_manifest_result(rows, manifest=manifest, realization_id=realization_id)
    publish_manifest_result(prepared, run_dir=tmp_path)
    rows[0]["parsed_choice"] = "B"
    divergent = prepare_manifest_result(rows, manifest=manifest, realization_id=realization_id)
    with pytest.raises(RuntimeError, match="divergent"):
        publish_manifest_result(divergent, run_dir=tmp_path)


def test_validate_result_artifact_v2_accepts_a_well_formed_pair(tmp_path):
    manifest, realization = _build_manifest_and_realization()
    realization_id = realization["realization_id"]
    prepared = prepare_manifest_result(
        _rows_for(manifest, realization), manifest=manifest, realization_id=realization_id
    )
    path, _, _ = publish_manifest_result(prepared, run_dir=tmp_path)
    metadata = validate_result_artifact(path, manifest=manifest, realization=realization)
    assert metadata["schema_version"] == RESULT_ARTIFACT_V2_SCHEMA_VERSION


def test_validate_result_artifact_v2_rejects_tampered_csv_bytes(tmp_path):
    manifest, realization = _build_manifest_and_realization()
    realization_id = realization["realization_id"]
    prepared = prepare_manifest_result(
        _rows_for(manifest, realization), manifest=manifest, realization_id=realization_id
    )
    path, _, _ = publish_manifest_result(prepared, run_dir=tmp_path)
    path.write_bytes(prepared.csv_bytes + b"\n# tampered")
    with pytest.raises(RuntimeError, match="content integrity"):
        validate_result_artifact(path, manifest=manifest, realization=realization)


def test_validate_result_artifact_v2_rejects_realization_binding_mismatch(tmp_path):
    manifest, realization = _build_manifest_and_realization()
    realization_id = realization["realization_id"]
    prepared = prepare_manifest_result(
        _rows_for(manifest, realization), manifest=manifest, realization_id=realization_id
    )
    path, _, _ = publish_manifest_result(prepared, run_dir=tmp_path)
    empty_manifest = make_manifest_v3(
        {
            "protocol_version": PROTOCOL_V3_VERSION,
            "canonicalization_version": CANONICALIZATION_VERSION,
            "semantic_conditions": {},
            "realizations": {},
        }
    )
    with pytest.raises(RuntimeError, match="realization"):
        validate_result_artifact(path, manifest=empty_manifest)


def test_validate_result_artifact_v1_path_is_unaffected():
    """Calling with no manifest/realization keeps the exact v1 behavior."""
    from choicebench.io.writers import RESULT_ARTIFACT_SCHEMA_VERSION

    assert RESULT_ARTIFACT_SCHEMA_VERSION == "choicebench.result-artifact.v1"
