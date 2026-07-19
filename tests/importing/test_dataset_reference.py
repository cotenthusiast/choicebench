from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path

import pandas as pd
import pytest

from choicebench.datasets import (
    NORMALIZATION_VERSION,
    dataset_content_digest,
    dataset_sample_identities,
)
from choicebench.identity import integrity_digest, short_id
from choicebench.importing.csv_adapter import OpenedSource
from choicebench.importing.dataset_reference import (
    DatasetReferenceError,
    build_expected_dataset,
    validate_expected_snapshot,
    write_expected_snapshot,
)
from choicebench.importing.schema import DatasetReferenceSpec
from choicebench.pipeline.options import build_option_map


def _csv_bytes(rows: list[dict[str, str]]) -> bytes:
    frame = pd.DataFrame(rows)
    return frame.to_csv(index=False, lineterminator="\n").encode("utf-8")


def _source(
    source_id: str,
    rows: list[dict[str, str]],
    *,
    logical_path: str | None = None,
    audit_path: Path | None = None,
) -> OpenedSource:
    data = _csv_bytes(rows)
    return OpenedSource(
        source_id=source_id,
        audit_path=audit_path or Path(f"/machine-a/{source_id}.csv"),
        logical_path=logical_path or f"publisher/{source_id}.csv",
        data=data,
        sha256=sha256(data).hexdigest(),
    )


def _rows(*, six_options: bool = False) -> list[dict[str, str]]:
    option_count = 6 if six_options else 3
    questions = (
        ("q1", "First?", "B", ("one", "two", "three", "four", "five", "six")),
        ("q2", "Second?", "A", ("red", "green", "blue", "cyan", "magenta", "yellow")),
        ("q3", "Third?", "C", ("cat", "dog", "owl", "fox", "yak", "eel")),
    )
    result: list[dict[str, str]] = []
    for question_id, question, gold, options in questions:
        row = {"qid": question_id, "stem": question, "gold": gold}
        row.update(
            {f"option_{index + 1}": option for index, option in enumerate(options[:option_count])}
        )
        result.append(row)
    return result


def _columns(option_count: int = 3) -> dict[str, str]:
    columns = {
        "question_id": "qid",
        "question_text": "stem",
        "correct_option": "gold",
    }
    columns.update(
        {f"choice_{chr(97 + index)}": f"option_{index + 1}" for index in range(option_count)}
    )
    return columns


def _declaration(
    *,
    source_ids: tuple[str, ...] = ("reference",),
    selection_source_id: str = "reference",
    expected_question_ids: tuple[str, ...] = ("q2", "q1"),
    reference_kind: str = "independent_input_snapshot",
    trust_label: str = "publisher-input",
    columns: dict[str, str] | None = None,
    revision: str | None = "publisher-r1",
    derivation: dict | None = None,
    limitations: tuple[str, ...] = (),
) -> DatasetReferenceSpec:
    return DatasetReferenceSpec(
        dataset_id="historical-test",
        benchmark_name="synthetic",
        split="test",
        reference_kind=reference_kind,  # type: ignore[arg-type]
        trust_label=trust_label,
        source_ids=source_ids,
        selection_source_id=selection_source_id,
        expected_question_ids=expected_question_ids,
        selection_seed=None,
        selection_n_samples=None,
        subject_filter=(),
        selection_unknown_reasons={
            "selection_seed": "not recorded by producer",
            "selection_n_samples": "not recorded by producer",
        },
        columns=columns or _columns(),
        revision=revision,
        fingerprint="publisher-fingerprint",
        derivation=derivation or {},
        limitations=limitations,
        native_compatibility_identity=None,
    )


def _derived_declaration(**changes: object) -> DatasetReferenceSpec:
    base = _declaration(
        source_ids=("results-a", "results-b"),
        selection_source_id="results-a",
        reference_kind="profile_derived_reference_snapshot",
        trust_label="cross-source-internal-consistency",
        derivation={
            "method": "exact selected-row agreement",
            "independent_source_groups": [["results-a"], ["results-b"]],
        },
    )
    return replace(base, **changes)


def _semantic_payloads(dataset) -> tuple[dict, dict]:
    artifact_payload = {
        "schema_version": "choicebench.semantic-dataset.v1",
        "benchmark": dataset.benchmark_name,
        "split": dataset.split,
        "content_digest": dataset_content_digest(dataset.artifact_frame),
    }
    selection_payload = {
        "artifact_id": short_id("ds", artifact_payload),
        "content_digest": dataset_content_digest(dataset.frame),
        "sample_identities": dataset_sample_identities(dataset.frame),
        "seed": None,
        "n_samples": None,
        "subject_filter": [],
    }
    return artifact_payload, selection_payload


def test_independent_reference_builds_full_semantic_identities_and_stable_order():
    dataset = build_expected_dataset(
        _declaration(), {"reference": _source("reference", _rows())}
    )

    artifact_payload, selection_payload = _semantic_payloads(dataset)
    assert dataset.selected_question_ids == ("q2", "q1")
    assert dataset.frame["question_id"].tolist() == ["q2", "q1"]
    assert dataset.artifact_digest == integrity_digest(artifact_payload)
    assert dataset.artifact_id == short_id("ds", artifact_payload)
    assert dataset.artifact_payload == artifact_payload
    assert dataset.selection_digest == integrity_digest(selection_payload)
    assert dataset.selection_id == short_id("sel", selection_payload)
    assert dataset.selection_payload == selection_payload
    assert dataset.identity_mode == "imported_semantic_fallback"
    assert dataset.selection_unknown_reasons == {
        "selection_seed": "not recorded by producer",
        "selection_n_samples": "not recorded by producer",
    }
    assert dataset.question_set_digest == integrity_digest(["q2", "q1"])
    assert dataset.reference_kind == "independent_input_snapshot"


def test_reference_kinds_are_not_equivalent():
    rows = _rows()
    independent = build_expected_dataset(
        _declaration(), {"reference": _source("reference", rows)}
    )
    derived = build_expected_dataset(
        _derived_declaration(),
        {
            "results-a": _source("results-a", rows),
            "results-b": _source("results-b", rows),
        },
    )

    assert independent.artifact_id == derived.artifact_id
    assert independent.selection_id == derived.selection_id
    assert independent.reference_kind == "independent_input_snapshot"
    assert derived.reference_kind == "profile_derived_reference_snapshot"
    assert independent.derivation_digest != derived.derivation_digest
    assert independent.snapshot_digest != derived.snapshot_digest
    assert "not independently authenticated" in derived.limitations


def test_profile_derived_reference_requires_two_independent_source_groups():
    declaration = _derived_declaration(
        derivation={
            "method": "exact agreement",
            "independent_source_groups": [["results-a", "results-b"]],
        }
    )
    with pytest.raises(DatasetReferenceError, match="two independent source groups"):
        build_expected_dataset(
            declaration,
            {
                "results-a": _source("results-a", _rows()),
                "results-b": _source("results-b", _rows()),
            },
        )


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("stem", "Disagrees?", "question_text"),
        ("gold", "C", "correct_option"),
        ("option_2", "different option", "ordered options"),
    ],
)
def test_profile_derived_reference_refuses_cross_source_semantic_disagreement(
    field: str, replacement: str, message: str
):
    changed = _rows()
    changed[0][field] = replacement
    with pytest.raises(DatasetReferenceError, match=message):
        build_expected_dataset(
            _derived_declaration(expected_question_ids=("q1", "q2")),
            {
                "results-a": _source("results-a", _rows()),
                "results-b": _source("results-b", changed),
            },
        )


def test_derivation_digest_binds_complete_stable_reference_chain_not_audit_path():
    declaration = _declaration()
    first = build_expected_dataset(
        declaration,
        {
            "reference": _source(
                "reference", _rows(), audit_path=Path("/machine-a/input.csv")
            )
        },
    )
    relocated = build_expected_dataset(
        declaration,
        {
            "reference": _source(
                "reference", _rows(), audit_path=Path("/machine-b/input.csv")
            )
        },
    )
    changed_chain = build_expected_dataset(
        declaration,
        {
            "reference": _source(
                "reference", _rows(), logical_path="publisher/mirror/input.csv"
            )
        },
    )

    assert first.derivation_digest == integrity_digest(first.derivation)
    assert relocated.derivation_digest == first.derivation_digest
    assert changed_chain.derivation_digest != first.derivation_digest


def test_unknown_publisher_revision_is_an_explicit_limitation():
    dataset = build_expected_dataset(
        _declaration(revision=None), {"reference": _source("reference", _rows())}
    )
    assert "publisher revision was not recorded" in dataset.limitations


def test_duplicate_selected_question_id_is_refused():
    with pytest.raises(DatasetReferenceError, match="duplicate selected question ID.*q1"):
        build_expected_dataset(
            _declaration(expected_question_ids=("q1", "q1")),
            {"reference": _source("reference", _rows())},
        )


def test_missing_selected_question_id_is_refused():
    with pytest.raises(DatasetReferenceError, match="missing selected question ID.*absent"):
        build_expected_dataset(
            _declaration(expected_question_ids=("q1", "absent")),
            {"reference": _source("reference", _rows())},
        )


def test_duplicate_source_question_id_is_refused():
    rows = _rows()
    rows.append(dict(rows[0]))
    with pytest.raises(DatasetReferenceError, match="duplicate question_id.*q1"):
        build_expected_dataset(
            _declaration(expected_question_ids=("q1",)),
            {"reference": _source("reference", rows)},
        )


def test_six_options_are_preserved_in_order():
    dataset = build_expected_dataset(
        _declaration(columns=_columns(6), expected_question_ids=("q1",)),
        {"reference": _source("reference", _rows(six_options=True))},
    )
    assert build_option_map(dataset.frame.iloc[0].to_dict()) == {
        "A": "one",
        "B": "two",
        "C": "three",
        "D": "four",
        "E": "five",
        "F": "six",
    }


def test_arc_style_three_options_have_no_phantom_fourth_option():
    dataset = build_expected_dataset(
        _declaration(expected_question_ids=("q1",)),
        {"reference": _source("reference", _rows())},
    )
    assert build_option_map(dataset.frame.iloc[0].to_dict()) == {
        "A": "one",
        "B": "two",
        "C": "three",
    }
    assert "choice_d" not in dataset.frame.columns


def test_source_mapping_trust_and_logical_provenance_are_not_semantic_identity():
    ordinary = build_expected_dataset(
        _declaration(), {"reference": _source("reference", _rows())}
    )
    renamed_rows = [
        {
            "id": row["qid"],
            "prompt": row["stem"],
            "answer": row["gold"],
            "a": row["option_1"],
            "b": row["option_2"],
            "c": row["option_3"],
        }
        for row in _rows()
    ]
    remapped = build_expected_dataset(
        _declaration(
            trust_label="archive-copy",
            columns={
                "question_id": "id",
                "question_text": "prompt",
                "correct_option": "answer",
                "choice_a": "a",
                "choice_b": "b",
                "choice_c": "c",
            },
        ),
        {
            "reference": _source(
                "reference", renamed_rows, logical_path="archive/remapped.csv"
            )
        },
    )

    assert remapped.artifact_id == ordinary.artifact_id
    assert remapped.artifact_digest == ordinary.artifact_digest
    assert remapped.selection_id == ordinary.selection_id
    assert remapped.selection_digest == ordinary.selection_digest
    assert remapped.derivation_digest != ordinary.derivation_digest
    assert remapped.snapshot_digest != ordinary.snapshot_digest


def test_semantic_content_membership_and_order_change_applicable_identities():
    base_rows = _rows()
    base = build_expected_dataset(
        _declaration(), {"reference": _source("reference", base_rows)}
    )
    changed_gold_rows = [dict(row) for row in base_rows]
    changed_gold_rows[1]["gold"] = "B"
    changed_gold = build_expected_dataset(
        _declaration(), {"reference": _source("reference", changed_gold_rows)}
    )
    changed_option_rows = [dict(row) for row in base_rows]
    changed_option_rows[1]["option_1"] = "scarlet"
    changed_option = build_expected_dataset(
        _declaration(), {"reference": _source("reference", changed_option_rows)}
    )
    changed_membership = build_expected_dataset(
        _declaration(expected_question_ids=("q2", "q3")),
        {"reference": _source("reference", base_rows)},
    )
    changed_order = build_expected_dataset(
        _declaration(expected_question_ids=("q1", "q2")),
        {"reference": _source("reference", base_rows)},
    )

    for changed in (changed_gold, changed_option):
        assert changed.artifact_id != base.artifact_id
        assert changed.selection_id != base.selection_id
    for changed in (changed_membership, changed_order):
        assert changed.artifact_id == base.artifact_id
        assert changed.selection_id != base.selection_id


def test_snapshot_is_written_atomically_and_self_validates(tmp_path: Path):
    dataset = build_expected_dataset(
        _declaration(), {"reference": _source("reference", _rows())}
    )
    record = write_expected_snapshot(tmp_path, dataset)

    assert record["artifact_digest"] == dataset.artifact_digest
    assert record["selection_digest"] == dataset.selection_digest
    assert record["derivation_digest"] == dataset.derivation_digest
    assert record["reference_kind"] == dataset.reference_kind
    assert record["selected_question_ids"] == ["q2", "q1"]
    assert (tmp_path / record["run_snapshot_path"]).is_file()
    assert (tmp_path / record["reference_metadata_path"]).is_file()
    validate_expected_snapshot(tmp_path, record)

    snapshot = tmp_path / record["run_snapshot_path"]
    snapshot.write_text(snapshot.read_text(encoding="utf-8") + "tampered", encoding="utf-8")
    with pytest.raises(DatasetReferenceError, match="snapshot.*integrity"):
        validate_expected_snapshot(tmp_path, record)


def test_snapshot_self_validation_preserves_known_selection_semantics(tmp_path: Path):
    declaration = replace(
        _declaration(),
        selection_seed=17,
        selection_n_samples=2,
        subject_filter=("science",),
        selection_unknown_reasons={},
    )
    dataset = build_expected_dataset(
        declaration, {"reference": _source("reference", _rows())}
    )
    record = write_expected_snapshot(tmp_path, dataset)

    assert record["selection_semantics"] == {
        "seed": 17,
        "n_samples": 2,
        "subject_filter": ["science"],
    }
    validate_expected_snapshot(tmp_path, record)


def test_known_selection_count_must_equal_selected_question_coverage():
    declaration = replace(
        _declaration(),
        selection_n_samples=99,
        selection_unknown_reasons={"selection_seed": "not recorded"},
    )
    with pytest.raises(DatasetReferenceError, match="n_samples|selected|coverage"):
        build_expected_dataset(
            declaration,
            {"reference": _source("reference", _rows())},
        )


def test_validated_native_compatibility_identity_is_preserved_exactly():
    full_artifact = build_expected_dataset(
        _declaration(expected_question_ids=("q1", "q2", "q3")),
        {"reference": _source("reference", _rows())},
    )
    native_artifact_payload = {
        "spec": {
            "benchmark": "synthetic",
            "split": "test",
            "hf_path": "publisher/synthetic",
            "hf_subset": None,
            "source_revision": "immutable-r1",
            "normalization_version": NORMALIZATION_VERSION,
            "transforms": [],
            "output_name": "synthetic",
        },
        "content_digest": dataset_content_digest(full_artifact.frame),
        "source": {"revision": "immutable-r1"},
    }
    artifact_digest = integrity_digest(native_artifact_payload)
    artifact_id = short_id("ds", native_artifact_payload)
    native_selection_payload = {
        "artifact_id": artifact_id,
        "content_digest": dataset_content_digest(full_artifact.frame.iloc[[1, 0]]),
        "sample_identities": dataset_sample_identities(full_artifact.frame.iloc[[1, 0]]),
        "seed": 23,
        "n_samples": 2,
        "subject_filter": ["science"],
    }
    native = {
        "artifact_payload": native_artifact_payload,
        "artifact_digest": artifact_digest,
        "artifact_id": artifact_id,
        "selection_payload": native_selection_payload,
        "selection_digest": integrity_digest(native_selection_payload),
        "selection_id": short_id("sel", native_selection_payload),
    }
    declaration = replace(
        _declaration(),
        selection_seed=23,
        selection_n_samples=2,
        subject_filter=("science",),
        selection_unknown_reasons={},
        native_compatibility_identity=native,
    )

    dataset = build_expected_dataset(
        declaration, {"reference": _source("reference", _rows())}
    )

    assert dataset.identity_mode == "native_compatibility"
    assert dataset.artifact_payload == native_artifact_payload
    assert dataset.selection_payload == native_selection_payload
    assert dataset.artifact_id == artifact_id
    assert dataset.selection_id == native["selection_id"]
    assert len(dataset.artifact_frame) == 3
    assert len(dataset.frame) == 2


def test_native_compatibility_identity_must_own_the_selected_semantic_rows():
    fallback = build_expected_dataset(
        _declaration(), {"reference": _source("reference", _rows())}
    )
    artifact_payload = {
        "spec": {
            "benchmark": "synthetic",
            "split": "test",
            "hf_path": None,
            "hf_subset": None,
            "source_revision": None,
            "normalization_version": NORMALIZATION_VERSION,
            "transforms": [],
            "output_name": "synthetic",
        },
        "content_digest": "0" * 64,
        "source": {},
    }
    artifact_id = short_id("ds", artifact_payload)
    selection_payload = {
        "artifact_id": artifact_id,
        "content_digest": dataset_content_digest(fallback.frame),
        "sample_identities": dataset_sample_identities(fallback.frame),
        "seed": 1,
        "n_samples": 2,
        "subject_filter": [],
    }
    native = {
        "artifact_payload": artifact_payload,
        "artifact_digest": integrity_digest(artifact_payload),
        "artifact_id": artifact_id,
        "selection_payload": selection_payload,
        "selection_digest": integrity_digest(selection_payload),
        "selection_id": short_id("sel", selection_payload),
    }
    declaration = replace(
        _declaration(),
        selection_seed=1,
        selection_n_samples=2,
        selection_unknown_reasons={},
        native_compatibility_identity=native,
    )

    with pytest.raises(DatasetReferenceError, match="native compatibility.*content"):
        build_expected_dataset(
            declaration, {"reference": _source("reference", _rows())}
        )


def test_unselected_semantic_content_changes_artifact_and_selection_identity():
    rows = _rows()
    original = build_expected_dataset(
        _declaration(), {"reference": _source("reference", rows)}
    )
    changed_rows = [dict(row) for row in rows]
    changed_rows[2]["stem"] = "Changed unselected question?"
    changed = build_expected_dataset(
        _declaration(), {"reference": _source("reference", changed_rows)}
    )

    assert dataset_content_digest(original.frame) == dataset_content_digest(changed.frame)
    assert original.artifact_id != changed.artifact_id
    assert original.selection_id != changed.selection_id


@pytest.mark.parametrize("source_index", ["not-an-integer", -1, 1])
def test_structured_choice_source_indices_fail_closed(source_index):
    rows = [
        {
            "qid": "q1",
            "stem": "First?",
            "gold": "A",
            "choices": json.dumps(
                [
                    {"text": "one", "source_index": source_index},
                    {"text": "two", "source_index": 1},
                ]
            ),
        }
    ]
    declaration = _declaration(
        expected_question_ids=("q1",),
        columns={
            "question_id": "qid",
            "question_text": "stem",
            "correct_option": "gold",
            "choices_json": "choices",
        },
    )

    with pytest.raises(DatasetReferenceError, match="source_index"):
        build_expected_dataset(
            declaration, {"reference": _source("reference", rows)}
        )


def test_ordered_options_reject_an_empty_middle_value():
    rows = _rows()
    rows[0]["option_2"] = ""

    with pytest.raises(DatasetReferenceError, match="empty option.*followed"):
        build_expected_dataset(
            _declaration(expected_question_ids=("q1",)),
            {"reference": _source("reference", rows)},
        )


def test_ordered_option_mapping_requires_contiguous_semantic_keys():
    declaration = _declaration(
        expected_question_ids=("q1",),
        columns={
            "question_id": "qid",
            "question_text": "stem",
            "correct_option": "gold",
            "choice_a": "option_1",
            "choice_c": "option_3",
        },
    )

    with pytest.raises(DatasetReferenceError, match="contiguous"):
        build_expected_dataset(
            declaration, {"reference": _source("reference", _rows())}
        )


def test_current_normalized_integer_fields_preserve_native_semantic_types():
    rows = _rows()
    for row in rows:
        row["correct_index"] = str(ord(row["gold"]) - ord("A"))
        row["n_choices"] = "3"
    declaration = _declaration(
        columns={
            **_columns(),
            "correct_index": "correct_index",
            "n_choices": "n_choices",
        }
    )

    dataset = build_expected_dataset(
        declaration, {"reference": _source("reference", rows)}
    )

    assert dataset.artifact_frame["correct_index"].tolist() == [1, 0, 2]
    assert dataset.artifact_frame["n_choices"].tolist() == [3, 3, 3]


@pytest.mark.parametrize("mutation", ["schema", "extra_field"])
def test_snapshot_record_schema_is_exact_and_fail_closed(tmp_path: Path, mutation: str):
    dataset = build_expected_dataset(
        _declaration(), {"reference": _source("reference", _rows())}
    )
    record = write_expected_snapshot(tmp_path, dataset)
    if mutation == "schema":
        record["schema_version"] = "choicebench.expected-dataset.v999"
    else:
        record["unexpected"] = "not allowed"
    unsigned = dict(record)
    unsigned.pop("record_digest")
    record["record_digest"] = integrity_digest(unsigned)

    with pytest.raises(DatasetReferenceError, match="schema|fields"):
        validate_expected_snapshot(tmp_path, record)


def test_snapshot_validation_rejects_symlinked_artifact_paths(tmp_path: Path):
    dataset = build_expected_dataset(
        _declaration(), {"reference": _source("reference", _rows())}
    )
    record = write_expected_snapshot(tmp_path, dataset)
    snapshot = tmp_path / record["run_snapshot_path"]
    outside = tmp_path.parent / f"{tmp_path.name}-outside.csv"
    outside.write_bytes(snapshot.read_bytes())
    snapshot.unlink()
    snapshot.symlink_to(outside)

    with pytest.raises(DatasetReferenceError, match="symlink|unsafe"):
        validate_expected_snapshot(tmp_path, record)
