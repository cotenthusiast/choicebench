from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
from pathlib import Path

import pytest

from choicebench.importing.csv_adapter import (
    CsvAdapterError,
    OpenedSource,
    effective_csv_adapter_projection,
    parse_csv_source,
    scan_csv_logical_records,
)
from choicebench.importing.schema import (
    CsvDialectSpec,
    NumericColumnSpec,
    OptionMappingSpec,
    SourceArtifactSpec,
)


def _opened(tmp_path: Path, data: bytes) -> OpenedSource:
    path = tmp_path / "source.csv"
    path.write_bytes(data)
    return OpenedSource(
        source_id="source",
        audit_path=path,
        logical_path="freeze/source.csv",
        data=data,
        sha256=sha256(data).hexdigest(),
    )


def _source_spec(
    data: bytes,
    *,
    expected_columns: tuple[str, ...] = ("id", "value"),
    columns: dict[str, str] | None = None,
    ignored_columns: dict[str, str] | None = None,
    null_values: tuple[str, ...] = (),
    numeric_columns: tuple[NumericColumnSpec, ...] = (),
    option_mapping: OptionMappingSpec | None = None,
    extra_field_policy: str = "preserve_unmapped",
    dialect: CsvDialectSpec | None = None,
) -> SourceArtifactSpec:
    return SourceArtifactSpec(
        source_id="source",
        path=Path("/explicit/read-only/source.csv"),
        logical_path="freeze/source.csv",
        expected_sha256=sha256(data).hexdigest(),
        format="csv",
        format_version="producer-v1",
        classification="raw",
        dialect=dialect or CsvDialectSpec(),
        columns=columns or {"question_id": "id", "prediction": "value"},
        expected_columns=expected_columns,
        ignored_columns=ignored_columns or {},
        null_values=null_values,
        numeric_columns=numeric_columns,
        option_mapping=option_mapping
        or OptionMappingSpec("ordered_columns", ("value",), None, None, None),
        extra_field_policy=extra_field_policy,  # type: ignore[arg-type]
        preserve_namespace="producer",
        source_run_id=None,
        source_repository=None,
        source_commit=None,
        notes={},
    )


def _parse(
    tmp_path: Path,
    data: bytes,
    *,
    strict: bool = False,
    **spec_kwargs: object,
):
    return parse_csv_source(
        _opened(tmp_path, data), _source_spec(data, **spec_kwargs), strict=strict
    )


@pytest.mark.parametrize(
    ("data", "terminators"),
    [
        (b"id,value\nq1,x\n", (b"\n", b"\n")),
        (b"id,value\r\nq1,x\r\n", (b"\r\n", b"\r\n")),
        (b"id,value\rq1,x\r", (b"\r", b"\r")),
        (b"id,value\r\nq1,x\nq2,y\r", (b"\r\n", b"\n", b"\r")),
    ],
)
def test_scanner_recognizes_declared_terminators(data: bytes, terminators: tuple[bytes, ...]):
    spans = scan_csv_logical_records(data, CsvDialectSpec())
    assert tuple(span.terminator for span in spans) == terminators
    assert b"".join(data[span.start : span.end] for span in spans) == data


def test_embedded_newline_span_is_exact():
    data = b'id,text\r\nq1,"line one\r\nline two"\r\nq2,end\n'
    spans = scan_csv_logical_records(data, CsvDialectSpec())
    assert data[spans[1].start : spans[1].end] == b'q1,"line one\r\nline two"\r\n'
    assert spans[1].terminator == b"\r\n"


@pytest.mark.parametrize("embedded", [b"\r", b"\n", b"\r\n"])
def test_each_embedded_terminator_remains_inside_quoted_record(embedded: bytes):
    data = b'id,text\nq1,"a' + embedded + b'b"\n'
    spans = scan_csv_logical_records(data, CsvDialectSpec())
    assert len(spans) == 2
    assert data[spans[1].start : spans[1].end] == b'q1,"a' + embedded + b'b"\n'


def test_doubled_quote_does_not_end_quoted_field():
    data = b'id,text\nq1,"a""b\nc"\n'
    spans = scan_csv_logical_records(data, CsvDialectSpec(double_quote=True))
    assert len(spans) == 2
    assert data[spans[1].start : spans[1].end] == b'q1,"a""b\nc"\n'


def test_explicit_escape_protects_quote_and_newline():
    data = b'id,text\nq1,"a\\"b\nc"\n'
    dialect = CsvDialectSpec(escape_character="\\", double_quote=False)
    spans = scan_csv_logical_records(data, dialect)
    assert len(spans) == 2
    assert data[spans[1].start : spans[1].end] == b'q1,"a\\"b\nc"\n'


def test_explicit_escape_outside_quotes_protects_one_lf_byte():
    data = b"id,text\nq1,a\\\nb\n"
    dialect = CsvDialectSpec(escape_character="\\")
    spans = scan_csv_logical_records(data, dialect)
    assert len(spans) == 2
    assert data[spans[1].start : spans[1].end] == b"q1,a\\\nb\n"


def test_unclosed_quote_is_rejected_with_exact_record_start():
    data = b'id,text\nq1,"unterminated\n'
    with pytest.raises(CsvAdapterError, match=r"unclosed quote.*byte 8"):
        scan_csv_logical_records(data, CsvDialectSpec())


def test_final_record_without_terminator_is_preserved():
    data = b"id,value\nq1,x"
    spans = scan_csv_logical_records(data, CsvDialectSpec())
    assert spans[-1].terminator == b""
    assert data[spans[-1].start : spans[-1].end] == b"q1,x"


def test_forbidden_final_record_without_terminator_is_rejected():
    data = b"id,value\nq1,x"
    dialect = CsvDialectSpec(final_record_without_terminator="forbid")
    with pytest.raises(CsvAdapterError, match="final logical record"):
        scan_csv_logical_records(data, dialect)


def test_forbidden_mixed_terminators_are_rejected():
    data = b"id,value\r\nq1,x\n"
    dialect = CsvDialectSpec(mixed_line_terminators="forbid")
    with pytest.raises(CsvAdapterError, match="mixed line terminators"):
        scan_csv_logical_records(data, dialect)


@pytest.mark.parametrize(
    ("declared", "actual"),
    [
        (("lf",), b"\r\n"),
        (("lf",), b"\r"),
        (("crlf",), b"\n"),
        (("crlf",), b"\r"),
        (("cr",), b"\r\n"),
        (("cr",), b"\n"),
    ],
)
def test_actual_undeclared_terminator_is_rejected(
    declared: tuple[str, ...], actual: bytes
):
    data = b"id,value" + actual + b"q1,x" + actual
    dialect = CsvDialectSpec(line_terminators=declared)
    with pytest.raises(CsvAdapterError, match="undeclared line terminator"):
        scan_csv_logical_records(data, dialect)


def test_mixed_policy_compares_actual_crlf_and_lf_terminators():
    data = b"id,value\r\nq1,x\n"
    dialect = CsvDialectSpec(
        line_terminators=("crlf", "lf"), mixed_line_terminators="forbid"
    )
    with pytest.raises(CsvAdapterError, match="mixed line terminators"):
        scan_csv_logical_records(data, dialect)


@pytest.mark.parametrize("blank", [b"\n", b"\r", b"\r\n"])
def test_blank_logical_records_are_rejected(blank: bytes):
    data = b"id,value\n" + blank + b"q1,x\n"
    with pytest.raises(CsvAdapterError, match="blank logical record"):
        scan_csv_logical_records(data, CsvDialectSpec())


@pytest.mark.parametrize("terminator", [b"\n", b"\r", b"\r\n"])
def test_bom_strip_rejects_blank_first_logical_record_directly(terminator: bytes):
    data = b"\xef\xbb\xbf" + terminator + b"q1,x" + terminator
    dialect = CsvDialectSpec(bom_policy="strip_utf8_bom")
    with pytest.raises(CsvAdapterError, match="blank logical record"):
        scan_csv_logical_records(data, dialect)


def test_bom_is_forbidden_by_default(tmp_path: Path):
    data = b"\xef\xbb\xbfid,value\nq1,x\n"
    with pytest.raises(CsvAdapterError, match="UTF-8 BOM"):
        _parse(tmp_path, data)


def test_bom_strip_preserves_original_byte_offsets(tmp_path: Path):
    data = b"\xef\xbb\xbfid,value\nq1,x\n"
    dialect = CsvDialectSpec(bom_policy="strip_utf8_bom")
    table = _parse(tmp_path, data, dialect=dialect)
    assert table.columns == ("id", "value")
    assert table.rows[0].span.start == len(b"\xef\xbb\xbfid,value\n")
    assert table.rows[0].values == {"id": "q1", "value": "x"}


def test_bom_strip_keeps_a_quoted_first_header_field_in_quote_state():
    data = b'\xef\xbb\xbf"id\ncontinued",value\nq1,x\n'
    dialect = CsvDialectSpec(bom_policy="strip_utf8_bom")
    spans = scan_csv_logical_records(data, dialect)
    assert len(spans) == 2
    assert data[spans[0].start : spans[0].end] == b'\xef\xbb\xbf"id\ncontinued",value\n'


def test_invalid_utf8_is_rejected_without_replacement(tmp_path: Path):
    data = b"id,value\nq1,\xff\n"
    with pytest.raises(CsvAdapterError, match="UTF-8.*logical record 1"):
        _parse(tmp_path, data)


def test_invalid_utf8_offset_in_bom_stripped_header_uses_original_bytes(tmp_path: Path):
    data = b"\xef\xbb\xbfid,\xff\nq1,x\n"
    dialect = CsvDialectSpec(bom_policy="strip_utf8_bom")
    with pytest.raises(CsvAdapterError, match=r"logical record 0 at source byte 6"):
        _parse(tmp_path, data, dialect=dialect)


def test_header_order_and_raw_row_hash_are_preserved(tmp_path: Path):
    data = b"value,id\r\nx,q1\r\n"
    table = _parse(
        tmp_path,
        data,
        expected_columns=("value", "id"),
        columns={"question_id": "id", "prediction": "value"},
    )
    assert table.columns == ("value", "id")
    assert table.rows[0].raw_sha256 == sha256(b"x,q1\r\n").hexdigest()


def test_duplicate_header_is_rejected(tmp_path: Path):
    data = b"id,id\nq1,x\n"
    with pytest.raises(CsvAdapterError, match="duplicate header"):
        _parse(tmp_path, data, expected_columns=("id",))


@pytest.mark.parametrize("credential_column", ["api_key", "nested_access_token", "password"])
def test_credential_shaped_header_is_rejected_in_every_mode(
    tmp_path: Path, credential_column: str
):
    data = f"id,value,{credential_column}\nq1,x,secret\n".encode()
    for strict in (False, True):
        with pytest.raises(CsvAdapterError, match="credential-named"):
            _parse(tmp_path, data, strict=strict)


def test_normal_mode_preserves_unmapped_extra_columns(tmp_path: Path):
    data = b"id,value,diagnostic\nq1,x,kept\n"
    table = _parse(tmp_path, data, strict=False)
    assert table.rows[0].values["diagnostic"] == "kept"


def test_cli_strict_mode_tightens_preserve_unmapped_policy(tmp_path: Path):
    data = b"id,value,diagnostic\nq1,x,kept\n"
    with pytest.raises(CsvAdapterError, match="unexpected source columns.*diagnostic"):
        _parse(tmp_path, data, strict=True)


def test_cli_normal_mode_cannot_loosen_declared_reject_policy(tmp_path: Path):
    data = b"id,value,diagnostic\nq1,x,kept\n"
    with pytest.raises(CsvAdapterError, match="unexpected source columns.*diagnostic"):
        _parse(tmp_path, data, strict=False, extra_field_policy="reject_unmapped")


def test_effective_strictness_is_identity_bearing():
    data = b"id,value\nq1,x\n"
    declaration = _source_spec(data)
    assert effective_csv_adapter_projection(
        declaration, strict=False
    ) != effective_csv_adapter_projection(declaration, strict=True)
    assert effective_csv_adapter_projection(
        replace(declaration, dialect=replace(declaration.dialect, delimiter=";")),
        strict=False,
    ) != effective_csv_adapter_projection(declaration, strict=False)


def test_declared_ignored_column_requires_reason_and_is_retained(tmp_path: Path):
    data = b"id,value,score\nq1,x,0.5\n"
    table = _parse(
        tmp_path,
        data,
        expected_columns=("id", "value", "score"),
        ignored_columns={"score": "producer aggregate only"},
    )
    assert table.rows[0].values["score"] == "0.5"


def test_missing_declared_column_is_rejected(tmp_path: Path):
    data = b"id,value\nq1,x\n"
    with pytest.raises(CsvAdapterError, match="missing declared source columns.*score"):
        _parse(tmp_path, data, expected_columns=("id", "value", "score"))


def test_empty_string_is_distinct_from_null_unless_declared(tmp_path: Path):
    data = b"id,value\nq1,\n"
    assert _parse(tmp_path, data).rows[0].values["value"] == ""
    assert _parse(tmp_path, data, null_values=("",)).rows[0].values["value"] is None


def test_literal_nan_is_not_implicit_null(tmp_path: Path):
    data = b"id,value\nq1,NaN\n"
    table = _parse(tmp_path, data, strict=True)
    assert table.rows[0].values["value"] == "NaN"


@pytest.mark.parametrize("value", [b"1", b"-2", b"+3"])
def test_integer_numeric_policy_accepts_exact_integer_strings(tmp_path: Path, value: bytes):
    data = b"id,value\nq1," + value + b"\n"
    numeric = (NumericColumnSpec("value", "integer", False),)
    assert _parse(tmp_path, data, numeric_columns=numeric).rows[0].values["value"] == value.decode()


@pytest.mark.parametrize("value", [b"1.0", b"1e2", b"x", b" 1"])
def test_integer_numeric_policy_rejects_malformed_strings(tmp_path: Path, value: bytes):
    data = b"id,value\nq1," + value + b"\n"
    numeric = (NumericColumnSpec("value", "integer", False),)
    with pytest.raises(CsvAdapterError, match="integer.*value"):
        _parse(tmp_path, data, numeric_columns=numeric)


@pytest.mark.parametrize("value", [b"1", b"-1.25", b"1e2"])
def test_float_numeric_policy_accepts_finite_numbers(tmp_path: Path, value: bytes):
    data = b"id,value\nq1," + value + b"\n"
    numeric = (NumericColumnSpec("value", "float", False),)
    _parse(tmp_path, data, numeric_columns=numeric)


@pytest.mark.parametrize("value", [b"NaN", b"inf", b"-Infinity", b"x", b" 1"])
def test_finite_float_policy_rejects_nonfinite_or_malformed_values(tmp_path: Path, value: bytes):
    data = b"id,value\nq1," + value + b"\n"
    numeric = (NumericColumnSpec("value", "float", False, finite_only=True),)
    with pytest.raises(CsvAdapterError, match="finite float.*value"):
        _parse(tmp_path, data, numeric_columns=numeric)


def test_nonfinite_float_is_allowed_only_when_explicit(tmp_path: Path):
    data = b"id,value\nq1,NaN\n"
    numeric = (NumericColumnSpec("value", "float", False, finite_only=False),)
    assert _parse(tmp_path, data, numeric_columns=numeric).rows[0].values["value"] == "NaN"


def test_numeric_null_policy_is_explicit(tmp_path: Path):
    data = b"id,value\nq1,NA\n"
    with pytest.raises(CsvAdapterError, match="null.*value"):
        _parse(
            tmp_path,
            data,
            null_values=("NA",),
            numeric_columns=(NumericColumnSpec("value", "float", False),),
        )
    table = _parse(
        tmp_path,
        data,
        null_values=("NA",),
        numeric_columns=(NumericColumnSpec("value", "float", True),),
    )
    assert table.rows[0].values["value"] is None


@pytest.mark.parametrize("option_count", [3, 6])
def test_ordered_option_mapping_supports_variable_option_counts(
    tmp_path: Path, option_count: int
):
    option_columns = tuple(f"choice_{index}" for index in range(option_count))
    header = ("id", *option_columns)
    data = (
        ",".join(header)
        + "\nq1,"
        + ",".join(f"v{index}" for index in range(option_count))
        + "\n"
    ).encode()
    table = _parse(
        tmp_path,
        data,
        expected_columns=header,
        columns={"question_id": "id"},
        option_mapping=OptionMappingSpec("ordered_columns", option_columns, None, None, None),
    )
    assert tuple(table.rows[0].values[name] for name in option_columns) == tuple(
        f"v{index}" for index in range(option_count)
    )


def test_empty_trailing_ordered_option_stays_null_not_phantom_choice(tmp_path: Path):
    data = b"id,a,b,c,d\nq1,A,B,C,\n"
    table = _parse(
        tmp_path,
        data,
        expected_columns=("id", "a", "b", "c", "d"),
        columns={"question_id": "id"},
        null_values=("",),
        option_mapping=OptionMappingSpec("ordered_columns", ("a", "b", "c", "d"), None, None, None),
    )
    assert [table.rows[0].values[name] for name in ("a", "b", "c", "d")] == ["A", "B", "C", None]


def test_structured_json_choices_accept_variable_option_list(tmp_path: Path):
    data = (
        b'id,choices\nq1,"[{""label"":""A"",""text"":""one""},'
        b'{""label"":""B"",""text"":""two""},'
        b'{""label"":""C"",""text"":""three""}]"\n'
    )
    mapping = OptionMappingSpec("structured_json", (), "choices", "label", "text")
    table = _parse(
        tmp_path,
        data,
        expected_columns=("id", "choices"),
        columns={"question_id": "id"},
        option_mapping=mapping,
    )
    assert table.rows[0].values["choices"].startswith('[{"label":"A"')


@pytest.mark.parametrize(
    "payload",
    [b"not-json", b"{}", b'[{"label":"A"}]', b'[{"label":"A","text":1}]'],
)
def test_structured_json_choices_reject_malformed_payloads(tmp_path: Path, payload: bytes):
    escaped = payload.replace(b'"', b'""')
    data = b'id,choices\nq1,"' + escaped + b'"\n'
    mapping = OptionMappingSpec("structured_json", (), "choices", "label", "text")
    with pytest.raises(CsvAdapterError, match="structured choices"):
        _parse(
            tmp_path,
            data,
            expected_columns=("id", "choices"),
            columns={"question_id": "id"},
            option_mapping=mapping,
        )


def test_structured_json_choices_accept_positional_source_index_labels(tmp_path: Path):
    """Regression: real fourth-cell CSVs use {"text": ..., "source_index": N}
    -- an integer positional index, no explicit string label -- unlike the
    explicit {"label": "A", "text": ...} shape tested above. This must be
    letterized (0 -> A, 1 -> B, ...), not rejected."""
    data = (
        b'id,choices\nq1,"[{""text"":""slower"",""source_index"":0},'
        b'{""text"":""faster"",""source_index"":1},'
        b'{""text"":""at the same speed"",""source_index"":2}]"\n'
    )
    mapping = OptionMappingSpec("structured_json", (), "choices", "source_index", "text")
    table = _parse(
        tmp_path,
        data,
        expected_columns=("id", "choices"),
        columns={"question_id": "id"},
        option_mapping=mapping,
    )
    assert table.rows[0].values["choices"].startswith('[{"text":"slower"')


def test_structured_json_choices_reject_source_index_out_of_letter_range(tmp_path: Path):
    data = b'id,choices\nq1,"[{""text"":""x"",""source_index"":99}]"\n'
    mapping = OptionMappingSpec("structured_json", (), "choices", "source_index", "text")
    with pytest.raises(CsvAdapterError, match="structured choices"):
        _parse(
            tmp_path,
            data,
            expected_columns=("id", "choices"),
            columns={"question_id": "id"},
            option_mapping=mapping,
        )


def test_structured_json_choices_reject_duplicate_source_index(tmp_path: Path):
    data = (
        b'id,choices\nq1,"[{""text"":""x"",""source_index"":0},'
        b'{""text"":""y"",""source_index"":0}]"\n'
    )
    mapping = OptionMappingSpec("structured_json", (), "choices", "source_index", "text")
    with pytest.raises(CsvAdapterError, match="duplicate labels"):
        _parse(
            tmp_path,
            data,
            expected_columns=("id", "choices"),
            columns={"question_id": "id"},
            option_mapping=mapping,
        )


def test_row_with_more_fields_than_header_is_rejected(tmp_path: Path):
    data = b"id,value\nq1,x,unexpected\n"
    with pytest.raises(CsvAdapterError, match="field count"):
        _parse(tmp_path, data)


def test_source_checksum_and_declaration_source_id_are_verified(tmp_path: Path):
    data = b"id,value\nq1,x\n"
    source = _opened(tmp_path, data)
    bad_digest = replace(_source_spec(data), expected_sha256="0" * 64)
    with pytest.raises(CsvAdapterError, match="checksum"):
        parse_csv_source(source, bad_digest, strict=False)
    bad_source_id = replace(_source_spec(data), source_id="different")
    with pytest.raises(CsvAdapterError, match="source_id"):
        parse_csv_source(source, bad_source_id, strict=False)


def test_source_logical_path_must_match_declaration(tmp_path: Path):
    data = b"id,value\nq1,x\n"
    source = replace(_opened(tmp_path, data), logical_path="other/source.csv")
    with pytest.raises(CsvAdapterError, match="logical_path"):
        parse_csv_source(source, _source_spec(data), strict=False)
