"""Deterministic byte-preserving adapter for external CSV result sources."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from hashlib import sha256
import io
import json
import math
from pathlib import Path
import re
from typing import Mapping, Protocol

from choicebench.identity import canonicalize, is_credential_key
from choicebench.importing.schema import CsvDialectSpec, SourceArtifactSpec


_UTF8_BOM = b"\xef\xbb\xbf"
_TERMINATORS = {"crlf": b"\r\n", "lf": b"\n", "cr": b"\r"}
_ACTUAL_TERMINATORS = (b"\r\n", b"\n", b"\r")
_INTEGER_RE = re.compile(r"[+-]?\d+\Z")
_FLOAT_RE = re.compile(
    r"[+-]?(?:(?:\d+(?:\.\d*)?)|(?:\.\d+))(?:[eE][+-]?\d+)?\Z"
)
_NONFINITE_FLOAT_RE = re.compile(
    r"[+-]?(?:nan|inf(?:inity)?)\Z", flags=re.IGNORECASE
)


class CsvAdapterError(ValueError):
    """Raised when source CSV bytes do not satisfy their declaration."""


@dataclass(frozen=True)
class OpenedSource:
    source_id: str
    audit_path: Path
    logical_path: str
    data: bytes
    sha256: str


@dataclass(frozen=True)
class LogicalRecordSpan:
    index: int
    start: int
    end: int
    terminator: bytes


@dataclass(frozen=True)
class SourceRow:
    values: Mapping[str, str | None]
    span: LogicalRecordSpan
    raw_sha256: str


@dataclass(frozen=True)
class AdaptedTable:
    columns: tuple[str, ...]
    rows: tuple[SourceRow, ...]
    source_sha256: str


class SourceAdapter(Protocol):
    def parse(
        self, source: OpenedSource, declaration: SourceArtifactSpec, *, strict: bool
    ) -> AdaptedTable: ...


def _dialect_bytes(dialect: CsvDialectSpec) -> tuple[int, int, int | None]:
    values = (dialect.delimiter, dialect.quote_character, dialect.escape_character)
    encoded: list[int | None] = []
    for name, value in zip(("delimiter", "quote", "escape"), values, strict=True):
        if value is None:
            encoded.append(None)
            continue
        raw = value.encode("utf-8")
        if len(raw) != 1 or raw[0] >= 128 or raw in {b"\x00", b"\r", b"\n"}:
            raise CsvAdapterError(f"CSV {name} must be one usable ASCII byte.")
        encoded.append(raw[0])
    non_null = [value for value in encoded if value is not None]
    if len(non_null) != len(set(non_null)):
        raise CsvAdapterError("CSV delimiter, quote, and escape bytes must be distinct.")
    return encoded[0], encoded[1], encoded[2]  # type: ignore[return-value]


def _declared_terminators(dialect: CsvDialectSpec) -> tuple[bytes, ...]:
    try:
        declared = tuple(_TERMINATORS[name] for name in dialect.line_terminators)
    except KeyError as exc:
        raise CsvAdapterError(f"Unsupported CSV line terminator {exc.args[0]!r}.") from exc
    if not declared or len(declared) != len(set(declared)):
        raise CsvAdapterError("CSV line terminators must be a non-empty unique list.")
    return tuple(sorted(declared, key=len, reverse=True))


def scan_csv_logical_records(
    data: bytes, dialect: CsvDialectSpec
) -> tuple[LogicalRecordSpan, ...]:
    """Return exact source-byte spans for logical CSV records."""
    if not isinstance(data, bytes):
        raise TypeError(f"data must be bytes; got {type(data).__name__}.")
    delimiter, quote, escape = _dialect_bytes(dialect)
    declared_terminators = set(_declared_terminators(dialect))
    spans: list[LogicalRecordSpan] = []
    observed_terminators: set[bytes] = set()
    start = 0
    if data.startswith(_UTF8_BOM):
        if dialect.bom_policy == "forbid":
            raise CsvAdapterError("Source contains a forbidden UTF-8 BOM.")
        index = len(_UTF8_BOM)
    else:
        index = 0
    content_start = index
    in_quotes = False
    at_field_start = True

    while index < len(data):
        byte = data[index]
        if in_quotes:
            if escape is not None and byte == escape:
                if index + 1 >= len(data):
                    raise CsvAdapterError(
                        f"CSV escape byte at byte {index} has no following byte."
                    )
                index += 2
                continue
            if byte == quote:
                if (
                    dialect.double_quote
                    and index + 1 < len(data)
                    and data[index + 1] == quote
                ):
                    index += 2
                    continue
                in_quotes = False
            index += 1
            continue

        if escape is not None and byte == escape:
            if index + 1 >= len(data):
                raise CsvAdapterError(
                    f"CSV escape byte at byte {index} has no following byte."
                )
            at_field_start = False
            index += 2
            continue
        terminator = next(
            (
                candidate
                for candidate in _ACTUAL_TERMINATORS
                if data.startswith(candidate, index)
            ),
            None,
        )
        if terminator is not None:
            if terminator not in declared_terminators:
                raise CsvAdapterError(
                    f"CSV contains undeclared line terminator {terminator!r} "
                    f"at byte {index}."
                )
            end = index + len(terminator)
            if index == content_start:
                raise CsvAdapterError(f"CSV blank logical record at byte {start} is forbidden.")
            observed_terminators.add(terminator)
            if (
                dialect.mixed_line_terminators == "forbid"
                and len(observed_terminators) > 1
            ):
                raise CsvAdapterError("CSV contains forbidden mixed line terminators.")
            spans.append(LogicalRecordSpan(len(spans), start, end, terminator))
            start = end
            index = end
            content_start = end
            at_field_start = True
            continue
        if byte == delimiter:
            at_field_start = True
        elif byte == quote and at_field_start:
            in_quotes = True
            at_field_start = False
        elif not (dialect.skip_initial_space and at_field_start and byte == 0x20):
            at_field_start = False
        index += 1

    if in_quotes:
        raise CsvAdapterError(
            f"CSV has an unclosed quote in logical record starting at byte {start}."
        )
    if start < len(data):
        if dialect.final_record_without_terminator == "forbid":
            raise CsvAdapterError("CSV final logical record has no declared terminator.")
        spans.append(LogicalRecordSpan(len(spans), start, len(data), b""))
    return tuple(spans)


def effective_csv_adapter_projection(
    declaration: SourceArtifactSpec, *, strict: bool
) -> dict[str, object]:
    """Return parsing controls that must participate in realization identity."""
    effective_extra_policy = (
        "reject_unmapped"
        if strict or declaration.extra_field_policy == "reject_unmapped"
        else "preserve_unmapped"
    )
    return canonicalize(
        {
            "adapter": "choicebench.csv.v1",
            "dialect": asdict(declaration.dialect),
            "declared_extra_field_policy": declaration.extra_field_policy,
            "cli_strict": strict,
            "effective_extra_field_policy": effective_extra_policy,
        }
    )


def _decode_record(
    data: bytes,
    span: LogicalRecordSpan,
    dialect: CsvDialectSpec,
    *,
    strip_bom: bool,
) -> list[str]:
    record = data[span.start : span.end - len(span.terminator) if span.terminator else span.end]
    stripped_prefix_size = 0
    if strip_bom:
        stripped_prefix_size = len(_UTF8_BOM)
        record = record[len(_UTF8_BOM) :]
    try:
        text = record.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise CsvAdapterError(
            f"Invalid UTF-8 in CSV logical record {span.index} at source byte "
            f"{span.start + stripped_prefix_size + exc.start}."
        ) from exc
    try:
        parsed = list(
            csv.reader(
                io.StringIO(text, newline=""),
                delimiter=dialect.delimiter,
                quotechar=dialect.quote_character,
                escapechar=dialect.escape_character,
                doublequote=dialect.double_quote,
                skipinitialspace=dialect.skip_initial_space,
                strict=dialect.strict_syntax,
            )
        )
    except csv.Error as exc:
        raise CsvAdapterError(
            f"Malformed CSV syntax in logical record {span.index}: {type(exc).__name__}."
        ) from exc
    if len(parsed) != 1:
        raise CsvAdapterError(
            f"CSV logical record {span.index} parsed into {len(parsed)} physical rows."
        )
    return parsed[0]


def _validate_numeric(
    value: str | None, *, source_column: str, value_type: str,
    null_allowed: bool, finite_only: bool, record_index: int
) -> None:
    if value is None:
        if not null_allowed:
            raise CsvAdapterError(
                f"CSV null is forbidden for numeric column {source_column!r} "
                f"in logical record {record_index}."
            )
        return
    if value_type == "integer":
        if _INTEGER_RE.fullmatch(value) is None:
            raise CsvAdapterError(
                f"CSV integer column {source_column!r} has a malformed value in "
                f"logical record {record_index}."
            )
        return
    if _FLOAT_RE.fullmatch(value) is not None:
        number = float(value)
    elif _NONFINITE_FLOAT_RE.fullmatch(value) is not None:
        number = float(value)
    else:
        raise CsvAdapterError(
            f"CSV finite float column {source_column!r} has a malformed value in "
            f"logical record {record_index}."
        )
    if finite_only and not math.isfinite(number):
        raise CsvAdapterError(
            f"CSV finite float column {source_column!r} has a non-finite value in "
            f"logical record {record_index}."
        )


_POSITIONAL_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def _validate_structured_choices(
    value: str | None, declaration: SourceArtifactSpec, *, record_index: int
) -> None:
    mapping = declaration.option_mapping
    if mapping.mode != "structured_json":
        return
    if value is None:
        raise CsvAdapterError(
            f"CSV structured choices are null in logical record {record_index}."
        )
    try:
        choices = json.loads(value)
    except (json.JSONDecodeError, UnicodeError) as exc:
        raise CsvAdapterError(
            f"CSV structured choices are malformed in logical record {record_index}."
        ) from exc
    if not isinstance(choices, list) or not choices:
        raise CsvAdapterError(
            f"CSV structured choices must be a non-empty list in logical record {record_index}."
        )
    assert mapping.structured_label_key is not None
    assert mapping.structured_text_key is not None
    labels: set[str] = set()
    for choice in choices:
        if not isinstance(choice, Mapping):
            raise CsvAdapterError(
                f"CSV structured choices contain a non-object in logical record {record_index}."
            )
        raw_label = choice.get(mapping.structured_label_key)
        text = choice.get(mapping.structured_text_key)
        # A structured-choice source may declare an explicit string letter
        # label (e.g. {"label": "A", "text": ...}), or a positional integer
        # index instead (e.g. {"source_index": 0, "text": ...} -- the same
        # convention choicebench's own dataset-side choices_json already
        # uses, see dataset_reference.py). The latter is letterized here
        # (0 -> "A", 1 -> "B", ...) rather than requiring every producer to
        # pre-compute letters that don't otherwise exist in their data.
        if isinstance(raw_label, bool):
            label = None
        elif isinstance(raw_label, int):
            label = (
                _POSITIONAL_LETTERS[raw_label]
                if 0 <= raw_label < len(_POSITIONAL_LETTERS) else None
            )
        elif isinstance(raw_label, str) and raw_label:
            label = raw_label
        else:
            label = None
        if label is None or not isinstance(text, str):
            raise CsvAdapterError(
                "CSV structured choices lack a string label or a valid positional "
                f"integer label, or a string text field, in logical record {record_index}."
            )
        if label in labels:
            raise CsvAdapterError(
                f"CSV structured choices contain duplicate labels in logical record {record_index}."
            )
        labels.add(label)


def parse_csv_source(
    source: OpenedSource, declaration: SourceArtifactSpec, *, strict: bool
) -> AdaptedTable:
    """Validate and parse one already-opened, checksum-bound CSV source."""
    if source.source_id != declaration.source_id:
        raise CsvAdapterError("Opened source_id does not match its source declaration.")
    if source.logical_path != declaration.logical_path:
        raise CsvAdapterError("Opened source logical_path does not match its declaration.")
    actual_sha256 = sha256(source.data).hexdigest()
    if source.sha256 != actual_sha256:
        raise CsvAdapterError("Opened source checksum does not match its bytes.")
    if declaration.expected_sha256 != actual_sha256:
        raise CsvAdapterError("Source checksum does not match the import declaration.")
    if declaration.format != "csv":
        raise CsvAdapterError(f"CSV adapter cannot parse format {declaration.format!r}.")
    if source.data.startswith(_UTF8_BOM) and declaration.dialect.bom_policy == "forbid":
        raise CsvAdapterError("Source contains a forbidden UTF-8 BOM.")

    spans = scan_csv_logical_records(source.data, declaration.dialect)
    if not spans:
        raise CsvAdapterError("CSV source has no header logical record.")
    header = tuple(
        _decode_record(
            source.data,
            spans[0],
            declaration.dialect,
            strip_bom=(
                declaration.dialect.bom_policy == "strip_utf8_bom"
                and source.data.startswith(_UTF8_BOM)
            ),
        )
    )
    duplicates = sorted({column for column in header if header.count(column) > 1})
    if duplicates:
        raise CsvAdapterError(f"CSV duplicate header columns are forbidden: {duplicates}.")
    credential_columns = sorted(column for column in header if is_credential_key(column))
    if credential_columns:
        raise CsvAdapterError(
            f"CSV credential-named columns are forbidden: {credential_columns}."
        )
    missing = sorted(set(declaration.expected_columns) - set(header))
    if missing:
        raise CsvAdapterError(f"CSV is missing declared source columns: {missing}.")
    extras = sorted(set(header) - set(declaration.expected_columns))
    effective = effective_csv_adapter_projection(declaration, strict=strict)
    if extras and effective["effective_extra_field_policy"] == "reject_unmapped":
        raise CsvAdapterError(f"CSV has unexpected source columns: {extras}.")

    rows: list[SourceRow] = []
    for span in spans[1:]:
        cells = _decode_record(
            source.data, span, declaration.dialect, strip_bom=False
        )
        if len(cells) != len(header):
            raise CsvAdapterError(
                f"CSV logical record {span.index} field count {len(cells)} does not "
                f"match header field count {len(header)}."
            )
        values: dict[str, str | None] = {
            column: None if cell in declaration.null_values else cell
            for column, cell in zip(header, cells, strict=True)
        }
        for numeric in declaration.numeric_columns:
            _validate_numeric(
                values[numeric.source_column],
                source_column=numeric.source_column,
                value_type=numeric.value_type,
                null_allowed=numeric.null_allowed,
                finite_only=numeric.finite_only,
                record_index=span.index,
            )
        if declaration.option_mapping.mode == "structured_json":
            assert declaration.option_mapping.structured_column is not None
            _validate_structured_choices(
                values[declaration.option_mapping.structured_column],
                declaration,
                record_index=span.index,
            )
        raw = source.data[span.start : span.end]
        rows.append(SourceRow(values, span, sha256(raw).hexdigest()))
    return AdaptedTable(header, tuple(rows), actual_sha256)
