"""Versioned canonicalization, redaction, and content identities."""

from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from dataclasses import asdict, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

CANONICALIZATION_VERSION = "choicebench.canonical-json.v1"
REDACTED = "<redacted>"

_SECRET_EXACT = {
    "api_key", "apikey", "access_key", "access_key_id", "aws_access_key_id",
    "aws_secret_access_key", "authorization", "auth", "bearer", "bearer_token",
    "client_secret", "id_token", "private_key", "refresh_token", "session_token",
    "signature", "token", "x_api_key", "x_auth_token",
}
_SECRET_MARKERS = ("credential", "password", "passwd", "secret")
_SECRET_QUERY_MARKERS = _SECRET_EXACT | {
    "key", "sig", "x_amz_credential", "x_amz_security_token", "x_amz_signature",
    "x_goog_credential", "x_goog_signature",
}
_SIGNED_URL_FIELDS = {
    "se", "sig", "sp", "spr", "srt", "ss", "st", "sv",
    "x_amz_algorithm", "x_amz_credential", "x_amz_date", "x_amz_expires",
    "x_amz_security_token", "x_amz_signature", "x_amz_signedheaders",
    "x_goog_algorithm", "x_goog_credential", "x_goog_date", "x_goog_expires",
    "x_goog_signature", "x_goog_signedheaders",
}


def _normalized_key(key: str) -> str:
    return unicodedata.normalize("NFC", key).lower().replace("-", "_")


def is_secret_key(key: str) -> bool:
    """Broad secret heuristic for URL query parameters (value context only)."""
    normalized = _normalized_key(key)
    return (
        normalized in _SECRET_EXACT
        or normalized.endswith(("_api_key", "_access_token", "_auth_token", "_private_key"))
        or any(marker in normalized for marker in _SECRET_MARKERS)
    )


_CREDENTIAL_EXACT = {
    "api_key", "apikey", "x_api_key", "access_key", "access_key_id",
    "auth", "aws_access_key_id", "aws_secret_access_key", "authorization",
    "bearer_token", "client_secret", "id_token", "private_key",
    "refresh_token", "session_token", "x_auth_token", "access_token",
    "auth_token", "password", "passwd",
}


def is_credential_key(key: str) -> bool:
    """True only for unambiguous credential parameter names.

    Deliberately much narrower than the URL-query heuristics above:
    secret-shaped but plausibly scientific names ("token", "secret",
    "secret_strength", "access_token_count", "signature") must keep their
    identity-affecting values, so only names that can only mean a credential
    are treated as such. Scientific identity payloads refuse these keys
    outright (fail closed: no leak, no silent identity collision).
    """
    normalized = _normalized_key(key)
    return normalized in _CREDENTIAL_EXACT or normalized.endswith(
        ("_api_key", "_access_token", "_auth_token", "_private_key", "_client_secret", "_password")
    )


def sanitize_url(value: str) -> str:
    """Canonicalize an endpoint while retaining non-credential query semantics."""
    try:
        parts = urlsplit(value)
        port = parts.port
    except ValueError:
        return value
    if not parts.scheme or not parts.netloc:
        return value
    scheme = parts.scheme.lower()
    hostname = (parts.hostname or "").lower()
    if ":" in hostname and not hostname.startswith("["):
        hostname = f"[{hostname}]"
    if port is not None and not ((scheme == "https" and port == 443) or (scheme == "http" and port == 80)):
        hostname = f"{hostname}:{port}"
    pairs = parse_qsl(parts.query, keep_blank_values=True)
    normalized_names = {_normalized_key(key) for key, _ in pairs}
    is_signed = bool(normalized_names & {"sig", "x_amz_signature", "x_goog_signature"})
    query = []
    for key, item in pairs:
        normalized = _normalized_key(key)
        secret = (
            normalized in _SECRET_QUERY_MARKERS or is_secret_key(normalized)
            or (is_signed and normalized in _SIGNED_URL_FIELDS)
        )
        if secret:
            continue
        query.append((unicodedata.normalize("NFC", key), unicodedata.normalize("NFC", item)))
    query.sort()
    path = parts.path or ""
    if path != "/":
        path = path.rstrip("/")
    return urlunsplit((scheme, hostname, path, urlencode(query), ""))


_BEARER_RE = re.compile(r"(?i)\b(bearer\s+)[A-Za-z0-9._~+/=-]+")
_ASSIGNMENT_RE = re.compile(
    r"(?i)\b(api[-_ ]?key|access[-_ ]?token|auth(?:orization)?|client[-_ ]?secret|"
    r"id[-_ ]?token|password|private[-_ ]?key|refresh[-_ ]?token|session[-_ ]?token|"
    r"x[-_ ]?api[-_ ]?key|x[-_ ]?auth[-_ ]?token)\b(\s*[:=]\s*)([^\s,;]+)"
)
_URL_RE = re.compile(r"https?://[^\s'\"<>]+", re.IGNORECASE)


def redact_text(value: object) -> str:
    """Remove common credentials from untrusted exception/log text."""
    text = str(value)
    text = _URL_RE.sub(lambda m: sanitize_url(m.group(0)), text)
    text = _BEARER_RE.sub(r"\1" + REDACTED, text)
    text = _ASSIGNMENT_RE.sub(lambda m: f"{m.group(1)}{m.group(2)}{REDACTED}", text)
    return text


def canonicalize(value: Any, *, redact_secrets: bool = True) -> Any:
    """Convert supported values to deterministic JSON-native canonical-json.v1 data."""
    if is_dataclass(value):
        value = asdict(value)
    if isinstance(value, Enum):
        return canonicalize(value.value, redact_secrets=redact_secrets)
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            bad = next(key for key in value if not isinstance(key, str))
            raise TypeError(f"Canonical mapping keys must be strings; got {type(bad).__name__}.")
        normalized_items: list[tuple[str, Any]] = []
        seen: set[str] = set()
        for original_key, item in value.items():
            key = unicodedata.normalize("NFC", original_key)
            if key in seen:
                raise ValueError(f"Canonical mapping contains duplicate normalized key {key!r}.")
            seen.add(key)
            if redact_secrets and is_credential_key(key):
                # Refusing (rather than redacting) keeps both guarantees at
                # once: a credential can never be persisted or hashed, and two
                # scientifically different configs can never collapse into one
                # identity because their values were blanked before hashing.
                raise ValueError(
                    f"Credential-named key {key!r} is not allowed in identity/manifest "
                    "payloads. ChoiceBench reads credentials from environment variables; "
                    "remove the value, or rename the parameter if it is scientific "
                    "configuration."
                )
            if key.lower() in {"base_url", "endpoint", "url"} and isinstance(item, str):
                item = sanitize_url(item)
            normalized_items.append((key, canonicalize(item, redact_secrets=redact_secrets)))
        return {key: item for key, item in sorted(normalized_items)}
    if isinstance(value, (list, tuple)):
        return [canonicalize(item, redact_secrets=redact_secrets) for item in value]
    if isinstance(value, (set, frozenset)):
        items = [canonicalize(item, redact_secrets=redact_secrets) for item in value]
        return sorted(items, key=lambda item: json.dumps(item, sort_keys=True, separators=(",", ":"), ensure_ascii=True))
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, (bytes, bytearray)):
        return bytes(value).hex()
    if isinstance(value, str):
        text = unicodedata.normalize("NFC", value)
        return redact_text(text) if redact_secrets else text
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Canonical identities cannot contain NaN or infinity.")
        return 0.0 if value == 0.0 else value
    if value is None or isinstance(value, (int, bool)):
        return value
    if hasattr(value, "item"):
        return canonicalize(value.item(), redact_secrets=redact_secrets)
    raise TypeError(f"Unsupported canonical identity value: {type(value).__name__}")


def canonical_json(value: Any, *, redact_secrets: bool = True) -> str:
    return json.dumps(
        canonicalize(value, redact_secrets=redact_secrets),
        sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False,
    )


def stable_digest(value: Any, *, redact_secrets: bool = True) -> str:
    return hashlib.sha256(canonical_json(value, redact_secrets=redact_secrets).encode()).hexdigest()


def integrity_digest(value: Any) -> str:
    """Digest trusted scientific content without name-based redaction."""
    return stable_digest(value, redact_secrets=False)


def short_id(prefix: str, value: Any, length: int = 16) -> str:
    return f"{prefix}_{stable_digest(value, redact_secrets=True)[:length]}"


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
