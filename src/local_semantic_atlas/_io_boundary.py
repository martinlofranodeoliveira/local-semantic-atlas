"""CLI-only bounded regular-file snapshots and a closed document JSON syntax.

This captures bytes from an open regular file. It does not freeze a path after
capture or sandbox a native tokenizer/model parser.
"""

from __future__ import annotations

import json
import os
import stat

DOCUMENT_JSON_BYTES = 2_097_152


class BoundaryError(Exception):
    def __init__(self, code: str, *, operational: bool = False):
        super().__init__(code)
        self.code = code
        self.operational = operational


def _identity(info: os.stat_result) -> tuple[int, ...]:
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def read_regular(path: str, maximum: int, *, exact: int | None = None) -> bytes:
    """Read at most maximum+1 bytes, reject special files and changed metadata."""
    if type(path) is not str or not path or "\x00" in path:
        raise BoundaryError("INVALID_PATH")
    try:
        if len(path.encode("utf-8", "strict")) > 4096:
            raise BoundaryError("INVALID_PATH")
    except UnicodeError:
        raise BoundaryError("INVALID_PATH") from None
    if type(maximum) is not int or maximum < 0:
        raise ValueError("invalid internal maximum")
    if exact is not None and (type(exact) is not int or not 0 <= exact <= maximum):
        raise ValueError("invalid internal exact length")
    descriptor = None
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK)
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise BoundaryError("REGULAR_FILE_REQUIRED")
        if before.st_size > maximum or (exact is not None and before.st_size != exact):
            raise BoundaryError("FILE_SIZE")
        payload = bytearray()
        while len(payload) <= maximum:
            chunk = os.read(descriptor, min(65_536, maximum + 1 - len(payload)))
            if not chunk:
                break
            payload.extend(chunk)
        after = os.fstat(descriptor)
        if len(payload) > maximum or (exact is not None and len(payload) != exact):
            raise BoundaryError("FILE_SIZE")
        if _identity(before) != _identity(after) or after.st_size != len(payload):
            raise BoundaryError("FILE_CHANGED")
        return bytes(payload)
    except OSError:
        raise BoundaryError("FILE_IO", operational=True) from None
    finally:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                raise BoundaryError("FILE_IO", operational=True) from None


def _pairs(items: list[tuple[str, object]]) -> dict[str, object]:
    result = {}
    for key, value in items:
        if key in result:
            raise BoundaryError("DUPLICATE_JSON_KEY")
        result[key] = value
    return result


def _integer(token: str) -> int:
    if len(token) > 20:
        raise BoundaryError("JSON_NUMBER")
    return int(token)


def _no_number(_: str) -> object:
    raise BoundaryError("JSON_NUMBER")


def parse_documents(payload: bytes) -> tuple[tuple[str, str], ...]:
    """Return owned id/text string pairs; the SDK then admits text and IDs."""
    if type(payload) is not bytes or len(payload) > DOCUMENT_JSON_BYTES:
        raise BoundaryError("JSON_BYTES")
    try:
        decoded = payload.decode("utf-8", "strict")
        document = json.loads(decoded, object_pairs_hook=_pairs, parse_int=_integer,
                              parse_float=_no_number, parse_constant=_no_number)
    except (UnicodeError, json.JSONDecodeError, RecursionError):
        raise BoundaryError("JSON_SYNTAX") from None
    if type(document) is not dict or set(document) != {"schema_version", "documents"}:
        raise BoundaryError("JSON_SCHEMA")
    if type(document["schema_version"]) is not int or document["schema_version"] != 1:
        raise BoundaryError("JSON_VERSION")
    rows = document["documents"]
    if type(rows) is not list or len(rows) > 256:
        raise BoundaryError("DOCUMENT_COUNT")
    result = []
    for row in rows:
        if type(row) is not dict or set(row) != {"id", "text"}:
            raise BoundaryError("DOCUMENT_SCHEMA")
        if type(row["id"]) is not str or type(row["text"]) is not str:
            raise BoundaryError("DOCUMENT_SCHEMA")
        result.append((row["id"], row["text"]))
    return tuple(result)
