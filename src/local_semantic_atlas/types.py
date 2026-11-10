"""Core public types, constants, admission validators, and JSON reporting."""

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
import math
import re
from typing import Final

MODEL_ID: Final[str] = "intfloat/multilingual-e5-small"
MODEL_REVISION: Final[str] = "614241f622f53c4eeff9890bdc4f31cfecc418b3"
MODEL_BYTE_LENGTH: Final[int] = 118346824
MODEL_SHA256: Final[str] = (
    "dd476dd0c2514e9b9be83aeb3853fac0763e0bdf4a71645407587d77c48a2d88"
)
TOKENIZER_BYTE_LENGTH: Final[int] = 17082730
TOKENIZER_SHA256: Final[str] = (
    "0b44a9d7b51c3c62626640cda0e2c2f70fdacdc25bbbd68038369d14ebdf4c39"
)
EMBEDDING_DIM: Final[int] = 384

MAX_DOCUMENTS: Final[int] = 256
MAX_TOKENS: Final[int] = 65536
MAX_SEQUENCE_TOKENS: Final[int] = 256
MAX_TEXT_BYTES: Final[int] = 8192
MAX_TOTAL_DOCUMENT_BYTES: Final[int] = 1048576
MAX_REPORT_JSON_BYTES: Final[int] = 262144
COSINE_OVERSHOOT_TOLERANCE: Final[float] = 1e-12
SCORE_UNIT_COSINE: Final[str] = "cosine_similarity"

_ID_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"\A[A-Za-z0-9][A-Za-z0-9_.-]{0,63}\Z", re.ASCII
)
_HEX64_PATTERN: Final[re.Pattern[str]] = re.compile(r"\A[0-9a-f]{64}\Z", re.ASCII)
_CODE_PATTERN: Final[re.Pattern[str]] = re.compile(r"\A[A-Z0-9_]{1,64}\Z", re.ASCII)
_DATASET_DOMAIN_TAG: Final[bytes] = b"local-semantic-atlas:dataset:v1\x00"
_RECORD_DOMAIN_TAG: Final[bytes] = b"doc\x00"


class Status(str, Enum):
    """Closed operational status vocabulary."""

    COMPLETE = "COMPLETE"
    COMPLETE_EMPTY = "COMPLETE_EMPTY"
    INVALID_INPUT = "INVALID_INPUT"
    INVALID_ARTIFACT = "INVALID_ARTIFACT"
    INCOMPLETE = "INCOMPLETE"
    NUMERICALLY_INDETERMINATE = "NUMERICALLY_INDETERMINATE"


SUCCESS_STATUSES: Final[frozenset[Status]] = frozenset(
    {Status.COMPLETE, Status.COMPLETE_EMPTY}
)


@dataclass(frozen=True, slots=True)
class Document:
    """Immutable input document record."""

    id: str
    text: str


@dataclass(frozen=True, slots=True)
class WorkBudget:
    """Explicit bounded work budget for indexing and search."""

    max_documents: int = MAX_DOCUMENTS
    max_tokens: int = MAX_TOKENS


@dataclass(frozen=True, slots=True)
class Result:
    """Uniform outcome wrapper. Values exist only for successful operations."""

    status: Status
    code: str
    value: object | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.status, Status):
            raise TypeError("status must be a Status enum member")
        if type(self.code) is not str or not _CODE_PATTERN.fullmatch(self.code):
            raise ValueError("code must be a non-empty uppercase ASCII token")
        if self.status in SUCCESS_STATUSES:
            if self.value is None:
                raise ValueError("successful Result requires a non-None value")
        else:
            if self.value is not None:
                raise ValueError("non-successful Result cannot carry a value")


@dataclass(frozen=True, slots=True)
class SearchHit:
    """Immutable similarity hit containing only document ID and finite cosine score."""

    id: str
    score: float


@dataclass(frozen=True, slots=True)
class SearchReport:
    """Immutable search report containing provenance, counts, and ranked hits only."""

    status: Status
    code: str
    model_id: str
    model_revision: str
    model_sha256: str
    tokenizer_sha256: str
    dataset_digest: str
    corpus_count: int
    document_token_count: int
    query_token_count: int
    top_k: int
    score_unit: str
    hits: tuple[SearchHit, ...]

    @property
    def document_total_token_count(self) -> int:
        return self.document_token_count

    @property
    def score_units(self) -> str:
        return self.score_unit


def is_true_int(value: object) -> bool:
    """Return True iff value is an int and not a bool."""
    return isinstance(value, int) and not isinstance(value, bool)


def validate_budget(budget: object) -> Result | None:
    """Validate a WorkBudget instance; return None if valid or an INVALID_INPUT Result."""
    if not isinstance(budget, WorkBudget):
        return Result(Status.INVALID_INPUT, "INVALID_BUDGET", None)
    if not is_true_int(budget.max_documents) or not (
        0 <= budget.max_documents <= MAX_DOCUMENTS
    ):
        return Result(Status.INVALID_INPUT, "INVALID_BUDGET", None)
    if not is_true_int(budget.max_tokens) or not (
        0 <= budget.max_tokens <= MAX_TOKENS
    ):
        return Result(Status.INVALID_INPUT, "INVALID_BUDGET", None)
    return None


def validate_document_id(doc_id: object) -> bytes | None:
    """Validate an ASCII document ID and return its ASCII bytes or None."""
    if type(doc_id) is not str:
        return None
    if not _ID_PATTERN.fullmatch(doc_id):
        return None
    return doc_id.encode("ascii")


def validate_utf8_text(text: object) -> bytes | None:
    """Validate non-empty strict UTF-8 text without NUL/surrogates <= 8192 bytes."""
    if type(text) is not str or len(text) == 0:
        return None
    if "\x00" in text:
        return None
    try:
        encoded = text.encode("utf-8", "strict")
    except UnicodeEncodeError:
        return None
    if not (1 <= len(encoded) <= MAX_TEXT_BYTES):
        return None
    return encoded


def compute_dataset_digest(sorted_encoded_docs: list[tuple[bytes, bytes]]) -> str:
    """Compute domain-separated, length-framed SHA-256 hex digest over sorted docs."""
    hasher = hashlib.sha256()
    hasher.update(_DATASET_DOMAIN_TAG)
    hasher.update(len(sorted_encoded_docs).to_bytes(8, "big", signed=False))
    for id_bytes, text_bytes in sorted_encoded_docs:
        hasher.update(_RECORD_DOMAIN_TAG)
        hasher.update(len(id_bytes).to_bytes(8, "big", signed=False))
        hasher.update(id_bytes)
        hasher.update(len(text_bytes).to_bytes(8, "big", signed=False))
        hasher.update(text_bytes)
    return hasher.hexdigest()


def _validate_search_report_for_json(result: Result, report: object) -> bool:
    """Validate closed schema fields and types on a successful SearchReport."""
    if type(report) is not SearchReport:
        return False
    if report.status != result.status or report.code != result.code:
        return False
    if (
        type(report.model_id) is not str
        or report.model_id != MODEL_ID
        or type(report.model_revision) is not str
        or report.model_revision != MODEL_REVISION
        or type(report.model_sha256) is not str
        or report.model_sha256 != MODEL_SHA256
        or type(report.tokenizer_sha256) is not str
        or report.tokenizer_sha256 != TOKENIZER_SHA256
    ):
        return False
    if (
        type(report.dataset_digest) is not str
        or not _HEX64_PATTERN.fullmatch(report.dataset_digest)
    ):
        return False
    if not is_true_int(report.corpus_count) or not (
        0 <= report.corpus_count <= MAX_DOCUMENTS
    ):
        return False
    if not is_true_int(report.document_token_count) or not (
        0 <= report.document_token_count <= MAX_TOKENS
    ):
        return False
    if not is_true_int(report.query_token_count) or not (
        1 <= report.query_token_count <= MAX_SEQUENCE_TOKENS
    ):
        return False
    if not is_true_int(report.top_k) or not (1 <= report.top_k <= MAX_DOCUMENTS):
        return False
    if type(report.score_unit) is not str or report.score_unit != SCORE_UNIT_COSINE:
        return False
    if type(report.hits) is not tuple:
        return False
    if len(report.hits) != min(report.corpus_count, report.top_k):
        return False
    if report.corpus_count == 0:
        if result.status != Status.COMPLETE_EMPTY or report.document_token_count != 0:
            return False
    elif result.status != Status.COMPLETE or report.document_token_count < report.corpus_count:
        return False
    if result.status == Status.COMPLETE_EMPTY and (
        report.corpus_count != 0 or len(report.hits) != 0
    ):
        return False
    seen_ids: set[str] = set()
    previous_key = None
    for hit in report.hits:
        if type(hit) is not SearchHit:
            return False
        if validate_document_id(hit.id) is None:
            return False
        if type(hit.score) is not float or not math.isfinite(hit.score):
            return False
        if hit.score < -1.0 or hit.score > 1.0:
            return False
        if hit.id in seen_ids:
            return False
        key = (-hit.score, hit.id)
        if previous_key is not None and key < previous_key:
            return False
        seen_ids.add(hit.id)
        previous_key = key
    return True


def report_json(result: object) -> Result:
    """Serialize a Result into closed version 1 compact UTF-8 JSON bytes ending with LF."""
    if type(result) is not Result:
        return Result(Status.INVALID_INPUT, "INVALID_RESULT_OBJECT", None)

    if result.status not in SUCCESS_STATUSES:
        payload: dict[str, object] = {
            "schema_version": 1,
            "status": result.status.value,
            "code": result.code,
        }
    else:
        report = result.value
        if not _validate_search_report_for_json(result, report):
            return Result(Status.INVALID_INPUT, "INVALID_REPORT_PAYLOAD", None)
        assert isinstance(report, SearchReport)
        payload = {
            "schema_version": 1,
            "status": report.status.value,
            "code": report.code,
            "model": {
                "id": report.model_id,
                "revision": report.model_revision,
                "model_sha256": report.model_sha256,
                "tokenizer_sha256": report.tokenizer_sha256,
            },
            "dataset_digest": report.dataset_digest,
            "corpus_count": report.corpus_count,
            "document_token_count": report.document_token_count,
            "query_token_count": report.query_token_count,
            "top_k": report.top_k,
            "score_unit": report.score_unit,
            "hits": [
                {"id": hit.id, "score": 0.0 if hit.score == 0.0 else hit.score}
                for hit in report.hits
            ],
        }

    try:
        text = (
            json.dumps(
                payload,
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=False,
            )
            + "\n"
        )
        encoded = text.encode("utf-8", "strict")
    except (TypeError, ValueError, UnicodeEncodeError):
        return Result(Status.INVALID_INPUT, "JSON_SERIALIZATION_FAILED", None)

    if len(encoded) > MAX_REPORT_JSON_BYTES:
        return Result(Status.INCOMPLETE, "REPORT_SIZE_EXCEEDED", None)

    return Result(Status.COMPLETE, "OK", encoded)
