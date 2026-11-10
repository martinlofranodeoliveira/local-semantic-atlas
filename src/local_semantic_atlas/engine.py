"""ApprovedEncoder snapshot verification, Index construction, and cosine search."""

from collections.abc import Sequence
from dataclasses import dataclass
import hashlib
from typing import Final
from weakref import WeakKeyDictionary

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

from .numerics import (
    make_immutable_matrix,
    pool_and_normalize,
    rank_candidates,
)
from .types import (
    EMBEDDING_DIM,
    MAX_DOCUMENTS,
    MAX_SEQUENCE_TOKENS,
    MAX_TOTAL_DOCUMENT_BYTES,
    MODEL_BYTE_LENGTH,
    MODEL_ID,
    MODEL_REVISION,
    MODEL_SHA256,
    SCORE_UNIT_COSINE,
    TOKENIZER_BYTE_LENGTH,
    TOKENIZER_SHA256,
    Document,
    Result,
    SearchReport,
    Status,
    WorkBudget,
    compute_dataset_digest,
    is_true_int,
    validate_budget,
    validate_document_id,
    validate_utf8_text,
)

_CONSTRUCTION_GUARD: Final[object] = object()
_DOCUMENT_PREFIX: Final[str] = "passage: "
_QUERY_PREFIX: Final[str] = "query: "
_REQUIRED_INPUT_NAMES: Final[tuple[str, str, str]] = (
    "input_ids",
    "attention_mask",
    "token_type_ids",
)
_OUTPUT_NAME: Final[str] = "last_hidden_state"


@dataclass(frozen=True, slots=True)
class _EncoderBackend:
    tokenizer: Tokenizer
    session: ort.InferenceSession


@dataclass(frozen=True, slots=True)
class _IndexBackend:
    encoder: "ApprovedEncoder"
    vectors: np.ndarray


_ENCODER_REGISTRY: WeakKeyDictionary["ApprovedEncoder", _EncoderBackend] = (
    WeakKeyDictionary()
)
_INDEX_REGISTRY: WeakKeyDictionary["Index", _IndexBackend] = WeakKeyDictionary()


def _verify_session_schema(session: ort.InferenceSession) -> bool:
    """Verify active CPUExecutionProvider and expected input/output tensor schema."""
    try:
        providers = session.get_providers()
        if not isinstance(providers, (list, tuple)) or list(providers) != [
            "CPUExecutionProvider"
        ]:
            return False

        inputs = session.get_inputs()
        if not isinstance(inputs, (list, tuple)) or len(inputs) != 3:
            return False
        input_map = {inp.name: inp for inp in inputs}
        if set(input_map.keys()) != set(_REQUIRED_INPUT_NAMES):
            return False

        for name in _REQUIRED_INPUT_NAMES:
            inp = input_map[name]
            if inp.type != "tensor(int64)":
                return False
            shape = inp.shape
            if not isinstance(shape, (list, tuple)) or len(shape) != 2:
                return False
            dim0, dim1 = shape[0], shape[1]
            if is_true_int(dim0) and dim0 != 1:
                return False
            if is_true_int(dim1):
                return False

        outputs = session.get_outputs()
        if not isinstance(outputs, (list, tuple)) or len(outputs) < 1:
            return False
        output_map = {out.name: out for out in outputs}
        if _OUTPUT_NAME not in output_map:
            return False
        out = output_map[_OUTPUT_NAME]
        if out.type not in (
            "tensor(float)",
            "tensor(float32)",
            "tensor(double)",
            "tensor(float64)",
        ):
            return False
        out_shape = out.shape
        if not isinstance(out_shape, (list, tuple)) or len(out_shape) != 3:
            return False
        if is_true_int(out_shape[0]) and out_shape[0] != 1:
            return False
        if is_true_int(out_shape[1]):
            return False
        if not is_true_int(out_shape[2]) or out_shape[2] != EMBEDDING_DIM:
            return False
    except (RuntimeError, ValueError, TypeError, AttributeError, Exception):
        return False
    return True


@dataclass(frozen=True, eq=False, slots=True, weakref_slot=True)
class ApprovedEncoder:
    """Verified immutable handle to the fixed local ONNX encoder and tokenizer."""

    model_id: str = MODEL_ID
    model_revision: str = MODEL_REVISION
    model_sha256: str = MODEL_SHA256
    tokenizer_sha256: str = TOKENIZER_SHA256
    embedding_dim: int = EMBEDDING_DIM

    def __init__(self, _guard: object = None) -> None:
        if _guard is not _CONSTRUCTION_GUARD:
            raise TypeError(
                "ApprovedEncoder must be created via ApprovedEncoder.from_snapshots"
            )
        object.__setattr__(self, "model_id", MODEL_ID)
        object.__setattr__(self, "model_revision", MODEL_REVISION)
        object.__setattr__(self, "model_sha256", MODEL_SHA256)
        object.__setattr__(self, "tokenizer_sha256", TOKENIZER_SHA256)
        object.__setattr__(self, "embedding_dim", EMBEDDING_DIM)

    @classmethod
    def from_snapshots(cls, model_bytes: object, tokenizer_bytes: object) -> Result:
        """Validate owned artifact byte lengths and SHA-256 digests before init."""
        if type(model_bytes) is not bytes or type(tokenizer_bytes) is not bytes:
            return Result(Status.INVALID_ARTIFACT, "ARTIFACT_NOT_BYTES", None)
        if len(model_bytes) != MODEL_BYTE_LENGTH:
            return Result(Status.INVALID_ARTIFACT, "MODEL_LENGTH_MISMATCH", None)
        if len(tokenizer_bytes) != TOKENIZER_BYTE_LENGTH:
            return Result(Status.INVALID_ARTIFACT, "TOKENIZER_LENGTH_MISMATCH", None)

        model_digest = hashlib.sha256(model_bytes).hexdigest()
        if model_digest != MODEL_SHA256:
            return Result(Status.INVALID_ARTIFACT, "MODEL_SHA256_MISMATCH", None)

        tokenizer_digest = hashlib.sha256(tokenizer_bytes).hexdigest()
        if tokenizer_digest != TOKENIZER_SHA256:
            return Result(Status.INVALID_ARTIFACT, "TOKENIZER_SHA256_MISMATCH", None)

        try:
            tokenizer_str = tokenizer_bytes.decode("utf-8", "strict")
        except UnicodeDecodeError:
            return Result(Status.INVALID_ARTIFACT, "TOKENIZER_UTF8_INVALID", None)

        try:
            tokenizer = Tokenizer.from_str(tokenizer_str)
            tokenizer.no_truncation()
            tokenizer.no_padding()
        except (RuntimeError, ValueError, TypeError, Exception):
            return Result(Status.INCOMPLETE, "TOKENIZER_INIT_FAILED", None)

        try:
            ort.disable_telemetry_events()
            sess_options = ort.SessionOptions()
            sess_options.intra_op_num_threads = 1
            sess_options.inter_op_num_threads = 1
            sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            sess_options.graph_optimization_level = (
                ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
            )
            sess_options.enable_mem_pattern = False
            session = ort.InferenceSession(
                model_bytes,
                sess_options=sess_options,
                providers=["CPUExecutionProvider"],
            )
        except (RuntimeError, ValueError, TypeError, OSError, MemoryError, Exception):
            return Result(Status.INCOMPLETE, "SESSION_INIT_FAILED", None)

        if not _verify_session_schema(session):
            return Result(Status.INCOMPLETE, "SESSION_SCHEMA_INVALID", None)

        instance = cls(_CONSTRUCTION_GUARD)
        _ENCODER_REGISTRY[instance] = _EncoderBackend(
            tokenizer=tokenizer, session=session
        )
        return Result(Status.COMPLETE, "OK", instance)


@dataclass(frozen=True, eq=False, slots=True, weakref_slot=True)
class Index:
    """Immutable semantic index exposing metadata while keeping vectors/encoder private."""

    dataset_digest: str
    document_ids: tuple[str, ...]
    document_count: int
    total_document_tokens: int
    embedding_dim: int
    model_id: str
    model_revision: str
    model_sha256: str
    tokenizer_sha256: str

    def __init__(
        self,
        *,
        _guard: object = None,
        dataset_digest: str,
        document_ids: tuple[str, ...],
        document_count: int,
        total_document_tokens: int,
    ) -> None:
        if _guard is not _CONSTRUCTION_GUARD:
            raise TypeError("Index must be constructed via build_index")
        object.__setattr__(self, "dataset_digest", dataset_digest)
        object.__setattr__(self, "document_ids", document_ids)
        object.__setattr__(self, "document_count", document_count)
        object.__setattr__(self, "total_document_tokens", total_document_tokens)
        object.__setattr__(self, "embedding_dim", EMBEDDING_DIM)
        object.__setattr__(self, "model_id", MODEL_ID)
        object.__setattr__(self, "model_revision", MODEL_REVISION)
        object.__setattr__(self, "model_sha256", MODEL_SHA256)
        object.__setattr__(self, "tokenizer_sha256", TOKENIZER_SHA256)


def _encode_prefixed_text(
    backend: _EncoderBackend,
    prefixed_text: str,
    remaining_tokens: int,
) -> tuple[Result, int]:
    """Tokenize, enforce token limits/budget before inference, and embed a single text."""
    try:
        encoding = backend.tokenizer.encode(prefixed_text, add_special_tokens=True)
        ids = encoding.ids
        attention_mask = encoding.attention_mask
        type_ids = encoding.type_ids
    except (RuntimeError, ValueError, TypeError, MemoryError, Exception):
        return Result(Status.INCOMPLETE, "TOKENIZER_ENCODE_FAILED", None), 0

    if (
        not isinstance(ids, (list, tuple))
        or not isinstance(attention_mask, (list, tuple))
        or not isinstance(type_ids, (list, tuple))
    ):
        return Result(Status.INCOMPLETE, "TOKENIZER_OUTPUT_INVALID", None), 0

    token_count = len(ids)
    if (
        token_count <= 0
        or len(attention_mask) != token_count
        or len(type_ids) != token_count
    ):
        return Result(Status.INCOMPLETE, "TOKENIZER_OUTPUT_INVALID", None), 0

    if token_count > MAX_SEQUENCE_TOKENS:
        return Result(Status.INVALID_INPUT, "TOKEN_LIMIT", None), token_count

    if token_count > remaining_tokens:
        return Result(Status.INCOMPLETE, "TOKEN_BUDGET_EXCEEDED", None), token_count

    try:
        input_ids_arr = np.asarray(ids, dtype=np.int64).reshape((1, token_count))
        mask_arr = np.asarray(attention_mask, dtype=np.int64).reshape((1, token_count))
        type_ids_arr = np.asarray(type_ids, dtype=np.int64).reshape((1, token_count))
    except (TypeError, ValueError, OverflowError):
        return Result(Status.INCOMPLETE, "TOKENIZER_OUTPUT_INVALID", None), token_count

    if not bool(np.all((mask_arr == 0) | (mask_arr == 1))):
        return (
            Result(Status.NUMERICALLY_INDETERMINATE, "INVALID_ATTENTION_MASK", None),
            token_count,
        )
    if int(np.sum(mask_arr, dtype=np.int64)) <= 0:
        return (
            Result(Status.NUMERICALLY_INDETERMINATE, "ALL_PADDING_MASK", None),
            token_count,
        )

    try:
        outputs = backend.session.run(
            [_OUTPUT_NAME],
            {
                "input_ids": input_ids_arr,
                "attention_mask": mask_arr,
                "token_type_ids": type_ids_arr,
            },
        )
    except (RuntimeError, ValueError, TypeError, MemoryError, Exception):
        return (
            Result(Status.INCOMPLETE, "MODEL_INFERENCE_FAILED", None),
            token_count,
        )

    if (
        not isinstance(outputs, (list, tuple))
        or len(outputs) != 1
        or not isinstance(outputs[0], np.ndarray)
    ):
        return (
            Result(Status.NUMERICALLY_INDETERMINATE, "INVALID_HIDDEN_SHAPE", None),
            token_count,
        )

    hidden = outputs[0]
    if hidden.shape != (1, token_count, EMBEDDING_DIM):
        return (
            Result(Status.NUMERICALLY_INDETERMINATE, "INVALID_HIDDEN_SHAPE", None),
            token_count,
        )

    pool_res = pool_and_normalize(hidden, mask_arr)
    return pool_res, token_count


def build_index(
    documents: Sequence[Document],
    encoder: ApprovedEncoder,
    budget: WorkBudget = WorkBudget(),
) -> Result:
    """Admit documents, sort canonically by ASCII ID, embed, and publish an Index."""
    if type(encoder) is not ApprovedEncoder or encoder not in _ENCODER_REGISTRY:
        return Result(Status.INVALID_INPUT, "INVALID_ENCODER", None)
    backend = _ENCODER_REGISTRY[encoder]

    budget_err = validate_budget(budget)
    if budget_err is not None:
        return budget_err

    if isinstance(documents, (str, bytes, bytearray, memoryview)) or not isinstance(
        documents, Sequence
    ):
        return Result(Status.INVALID_INPUT, "INVALID_DOCUMENTS_SEQUENCE", None)

    doc_len = len(documents)
    if not is_true_int(doc_len) or doc_len < 0:
        return Result(Status.INVALID_INPUT, "INVALID_DOCUMENTS_SEQUENCE", None)
    if doc_len > MAX_DOCUMENTS:
        return Result(Status.INVALID_INPUT, "DOCUMENT_LIMIT_EXCEEDED", None)
    if doc_len > budget.max_documents:
        return Result(Status.INCOMPLETE, "DOCUMENT_BUDGET_EXCEEDED", None)

    docs_snapshot = tuple(documents[i] for i in range(doc_len))

    seen_ids: set[str] = set()
    admitted_records: list[tuple[str, str, bytes, bytes]] = []
    total_text_bytes = 0

    for record in docs_snapshot:
        if type(record) is not Document:
            return Result(Status.INVALID_INPUT, "INVALID_DOCUMENT_RECORD", None)
        id_bytes = validate_document_id(record.id)
        if id_bytes is None:
            return Result(Status.INVALID_INPUT, "INVALID_DOCUMENT_ID", None)
        if record.id in seen_ids:
            return Result(Status.INVALID_INPUT, "DUPLICATE_DOCUMENT_ID", None)
        seen_ids.add(record.id)

        text_bytes = validate_utf8_text(record.text)
        if text_bytes is None:
            return Result(Status.INVALID_INPUT, "INVALID_DOCUMENT_TEXT", None)
        total_text_bytes += len(text_bytes)
        if total_text_bytes > MAX_TOTAL_DOCUMENT_BYTES:
            return Result(Status.INVALID_INPUT, "TOTAL_BYTES_LIMIT_EXCEEDED", None)

        admitted_records.append((record.id, record.text, id_bytes, text_bytes))

    admitted_records.sort(key=lambda item: item[0])
    dataset_digest = compute_dataset_digest(
        [(item[2], item[3]) for item in admitted_records]
    )

    if len(admitted_records) == 0:
        empty_matrix = make_immutable_matrix(
            np.zeros((0, EMBEDDING_DIM), dtype=np.float64)
        )
        empty_index = Index(
            _guard=_CONSTRUCTION_GUARD,
            dataset_digest=dataset_digest,
            document_ids=(),
            document_count=0,
            total_document_tokens=0,
        )
        _INDEX_REGISTRY[empty_index] = _IndexBackend(
            encoder=encoder, vectors=empty_matrix
        )
        return Result(Status.COMPLETE, "OK", empty_index)

    remaining_tokens = budget.max_tokens
    total_tokens = 0
    vectors: list[np.ndarray] = []
    sorted_ids: list[str] = []

    for doc_id, doc_text, _, _ in admitted_records:
        prefixed = _DOCUMENT_PREFIX + doc_text
        embed_res, used_tokens = _encode_prefixed_text(
            backend, prefixed, remaining_tokens
        )
        if embed_res.status != Status.COMPLETE:
            return embed_res
        remaining_tokens -= used_tokens
        total_tokens += used_tokens
        assert isinstance(embed_res.value, np.ndarray)
        vectors.append(embed_res.value)
        sorted_ids.append(doc_id)

    matrix = make_immutable_matrix(np.vstack(vectors))
    built_index = Index(
        _guard=_CONSTRUCTION_GUARD,
        dataset_digest=dataset_digest,
        document_ids=tuple(sorted_ids),
        document_count=len(sorted_ids),
        total_document_tokens=total_tokens,
    )
    _INDEX_REGISTRY[built_index] = _IndexBackend(encoder=encoder, vectors=matrix)
    return Result(Status.COMPLETE, "OK", built_index)


def search(
    index: Index,
    query: str,
    top_k: int = 5,
    budget: WorkBudget = WorkBudget(),
) -> Result:
    """Embed query and rank indexed candidates by float64 cosine similarity."""
    if type(index) is not Index or index not in _INDEX_REGISTRY:
        return Result(Status.INVALID_INPUT, "INVALID_INDEX", None)
    index_backend = _INDEX_REGISTRY[index]
    encoder = index_backend.encoder
    if type(encoder) is not ApprovedEncoder or encoder not in _ENCODER_REGISTRY:
        return Result(Status.INVALID_INPUT, "INVALID_ENCODER", None)
    encoder_backend = _ENCODER_REGISTRY[encoder]

    if validate_utf8_text(query) is None:
        return Result(Status.INVALID_INPUT, "INVALID_QUERY", None)

    if not is_true_int(top_k) or not (1 <= top_k <= MAX_DOCUMENTS):
        return Result(Status.INVALID_INPUT, "INVALID_TOP_K", None)

    budget_err = validate_budget(budget)
    if budget_err is not None:
        return budget_err

    if index.document_count > budget.max_documents:
        return Result(Status.INCOMPLETE, "DOCUMENT_BUDGET_EXCEEDED", None)

    prefixed_query = _QUERY_PREFIX + query
    query_res, query_tokens = _encode_prefixed_text(
        encoder_backend, prefixed_query, budget.max_tokens
    )
    if query_res.status != Status.COMPLETE:
        return query_res
    assert isinstance(query_res.value, np.ndarray)

    if index.document_count == 0:
        empty_report = SearchReport(
            status=Status.COMPLETE_EMPTY,
            code="EMPTY_CORPUS",
            model_id=index.model_id,
            model_revision=index.model_revision,
            model_sha256=index.model_sha256,
            tokenizer_sha256=index.tokenizer_sha256,
            dataset_digest=index.dataset_digest,
            corpus_count=0,
            document_token_count=index.total_document_tokens,
            query_token_count=query_tokens,
            top_k=top_k,
            score_unit=SCORE_UNIT_COSINE,
            hits=(),
        )
        return Result(Status.COMPLETE_EMPTY, "EMPTY_CORPUS", empty_report)

    rank_res = rank_candidates(
        query_res.value, index.document_ids, index_backend.vectors, top_k
    )
    if rank_res.status != Status.COMPLETE:
        return rank_res
    assert isinstance(rank_res.value, tuple)

    report = SearchReport(
        status=Status.COMPLETE,
        code="OK",
        model_id=index.model_id,
        model_revision=index.model_revision,
        model_sha256=index.model_sha256,
        tokenizer_sha256=index.tokenizer_sha256,
        dataset_digest=index.dataset_digest,
        corpus_count=index.document_count,
        document_token_count=index.total_document_tokens,
        query_token_count=query_tokens,
        top_k=top_k,
        score_unit=SCORE_UNIT_COSINE,
        hits=rank_res.value,
    )
    return Result(Status.COMPLETE, "OK", report)
