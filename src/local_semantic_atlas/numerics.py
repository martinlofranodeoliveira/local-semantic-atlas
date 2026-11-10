"""Float64 masked mean pooling, L2 normalization, and non-BLAS cosine ranking."""

from collections.abc import Sequence
import math
import numpy as np

from .types import (
    COSINE_OVERSHOOT_TOLERANCE,
    EMBEDDING_DIM,
    MAX_DOCUMENTS,
    MAX_SEQUENCE_TOKENS,
    Result,
    SearchHit,
    Status,
    is_true_int,
    validate_document_id,
)


def make_immutable_vector(vector: np.ndarray) -> np.ndarray:
    """Return a 1D float64 ndarray backed by an immutable Python bytes buffer."""
    if not isinstance(vector, np.ndarray) or vector.shape != (EMBEDDING_DIM,):
        raise ValueError("vector must have shape (384,)")
    arr = np.asarray(vector, dtype=np.float64)
    raw_bytes = arr.tobytes()
    view = np.frombuffer(raw_bytes, dtype=np.float64)
    view.setflags(write=False)
    return view


def make_immutable_matrix(matrix: np.ndarray) -> np.ndarray:
    """Return a 2D float64 ndarray backed by an immutable Python bytes buffer."""
    if (not isinstance(matrix, np.ndarray) or matrix.ndim != 2
            or matrix.shape[1] != EMBEDDING_DIM or matrix.shape[0] > MAX_DOCUMENTS):
        raise ValueError("matrix must have shape (N, 384), N <= 256")
    arr = np.asarray(matrix, dtype=np.float64)
    rows = arr.shape[0]
    raw_bytes = arr.tobytes()
    view = np.frombuffer(raw_bytes, dtype=np.float64).reshape((rows, EMBEDDING_DIM))
    view.setflags(write=False)
    return view


def _extract_valid_mask(attention_mask: object, seq_len: int) -> np.ndarray | None:
    """Validate attention mask of shape (L,) or (1, L) with values strictly in {0, 1}."""
    try:
        mask_arr = np.asarray(attention_mask)
    except (TypeError, ValueError):
        return None
    if mask_arr.dtype == np.bool_ or mask_arr.dtype.kind not in ("i", "u"):
        return None
    if mask_arr.ndim == 2:
        if mask_arr.shape != (1, seq_len):
            return None
        mask_1d = mask_arr[0]
    elif mask_arr.ndim == 1:
        if mask_arr.shape != (seq_len,):
            return None
        mask_1d = mask_arr
    else:
        return None
    if not bool(np.all((mask_1d == 0) | (mask_1d == 1))):
        return None
    return mask_1d.astype(np.int64, copy=False)


def pool_and_normalize(hidden_state: object, attention_mask: object) -> Result:
    """Pool [1, L, 384] hidden states by masked float64 mean and L2-normalize."""
    if not isinstance(hidden_state, np.ndarray):
        try:
            hidden_arr = np.asarray(hidden_state)
        except (TypeError, ValueError):
            return Result(
                Status.NUMERICALLY_INDETERMINATE, "INVALID_HIDDEN_SHAPE", None
            )
    else:
        hidden_arr = hidden_state

    if (
        hidden_arr.dtype == np.bool_
        or hidden_arr.dtype.kind not in ("f", "i", "u")
        or hidden_arr.ndim != 3
        or hidden_arr.shape[0] != 1
        or hidden_arr.shape[1] < 1
        or hidden_arr.shape[1] > MAX_SEQUENCE_TOKENS
        or hidden_arr.shape[2] != EMBEDDING_DIM
    ):
        return Result(Status.NUMERICALLY_INDETERMINATE, "INVALID_HIDDEN_SHAPE", None)

    seq_len = int(hidden_arr.shape[1])
    mask_1d = _extract_valid_mask(attention_mask, seq_len)
    if mask_1d is None:
        return Result(
            Status.NUMERICALLY_INDETERMINATE, "INVALID_ATTENTION_MASK", None
        )

    active_tokens = int(np.sum(mask_1d, dtype=np.int64))
    if active_tokens <= 0:
        return Result(Status.NUMERICALLY_INDETERMINATE, "ALL_PADDING_MASK", None)

    hidden_f64 = hidden_arr.astype(np.float64, copy=False)[0]
    if not bool(np.all(np.isfinite(hidden_f64))):
        return Result(Status.NUMERICALLY_INDETERMINATE, "NONFINITE_HIDDEN_STATE", None)

    mask_f64 = mask_1d.astype(np.float64).reshape((seq_len, 1))
    denom = float(active_tokens)
    masked_values = np.multiply(hidden_f64, mask_f64, dtype=np.float64)
    summed = np.sum(masked_values, axis=0, dtype=np.float64)
    pooled = np.divide(summed, denom, dtype=np.float64)
    if pooled.shape != (EMBEDDING_DIM,) or not bool(np.all(np.isfinite(pooled))):
        return Result(
            Status.NUMERICALLY_INDETERMINATE, "NONFINITE_POOLED_VECTOR", None
        )

    sq_sum = float(
        np.sum(np.multiply(pooled, pooled, dtype=np.float64), dtype=np.float64)
    )
    if not math.isfinite(sq_sum) or sq_sum <= 0.0:
        return Result(Status.NUMERICALLY_INDETERMINATE, "ZERO_OR_NONFINITE_NORM", None)

    norm = float(np.sqrt(np.float64(sq_sum)))
    if not math.isfinite(norm) or norm <= 0.0:
        return Result(Status.NUMERICALLY_INDETERMINATE, "ZERO_OR_NONFINITE_NORM", None)

    unit_vec = np.divide(pooled, norm, dtype=np.float64)
    if not bool(np.all(np.isfinite(unit_vec))):
        return Result(Status.NUMERICALLY_INDETERMINATE, "NONFINITE_UNIT_VECTOR", None)

    unit_sq_sum = float(
        np.sum(np.multiply(unit_vec, unit_vec, dtype=np.float64), dtype=np.float64)
    )
    if not math.isfinite(unit_sq_sum) or abs(unit_sq_sum - 1.0) > 1e-6:
        return Result(Status.NUMERICALLY_INDETERMINATE, "ZERO_OR_NONFINITE_NORM", None)

    return Result(Status.COMPLETE, "OK", make_immutable_vector(unit_vec))


def _validate_vector_1d(vec: object) -> np.ndarray | None:
    """Validate a finite unit vector of length384 within float64 tolerance."""
    if not isinstance(vec, np.ndarray):
        try:
            arr = np.asarray(vec)
        except (TypeError, ValueError):
            return None
    else:
        arr = vec
    if (
        arr.dtype == np.bool_
        or arr.dtype.kind not in ("f", "i", "u")
        or arr.ndim != 1
        or arr.shape != (EMBEDDING_DIM,)
    ):
        return None
    arr_f64 = arr.astype(np.float64, copy=False)
    if not bool(np.all(np.isfinite(arr_f64))):
        return None
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        norm_squared = float(np.sum(np.multiply(arr_f64, arr_f64, dtype=np.float64), dtype=np.float64))
    if not math.isfinite(norm_squared) or abs(norm_squared - 1.0) > 4e-12:
        return None
    return arr_f64


def cosine_similarity(vec_a: object, vec_b: object) -> Result:
    """Compute cosine of admitted unit vectors; reject non-unit or zero input."""
    a = _validate_vector_1d(vec_a)
    b = _validate_vector_1d(vec_b)
    if a is None or b is None:
        return Result(Status.NUMERICALLY_INDETERMINATE, "INVALID_VECTOR", None)

    prod = np.multiply(a, b, dtype=np.float64)
    score = float(np.sum(prod, dtype=np.float64))
    if not math.isfinite(score):
        return Result(Status.NUMERICALLY_INDETERMINATE, "NONFINITE_COSINE_SCORE", None)

    if score > 1.0:
        if score - 1.0 <= COSINE_OVERSHOOT_TOLERANCE:
            score = 1.0
        else:
            return Result(
                Status.NUMERICALLY_INDETERMINATE, "COSINE_OUT_OF_BOUNDS", None
            )
    elif score < -1.0:
        if -1.0 - score <= COSINE_OVERSHOOT_TOLERANCE:
            score = -1.0
        else:
            return Result(
                Status.NUMERICALLY_INDETERMINATE, "COSINE_OUT_OF_BOUNDS", None
            )

    if score == 0.0:
        score = 0.0
    return Result(Status.COMPLETE, "OK", score)


def rank_candidates(
    query_vec: object,
    doc_ids: Sequence[str],
    doc_vectors: Sequence[np.ndarray] | np.ndarray,
    top_k: int,
) -> Result:
    """Score and rank candidate documents by (-score, ascii_id) up to top_k."""
    if not is_true_int(top_k) or not (1 <= top_k <= MAX_DOCUMENTS):
        return Result(Status.INVALID_INPUT, "INVALID_TOP_K", None)

    q = _validate_vector_1d(query_vec)
    if q is None:
        return Result(Status.NUMERICALLY_INDETERMINATE, "INVALID_QUERY_VECTOR", None)

    if isinstance(doc_ids, (str, bytes, bytearray, memoryview)) or not isinstance(
        doc_ids, Sequence
    ):
        return Result(Status.INVALID_INPUT, "INVALID_CANDIDATE_IDS", None)

    n_docs = len(doc_ids)
    if n_docs > MAX_DOCUMENTS:
        return Result(Status.INVALID_INPUT, "CANDIDATE_LIMIT_EXCEEDED", None)
    if isinstance(doc_vectors, np.ndarray):
        if doc_vectors.ndim != 2 or doc_vectors.shape != (n_docs, EMBEDDING_DIM):
            return Result(
                Status.NUMERICALLY_INDETERMINATE, "INVALID_CANDIDATE_VECTORS", None
            )
        vectors_seq: Sequence[np.ndarray] | np.ndarray = doc_vectors
    elif isinstance(doc_vectors, Sequence) and not isinstance(
        doc_vectors, (str, bytes, bytearray, memoryview)
    ):
        if len(doc_vectors) != n_docs:
            return Result(
                Status.NUMERICALLY_INDETERMINATE, "INVALID_CANDIDATE_VECTORS", None
            )
        vectors_seq = doc_vectors
    else:
        return Result(
            Status.NUMERICALLY_INDETERMINATE, "INVALID_CANDIDATE_VECTORS", None
        )

    seen_ids: set[str] = set()
    scored: list[tuple[str, float]] = []
    for idx in range(n_docs):
        doc_id = doc_ids[idx]
        if validate_document_id(doc_id) is None or doc_id in seen_ids:
            return Result(Status.INVALID_INPUT, "INVALID_CANDIDATE_IDS", None)
        seen_ids.add(doc_id)
        sim_res = cosine_similarity(q, vectors_seq[idx])
        if sim_res.status != Status.COMPLETE:
            return sim_res
        assert isinstance(sim_res.value, float)
        scored.append((doc_id, sim_res.value))

    scored.sort(key=lambda item: (-item[1], item[0]))
    hits = tuple(SearchHit(id=doc_id, score=score) for doc_id, score in scored[:top_k])
    return Result(Status.COMPLETE, "OK", hits)
