"""Unit tests for Local Semantic Atlas core numerics, admission, and reporting."""

from collections.abc import Sequence
import hashlib
import json
import math
import unittest
from unittest.mock import patch

import numpy as np

from local_semantic_atlas import (  # noqa: E402
    ApprovedEncoder,
    Document,
    EMBEDDING_DIM,
    Index,
    MAX_DOCUMENTS,
    MAX_REPORT_JSON_BYTES,
    MAX_SEQUENCE_TOKENS,
    MAX_TEXT_BYTES,
    MAX_TOTAL_DOCUMENT_BYTES,
    MODEL_BYTE_LENGTH,
    MODEL_ID,
    MODEL_REVISION,
    MODEL_SHA256,
    Result,
    SCORE_UNIT_COSINE,
    SearchHit,
    SearchReport,
    Status,
    TOKENIZER_BYTE_LENGTH,
    TOKENIZER_SHA256,
    WorkBudget,
    build_index,
    cosine_similarity,
    pool_and_normalize,
    rank_candidates,
    report_json,
    search,
)
from local_semantic_atlas.engine import (  # noqa: E402
    _CONSTRUCTION_GUARD,
    _ENCODER_REGISTRY,
    _EncoderBackend,
)


class _TrapSequence(Sequence[Document]):
    """Sequence whose length is inspectable but fails if iterated or indexed."""

    def __init__(self, reported_length: int) -> None:
        self._length = reported_length
        self.accessed = False

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, index: int) -> Document:
        self.accessed = True
        raise AssertionError("Sequence elements must not be accessed before len check")


class _FakeEncoding:
    def __init__(self, ids: list[int], attention_mask: list[int], type_ids: list[int]):
        self.ids = ids
        self.attention_mask = attention_mask
        self.type_ids = type_ids


class _FakeTokenizer:
    """Deterministic test double used only for admission/budget flow unit tests."""

    def __init__(self, token_count: int = 4) -> None:
        self.token_count = token_count

    def encode(self, text: str, add_special_tokens: bool = True) -> _FakeEncoding:
        if "LONG_TOKENS" in text:
            n = MAX_SEQUENCE_TOKENS + 1
        else:
            n = self.token_count
        return _FakeEncoding(
            ids=[101] + [42] * (n - 1),
            attention_mask=[1] * n,
            type_ids=[0] * n,
        )


class _FakeSession:
    """Deterministic test double used only for budget/ordering unit tests."""

    def __init__(self) -> None:
        self.invocations = 0

    def run(
        self, output_names: list[str], input_feed: dict[str, np.ndarray]
    ) -> list[np.ndarray]:
        self.invocations += 1
        seq_len = input_feed["input_ids"].shape[1]
        hidden = np.zeros((1, seq_len, EMBEDDING_DIM), dtype=np.float32)
        hidden[0, :, 0] = 3.0
        hidden[0, :, 1] = 4.0
        return [hidden]


def _make_stub_encoder(token_count: int = 4) -> tuple[ApprovedEncoder, _FakeSession]:
    encoder = ApprovedEncoder(_CONSTRUCTION_GUARD)
    session = _FakeSession()
    _ENCODER_REGISTRY[encoder] = _EncoderBackend(
        tokenizer=_FakeTokenizer(token_count=token_count),
        session=session,
    )
    return encoder, session


class TestMaskedPoolingAndNumerics(unittest.TestCase):
    """Tests for float64 masked mean pooling, normalization, and cosine ranking."""

    def test_masked_mean_pooling_matches_independent_fsum_reference(self) -> None:
        hidden = np.zeros((1, 3, EMBEDDING_DIM), dtype=np.float32)
        hidden[0, 0, 0] = 1.0
        hidden[0, 0, 1] = 2.0
        hidden[0, 1, 0] = 3.0
        hidden[0, 1, 1] = 4.0
        hidden[0, 2, 0] = 100.0
        hidden[0, 2, 1] = 200.0
        mask = [1, 1, 0]

        mean_0 = math.fsum([1.0, 3.0]) / 2.0
        mean_1 = math.fsum([2.0, 4.0]) / 2.0
        norm = math.sqrt(math.fsum([mean_0 * mean_0, mean_1 * mean_1]))
        expected = [mean_0 / norm, mean_1 / norm] + [0.0] * (EMBEDDING_DIM - 2)

        res = pool_and_normalize(hidden, mask)
        self.assertEqual(res.status, Status.COMPLETE)
        self.assertEqual(res.code, "OK")
        vec = res.value
        self.assertIsInstance(vec, np.ndarray)
        assert isinstance(vec, np.ndarray)
        self.assertEqual(vec.shape, (EMBEDDING_DIM,))
        self.assertEqual(vec.dtype, np.float64)

        for idx, exp_val in enumerate(expected):
            self.assertLessEqual(abs(float(vec[idx]) - exp_val), 1e-14)

    def test_pooling_rejects_all_padding_nonfinite_wrong_dim_and_zero_norm(self) -> None:
        valid_hidden = np.ones((1, 3, EMBEDDING_DIM), dtype=np.float32)

        all_pad = pool_and_normalize(valid_hidden, [0, 0, 0])
        self.assertEqual(all_pad.status, Status.NUMERICALLY_INDETERMINATE)
        self.assertIsNone(all_pad.value)

        bad_mask_val = pool_and_normalize(valid_hidden, [1, 2, 0])
        self.assertEqual(bad_mask_val.status, Status.NUMERICALLY_INDETERMINATE)
        self.assertIsNone(bad_mask_val.value)

        for nonfinite_val in (float("nan"), float("inf"), float("-inf")):
            corrupted = valid_hidden.copy()
            corrupted[0, 2, 0] = nonfinite_val
            res = pool_and_normalize(corrupted, [1, 1, 0])
            self.assertEqual(res.status, Status.NUMERICALLY_INDETERMINATE)
            self.assertIsNone(res.value)

        for wrong_shape in ((3, EMBEDDING_DIM), (2, 3, EMBEDDING_DIM), (1, 3, 383), (1, 0, 384)):
            bad_arr = np.ones(wrong_shape, dtype=np.float32)
            res = pool_and_normalize(bad_arr, [1, 1, 1])
            self.assertEqual(res.status, Status.NUMERICALLY_INDETERMINATE)
            self.assertIsNone(res.value)

        zero_hidden = np.zeros((1, 2, EMBEDDING_DIM), dtype=np.float64)
        zero_res = pool_and_normalize(zero_hidden, [1, 1])
        self.assertEqual(zero_res.status, Status.NUMERICALLY_INDETERMINATE)
        self.assertIsNone(zero_res.value)

        cancelling_hidden = np.zeros((1, 2, EMBEDDING_DIM), dtype=np.float64)
        cancelling_hidden[0, 0, 0] = 5.0
        cancelling_hidden[0, 1, 0] = -5.0
        cancel_res = pool_and_normalize(cancelling_hidden, [1, 1])
        self.assertEqual(cancel_res.status, Status.NUMERICALLY_INDETERMINATE)
        self.assertIsNone(cancel_res.value)

    def test_ranking_tie_breaking_and_cosine_limits(self) -> None:
        e0 = np.zeros(EMBEDDING_DIM, dtype=np.float64)
        e0[0] = 1.0
        e1 = np.zeros(EMBEDDING_DIM, dtype=np.float64)
        e1[1] = 1.0
        q = np.zeros(EMBEDDING_DIM, dtype=np.float64)
        inv_sqrt2 = 1.0 / math.sqrt(2.0)
        q[0] = inv_sqrt2
        q[1] = inv_sqrt2

        forward = rank_candidates(q, ["a", "b"], [e0, e1], top_k=2)
        reverse = rank_candidates(q, ["b", "a"], [e1, e0], top_k=2)
        self.assertEqual(forward.status, Status.COMPLETE)
        self.assertEqual(reverse.status, Status.COMPLETE)
        self.assertEqual(forward.value, reverse.value)
        hits = forward.value
        assert isinstance(hits, tuple)
        self.assertEqual([h.id for h in hits], ["a", "b"])
        self.assertLessEqual(abs(hits[0].score - inv_sqrt2), 1e-15)
        self.assertEqual(hits[0].score, hits[1].score)

        slight_over = e0.copy()
        slight_over[0] = 1.0 + 5e-13
        clamped_pos = cosine_similarity(e0, slight_over)
        self.assertEqual(clamped_pos.status, Status.COMPLETE)
        self.assertEqual(clamped_pos.value, 1.0)

        cl_neg_vec = -slight_over
        clamped_neg = cosine_similarity(e0, cl_neg_vec)
        self.assertEqual(clamped_neg.status, Status.COMPLETE)
        self.assertEqual(clamped_neg.value, -1.0)

        too_large = e0.copy()
        too_large[0] = 1.0 + 1e-9
        rejected_pos = cosine_similarity(e0, too_large)
        self.assertEqual(rejected_pos.status, Status.NUMERICALLY_INDETERMINATE)
        self.assertIsNone(rejected_pos.value)

        rejected_neg = cosine_similarity(e0, -too_large)
        self.assertEqual(rejected_neg.status, Status.NUMERICALLY_INDETERMINATE)
        self.assertIsNone(rejected_neg.value)

    def test_immutable_vector_and_frozen_dataclasses(self) -> None:
        hidden = np.ones((1, 2, EMBEDDING_DIM), dtype=np.float64)
        res = pool_and_normalize(hidden, [1, 1])
        self.assertEqual(res.status, Status.COMPLETE)
        vec = res.value
        assert isinstance(vec, np.ndarray)

        self.assertFalse(vec.flags.writeable)
        with self.assertRaises(ValueError):
            vec[0] = 42.0
        with self.assertRaises(ValueError):
            vec.setflags(write=True)

        base = vec.base
        self.assertIsInstance(base, bytes)
        re_viewed = np.frombuffer(base, dtype=np.float64)
        self.assertFalse(re_viewed.flags.writeable)
        with self.assertRaises(ValueError):
            re_viewed.setflags(write=True)
        with self.assertRaises((ValueError, TypeError, BufferError)):
            np.ndarray(shape=(EMBEDDING_DIM,), dtype=np.float64, buffer=base).setflags(
                write=True
            )

        doc = Document(id="doc1", text="hello")
        with self.assertRaises(AttributeError):
            doc.id = "doc2"  # type: ignore[misc]

        budget = WorkBudget(max_documents=10, max_tokens=100)
        with self.assertRaises(AttributeError):
            budget.max_documents = 20  # type: ignore[misc]

        with self.assertRaises(TypeError):
            ApprovedEncoder()
        with self.assertRaises(TypeError):
            Index(
                dataset_digest="0" * 64,
                document_ids=(),
                document_count=0,
                total_document_tokens=0,
            )


class TestArtifactValidation(unittest.TestCase):
    """Tests that artifact length and SHA-256 checks happen before native init."""

    def test_wrong_type_length_or_hash_fails_before_native_constructors(self) -> None:
        with (
            patch("local_semantic_atlas.engine.Tokenizer.from_str") as mock_tok,
            patch("local_semantic_atlas.engine.ort.InferenceSession") as mock_sess,
            patch("local_semantic_atlas.engine.ort.disable_telemetry_events") as mock_tel,
        ):
            r_type = ApprovedEncoder.from_snapshots(bytearray(b"abc"), b"def")
            self.assertEqual(r_type.status, Status.INVALID_ARTIFACT)
            self.assertIsNone(r_type.value)

            r_model_len = ApprovedEncoder.from_snapshots(b"short", b"x" * TOKENIZER_BYTE_LENGTH)
            self.assertEqual(r_model_len.status, Status.INVALID_ARTIFACT)
            self.assertEqual(r_model_len.code, "MODEL_LENGTH_MISMATCH")

            r_tok_len = ApprovedEncoder.from_snapshots(b"x" * MODEL_BYTE_LENGTH, b"short")
            self.assertEqual(r_tok_len.status, Status.INVALID_ARTIFACT)
            self.assertEqual(r_tok_len.code, "TOKENIZER_LENGTH_MISMATCH")

            fake_model = b"\x00" * MODEL_BYTE_LENGTH
            fake_tok = b"\x00" * TOKENIZER_BYTE_LENGTH
            r_hash = ApprovedEncoder.from_snapshots(fake_model, fake_tok)
            self.assertEqual(r_hash.status, Status.INVALID_ARTIFACT)
            self.assertEqual(r_hash.code, "MODEL_SHA256_MISMATCH")

            real_sha256 = hashlib.sha256

            def _selective_sha256(data: bytes = b"") -> object:
                if len(data) == MODEL_BYTE_LENGTH:
                    class _MatchedHash:
                        def hexdigest(self) -> str:
                            return MODEL_SHA256
                    return _MatchedHash()
                return real_sha256(data)

            with patch("local_semantic_atlas.engine.hashlib.sha256", side_effect=_selective_sha256):
                r_tok_hash = ApprovedEncoder.from_snapshots(fake_model, fake_tok)
                self.assertEqual(r_tok_hash.status, Status.INVALID_ARTIFACT)
                self.assertEqual(r_tok_hash.code, "TOKENIZER_SHA256_MISMATCH")

            mock_tok.assert_not_called()
            mock_sess.assert_not_called()
            mock_tel.assert_not_called()


class TestAdmissionAndBudgets(unittest.TestCase):
    """Tests for document/query/budget admission and fail-before-copy guarantees."""

    def test_sequence_oversize_fails_before_copy_or_iteration(self) -> None:
        encoder, session = _make_stub_encoder()

        trap_hard_limit = _TrapSequence(MAX_DOCUMENTS + 1)
        res_hard = build_index(trap_hard_limit, encoder)
        self.assertEqual(res_hard.status, Status.INVALID_INPUT)
        self.assertEqual(res_hard.code, "DOCUMENT_LIMIT_EXCEEDED")
        self.assertFalse(trap_hard_limit.accessed)

        trap_budget = _TrapSequence(10)
        res_budget = build_index(trap_budget, encoder, WorkBudget(max_documents=5))
        self.assertEqual(res_budget.status, Status.INCOMPLETE)
        self.assertEqual(res_budget.code, "DOCUMENT_BUDGET_EXCEEDED")
        self.assertFalse(trap_budget.accessed)
        self.assertEqual(session.invocations, 0)

        gen = (Document(id=f"d{i}", text="hi") for i in range(2))
        res_gen = build_index(gen, encoder)  # type: ignore[arg-type]
        self.assertEqual(res_gen.status, Status.INVALID_INPUT)

    def test_id_utf8_duplicates_and_total_bytes_validation(self) -> None:
        encoder, session = _make_stub_encoder()

        for bad_id in ("", "_bad", "-bad", ".bad", "a" * 65, "bad id", "bad\n", "café"):
            res = build_index([Document(id=bad_id, text="ok")], encoder)
            self.assertEqual(res.status, Status.INVALID_INPUT)
            self.assertEqual(res.code, "INVALID_DOCUMENT_ID")

        max_id = "A" + "a_.-0" * 12 + "z" * 3
        self.assertEqual(len(max_id), 64)
        res_valid_id = build_index([Document(id=max_id, text="valid")], encoder)
        self.assertEqual(res_valid_id.status, Status.COMPLETE)

        res_dup = build_index(
            [Document(id="dup", text="one"), Document(id="dup", text="two")], encoder
        )
        self.assertEqual(res_dup.status, Status.INVALID_INPUT)
        self.assertEqual(res_dup.code, "DUPLICATE_DOCUMENT_ID")

        for bad_text in ("", "nul\x00byte", "surrogate\ud800char", "a" * (MAX_TEXT_BYTES + 1)):
            res_txt = build_index([Document(id="d1", text=bad_text)], encoder)
            self.assertEqual(res_txt.status, Status.INVALID_INPUT)
            self.assertEqual(res_txt.code, "INVALID_DOCUMENT_TEXT")

        exact_8k = "a" * MAX_TEXT_BYTES
        docs_1mib = [Document(id=f"d{i:03d}", text=exact_8k) for i in range(128)]
        self.assertEqual(128 * MAX_TEXT_BYTES, MAX_TOTAL_DOCUMENT_BYTES)
        res_exact_total = build_index(docs_1mib, encoder)
        self.assertEqual(res_exact_total.status, Status.COMPLETE)

        docs_over_1mib = docs_1mib + [Document(id="d128", text="x")]
        before_calls = session.invocations
        res_over_total = build_index(docs_over_1mib, encoder)
        self.assertEqual(res_over_total.status, Status.INVALID_INPUT)
        self.assertEqual(res_over_total.code, "TOTAL_BYTES_LIMIT_EXCEEDED")
        self.assertEqual(session.invocations, before_calls)

    def test_budget_validation_and_token_limits_before_inference(self) -> None:
        encoder, session = _make_stub_encoder(token_count=10)
        doc = Document(id="d1", text="hello")

        for bad_budget in (
            WorkBudget(max_documents=True, max_tokens=100),  # type: ignore[arg-type]
            WorkBudget(max_documents=10, max_tokens=False),  # type: ignore[arg-type]
            WorkBudget(max_documents=-1, max_tokens=100),
            WorkBudget(max_documents=257, max_tokens=100),
            WorkBudget(max_documents=10, max_tokens=-1),
            WorkBudget(max_documents=10, max_tokens=65537),
        ):
            res = build_index([doc], encoder, bad_budget)
            self.assertEqual(res.status, Status.INVALID_INPUT)
            self.assertEqual(res.code, "INVALID_BUDGET")

        calls_before = session.invocations
        res_tok_limit = build_index([Document(id="d1", text="LONG_TOKENS")], encoder)
        self.assertEqual(res_tok_limit.status, Status.INVALID_INPUT)
        self.assertEqual(res_tok_limit.code, "TOKEN_LIMIT")
        self.assertEqual(session.invocations, calls_before)

        res_tok_budget = build_index(
            [Document(id="d1", text="ok"), Document(id="d2", text="ok")],
            encoder,
            WorkBudget(max_documents=2, max_tokens=15),
        )
        self.assertEqual(res_tok_budget.status, Status.INCOMPLETE)
        self.assertEqual(res_tok_budget.code, "TOKEN_BUDGET_EXCEEDED")
        self.assertIsNone(res_tok_budget.value)

    def test_empty_corpus_and_search_budget_contracts(self) -> None:
        encoder, session = _make_stub_encoder(token_count=6)
        empty_idx_res = build_index([], encoder, WorkBudget(max_documents=0, max_tokens=0))
        self.assertEqual(empty_idx_res.status, Status.COMPLETE)
        self.assertEqual(session.invocations, 0)
        empty_index = empty_idx_res.value
        assert isinstance(empty_index, Index)
        self.assertEqual(empty_index.document_count, 0)

        bad_q = search(empty_index, "", top_k=5)
        self.assertEqual(bad_q.status, Status.INVALID_INPUT)

        bad_k = search(empty_index, "ola", top_k=True)  # type: ignore[arg-type]
        self.assertEqual(bad_k.status, Status.INVALID_INPUT)

        no_tok = search(empty_index, "ola", top_k=5, budget=WorkBudget(max_tokens=5))
        self.assertEqual(no_tok.status, Status.INCOMPLETE)
        self.assertEqual(no_tok.code, "TOKEN_BUDGET_EXCEEDED")
        self.assertEqual(session.invocations, 0)

        ok_empty = search(empty_index, "ola mundo", top_k=5, budget=WorkBudget(max_tokens=6))
        self.assertEqual(ok_empty.status, Status.COMPLETE_EMPTY)
        self.assertEqual(ok_empty.code, "EMPTY_CORPUS")
        self.assertEqual(session.invocations, 1)


class TestReportJson(unittest.TestCase):
    """Tests for closed version 1 JSON report serialization and privacy guarantees."""

    def test_report_json_closed_schema_privacy_and_utf8_bytes(self) -> None:
        report = SearchReport(
            status=Status.COMPLETE,
            code="OK",
            model_id=MODEL_ID,
            model_revision=MODEL_REVISION,
            model_sha256=MODEL_SHA256,
            tokenizer_sha256=TOKENIZER_SHA256,
            dataset_digest="a" * 64,
            corpus_count=2,
            document_token_count=14,
            query_token_count=5,
            top_k=5,
            score_unit=SCORE_UNIT_COSINE,
            hits=(SearchHit(id="doc-a", score=0.875), SearchHit(id="doc-b", score=-0.0)),
        )
        res = report_json(Result(Status.COMPLETE, "OK", report))
        self.assertEqual(res.status, Status.COMPLETE)
        raw = res.value
        self.assertIsInstance(raw, bytes)
        assert isinstance(raw, bytes)
        self.assertTrue(raw.endswith(b"\n"))
        self.assertLessEqual(len(raw), MAX_REPORT_JSON_BYTES)

        parsed = json.loads(raw.decode("utf-8", "strict"))
        self.assertEqual(
            list(parsed.keys()),
            [
                "schema_version",
                "status",
                "code",
                "model",
                "dataset_digest",
                "corpus_count",
                "document_token_count",
                "query_token_count",
                "top_k",
                "score_unit",
                "hits",
            ],
        )
        self.assertEqual(parsed["schema_version"], 1)
        self.assertEqual(parsed["hits"][1]["score"], 0.0)
        self.assertNotIn(b"-0.0", raw)

        err_res = report_json(Result(Status.INVALID_INPUT, "INVALID_QUERY", None))
        self.assertEqual(err_res.status, Status.COMPLETE)
        assert isinstance(err_res.value, bytes)
        err_parsed = json.loads(err_res.value.decode("utf-8", "strict"))
        self.assertEqual(
            err_parsed,
            {"schema_version": 1, "status": "INVALID_INPUT", "code": "INVALID_QUERY"},
        )

        bad_obj_res = report_json(Result(Status.COMPLETE, "OK", object()))
        self.assertEqual(bad_obj_res.status, Status.INVALID_INPUT)
        self.assertIsNone(bad_obj_res.value)

        many_hits = tuple(
            SearchHit(id=f"doc-{i:05d}-" + "x" * 50, score=0.5) for i in range(3600)
        )
        oversize_report = SearchReport(
            status=Status.COMPLETE,
            code="OK",
            model_id=MODEL_ID,
            model_revision=MODEL_REVISION,
            model_sha256=MODEL_SHA256,
            tokenizer_sha256=TOKENIZER_SHA256,
            dataset_digest="b" * 64,
            corpus_count=256,
            document_token_count=1024,
            query_token_count=4,
            top_k=256,
            score_unit=SCORE_UNIT_COSINE,
            hits=many_hits,
        )
        over_res = report_json(Result(Status.COMPLETE, "OK", oversize_report))
        # A report with3600hits violates the closed256candidate schema before
        # serialization. The independent public-boundary suite separately
        # exercises exact byte staging on a valid complete report.
        self.assertEqual(over_res.status, Status.INVALID_INPUT)
        self.assertEqual(over_res.code, "INVALID_REPORT_PAYLOAD")
        self.assertIsNone(over_res.value)


if __name__ == "__main__":
    unittest.main()
