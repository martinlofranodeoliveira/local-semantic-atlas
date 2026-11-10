"""Independent regressions for public numerical/report admission."""
from collections.abc import Sequence
from dataclasses import replace
import unittest
from unittest.mock import patch
import numpy as np

from local_semantic_atlas import (
    EMBEDDING_DIM, MODEL_ID, MODEL_REVISION, MODEL_SHA256, TOKENIZER_SHA256,
    Result, SearchHit, SearchReport, Status, cosine_similarity, rank_candidates,
    pool_and_normalize, report_json,
)


class TrapIDs(Sequence):
    def __len__(self):
        return 257

    def __getitem__(self, index):
        raise AssertionError("oversize candidate sequence was accessed")


def valid_report():
    return SearchReport(Status.COMPLETE, "OK", MODEL_ID, MODEL_REVISION,
                        MODEL_SHA256, TOKENIZER_SHA256, "a" * 64, 2, 10, 5, 2,
                        "cosine_similarity", (SearchHit("a", .75), SearchHit("b", .25)))


class PublicBoundaries(unittest.TestCase):
    def test_cosine_requires_admitted_unit_vectors(self):
        unit = np.zeros(384, dtype=np.float64)
        unit[0] = 1
        for vector in (unit * .5, unit * 0, unit * 1e150):
            result = cosine_similarity(unit, vector)
            self.assertEqual(result.status, Status.NUMERICALLY_INDETERMINATE)
            self.assertIsNone(result.value)

    def test_candidate_limit_precedes_iteration(self):
        unit = np.zeros(384, dtype=np.float64)
        unit[0] = 1
        result = rank_candidates(unit, TrapIDs(), np.zeros((257, EMBEDDING_DIM)), 1)
        self.assertEqual(result.status, Status.INVALID_INPUT)
        self.assertIsNone(result.value)

    def test_pooling_sequence_limit(self):
        result = pool_and_normalize(np.ones((1, 257, 384)), np.ones(257, dtype=np.int64))
        self.assertEqual(result.status, Status.NUMERICALLY_INDETERMINATE)
        self.assertIsNone(result.value)

    def test_report_cannot_claim_inconsistent_or_duplicate_ranked_hits(self):
        original = valid_report()
        malformed = [replace(original, hits=original.hits * 2),
                     replace(original, hits=(original.hits[0], original.hits[0])),
                     replace(original, hits=tuple(reversed(original.hits))),
                     replace(original, hits=()),
                     replace(original, corpus_count=0, hits=(), document_token_count=0),
                     replace(original, document_token_count=0)]
        for report in malformed:
            with self.subTest(report=report):
                result = report_json(Result(Status.COMPLETE, "OK", report))
                self.assertEqual(result.status, Status.INVALID_INPUT)
                self.assertIsNone(result.value)

    def test_complete_json_limit_counts_final_lf(self):
        result = Result(Status.COMPLETE, "OK", valid_report())
        size = len(report_json(result).value)
        with patch("local_semantic_atlas.types.MAX_REPORT_JSON_BYTES", size):
            self.assertEqual(len(report_json(result).value), size)
        with patch("local_semantic_atlas.types.MAX_REPORT_JSON_BYTES", size - 1):
            limited = report_json(result)
            self.assertEqual(limited.status, Status.INCOMPLETE)
            self.assertIsNone(limited.value)


if __name__ == "__main__":
    unittest.main()
