"""CLI/HTML boundaries; stubs here test routing only, never neural accuracy."""
from dataclasses import replace
from html.parser import HTMLParser
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from local_semantic_atlas import (MODEL_ID, MODEL_REVISION, MODEL_SHA256, TOKENIZER_SHA256,
    Result, SearchHit, SearchReport, Status, cosine_similarity, pool_and_normalize, report_json)
from local_semantic_atlas import cli
from local_semantic_atlas._io_boundary import BoundaryError, parse_documents, read_regular
from local_semantic_atlas.html_report import MAX_HTML_BYTES, report_html
from local_semantic_atlas.types import validate_utf8_text


def report(empty=False):
    status = Status.COMPLETE_EMPTY if empty else Status.COMPLETE
    code = "EMPTY_CORPUS" if empty else "OK"
    return Result(status, code, SearchReport(status, code, MODEL_ID, MODEL_REVISION,
        MODEL_SHA256, TOKENIZER_SHA256, "a" * 64, 0 if empty else 2,
        0 if empty else 10, 5, 2, "cosine_similarity",
        () if empty else (SearchHit("a", .75), SearchHit("b", .25))))


class Markup(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.tags = []
        self.attrs = []
        self.text = []

    def handle_starttag(self, tag, attrs):
        self.tags.append(tag)
        self.attrs.append((tag, dict(attrs)))

    def handle_data(self, data):
        self.text.append(data)


class CoreIndependentAdmission(unittest.TestCase):
    def test_literal_utf8_bounds_do_not_count_codepoints(self):
        self.assertEqual(len(validate_utf8_text("é" * 4096)), 8192)
        for value in ("é" * 4097, "a" * 8193, "\x00", "\ud800", ""):
            self.assertIsNone(validate_utf8_text(value))

    def test_numeric_types_and_masked_nonfinite_are_rejected(self):
        unit = np.zeros(384); unit[0] = 1
        for value in (unit.astype(bool), unit.astype(complex), np.full(384, np.nan), unit * .5):
            checked = cosine_similarity(unit, value)
            self.assertEqual(checked.status, Status.NUMERICALLY_INDETERMINATE)
            self.assertIsNone(checked.value)
        hidden = np.ones((1, 2, 384)); hidden[0, 1, 0] = np.inf
        self.assertEqual(pool_and_normalize(hidden, np.array([1, 0])).status,
                         Status.NUMERICALLY_INDETERMINATE)

    def test_closed_report_rejects_foreign_types_nan_order_and_count(self):
        original = report()
        malformed = [replace(original.value, hits=(SearchHit("a", float("nan")), SearchHit("b", .25))),
                     replace(original.value, hits=(SearchHit("a", 1), SearchHit("b", .25))),
                     replace(original.value, hits=(SearchHit("b", .75), SearchHit("a", .75))),
                     replace(original.value, corpus_count=True),
                     replace(original.value, dataset_digest="A" * 64),
                     replace(original.value, query_token_count=257),
                     replace(original.value, model_revision="unapproved")]
        for value in malformed:
            with self.subTest(value=value):
                self.assertEqual(report_json(Result(original.status, original.code, value)).status,
                                 Status.INVALID_INPUT)
        self.assertEqual(report_json({"text": "SECRET"}).status, Status.INVALID_INPUT)


class HTMLBoundary(unittest.TestCase):
    def test_both_locales_have_semantics_no_external_resources_or_query_text(self):
        for locale, title in (("pt-BR", "Relatório de busca semântica"), ("en", "Semantic search report")):
            frame = report_html(report(), locale=locale).value
            parsed = Markup(); parsed.feed(frame.decode("utf-8"))
            self.assertIn(title, "".join(parsed.text))
            self.assertIn(("html", {"lang": locale}), parsed.attrs)
            self.assertEqual(parsed.tags.count("main"), 1)
            self.assertEqual(parsed.tags.count("h1"), 1)
            self.assertEqual(parsed.tags.count("caption"), 1)
            self.assertEqual(sum(tag == "th" and attrs.get("scope") == "col" for tag, attrs in parsed.attrs), 3)
            self.assertFalse(set(parsed.tags) & {"script", "iframe", "img", "link", "form"})
            self.assertFalse(any(set(attrs) & {"src", "href"} for _, attrs in parsed.attrs))
            self.assertIn(MODEL_SHA256.encode(), frame)
            self.assertLessEqual(len(frame), MAX_HTML_BYTES)

    def test_error_empty_and_forged_markup_are_closed(self):
        for original in (Result(Status.INVALID_INPUT, "BAD_QUERY"), report(empty=True)):
            frame = report_html(original).value
            self.assertNotIn(b"<table", frame)
            self.assertIn(original.code.encode(), frame)
        forged = report()
        injected = replace(forged.value, hits=(SearchHit("<script>", .75), SearchHit("b", .25)))
        self.assertEqual(report_html(Result(forged.status, forged.code, injected)).status, Status.INVALID_INPUT)
        self.assertEqual(report_html(forged, locale="xx<script>").status, Status.INVALID_INPUT)

    def test_html_limit_includes_complete_utf8_frame(self):
        original = report()
        size = len(report_html(original).value)
        with patch("local_semantic_atlas.html_report.MAX_HTML_BYTES", size):
            self.assertEqual(len(report_html(original).value), size)
        with patch("local_semantic_atlas.html_report.MAX_HTML_BYTES", size - 1):
            result = report_html(original)
            self.assertEqual(result.status, Status.INCOMPLETE)
            self.assertIsNone(result.value)


class CLIStatusAndIO(unittest.TestCase):
    def invoke(self, argv, result=None):
        output = io.TextIOWrapper(io.BytesIO(), encoding="utf-8")
        with patch.object(sys, "stdout", output), patch.object(cli, "_search", return_value=result) as search_:
            status = cli.main(argv)
            output.flush()
            return status, output.buffer.getvalue(), search_.call_count

    def args(self, *extra):
        return ["search", "--documents", "private-docs", "--query-file", "private-query",
                "--model", "private-model", "--tokenizer", "private-tokenizer", *extra]

    def test_operation_status_survives_successful_serialization(self):
        for original, expected in ((report(), 0), (report(empty=True), 0),
            (Result(Status.INVALID_INPUT, "BAD_QUERY"), 2),
            (Result(Status.INVALID_ARTIFACT, "MODEL_HASH"), 2),
            (Result(Status.INCOMPLETE, "QUERY_TOKEN_BUDGET"), 3),
            (Result(Status.NUMERICALLY_INDETERMINATE, "ZERO_NORM"), 4)):
            self.assertEqual(report_json(original).status, Status.COMPLETE)
            code, frame, called = self.invoke(self.args(), original)
            self.assertEqual((code, called), (expected, 1))
            self.assertEqual(json.loads(frame)["status"], original.status.value)
            self.assertTrue(frame.endswith(b"\n"))
            self.assertNotIn(b"private-", frame)

    def test_usage_is_closed_and_never_echoes_arguments(self):
        for args in (["unknown-SECRET"], self.args("--top-k", "2.5"), self.args("--top", "2"),
                     self.args("--locale", "SECRET"), []):
            code, frame, called = self.invoke(args)
            self.assertEqual((code, called), (2, 0))
            self.assertEqual(json.loads(frame), {"schema_version": 1, "status": "INVALID_INPUT", "code": "USAGE"})

    def test_help_is_staged_and_defaults_to_portuguese_html(self):
        code, frame, called = self.invoke(["--help"])
        self.assertEqual((code, called), (0, 0)); self.assertIn(b"prepare-model", frame)
        code, frame, called = self.invoke(self.args("--format", "html"), report())
        self.assertEqual(code, 0); self.assertIn(b'<html lang="pt-BR">', frame)

    def test_stream_write_short_and_flush_failures_return_three(self):
        class BadBuffer:
            def __init__(self, flush=False): self.flush_ = flush
            def write(self, frame):
                if self.flush_: return len(frame)
                return len(frame) - 1
            def flush(self): raise OSError("SECRET")
        class BadOutput:
            def __init__(self, flush=False): self.buffer = BadBuffer(flush)
            def fileno(self): raise ValueError()
        class BadError:
            def write(self, frame): raise OSError("SECRET")
            def flush(self): raise OSError("SECRET")
        for flush in (False, True):
            with patch.object(sys, "stdout", BadOutput(flush)), patch.object(sys, "stderr", BadError()), patch.object(cli, "_search", return_value=report()):
                self.assertEqual(cli.main(self.args()), 3)

    def test_native_help_and_error_with_both_streams_full(self):
        for args in (["--help"], ["bad-SECRET"], self.args()):
            with open("/dev/full", "wb", buffering=0) as full:
                result = subprocess.run([sys.executable, "-B", "-m", "local_semantic_atlas.cli", *args],
                                        stdout=full, stderr=full, timeout=20)
            self.assertEqual(result.returncode, 3)


class FileBoundary(unittest.TestCase):
    def test_closed_json_duplicate_keys_invalid_numbers_and_bytes(self):
        for payload in (b'{"schema_version":1,"schema_version":1,"documents":[]}',
                        b'{"schema_version":1.0,"documents":[]}',
                        b'{"schema_version":NaN,"documents":[]}',
                        b'{"schema_version":true,"documents":[]}',
                        b'{"schema_version":1,"documents":[],"text":"SECRET"}', b"\xff"):
            with self.assertRaises(BoundaryError): parse_documents(payload)
        self.assertEqual(parse_documents(b'{"schema_version":1,"documents":[{"id":"a","text":" literal "}]}'),
                         (("a", " literal "),))

    def test_regular_snapshot_symlink_fifo_and_exact_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); path = root / "input"; path.write_bytes(b"abc")
            self.assertEqual(read_regular(str(path), 3, exact=3), b"abc")
            with self.assertRaises(BoundaryError): read_regular(str(path), 2)
            (root / "link").symlink_to(path)
            os.mkfifo(root / "fifo")
            for name in ("link", "fifo"):
                with self.assertRaises(BoundaryError): read_regular(str(root / name), 3)


if __name__ == "__main__":
    unittest.main()
