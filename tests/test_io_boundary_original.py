import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from local_semantic_atlas._io_boundary import BoundaryError, DOCUMENT_JSON_BYTES, parse_documents, read_regular


class ClosedDocuments(unittest.TestCase):
    def test_literal_unicode_and_empty(self):
        payload = b'{"schema_version":1,"documents":[{"id":"a","text":"A\\u0301"}]}'
        self.assertEqual(parse_documents(payload), (("a", "A\u0301"),))
        self.assertEqual(parse_documents(b'{"schema_version":1,"documents":[]}'), ())

    def test_closed_shapes_and_numbers(self):
        malformed = [b'[]', b'{}', b'{"schema_version":true,"documents":[]}',
                     b'{"schema_version":1.0,"documents":[]}',
                     b'{"schema_version":1,"documents":[],"extra":0}',
                     b'{"schema_version":1,"documents":[{"id":"a","text":null}]}',
                     b'{"schema_version":1,"documents":[{"id":"a","text":"b","x":1}]}',
                     b'{"schema_version":NaN,"documents":[]}',
                     b'{"schema_version":' + b'1' * 21 + b',"documents":[]}']
        for payload in malformed:
            with self.subTest(payload=payload[:80]), self.assertRaises(BoundaryError):
                parse_documents(payload)

    def test_duplicates_invalid_utf8_and_depth(self):
        for payload in [b'{"schema_version":1,"documents":[],"documents":[]}',
                        b'{"schema_version":1,"documents":[{"id":"a","text":"x","id":"b"}]}',
                        b'\xff', b'[' * 2000 + b']' * 2000]:
            with self.assertRaises(BoundaryError):
                parse_documents(payload)

    def test_byte_bound_before_parser_and_row_count(self):
        with mock.patch("local_semantic_atlas._io_boundary.json.loads", side_effect=AssertionError("parser reached")):
            with self.assertRaises(BoundaryError):
                parse_documents(b" " * (DOCUMENT_JSON_BYTES + 1))
        for size in (256, 257):
            payload = json.dumps({"schema_version": 1, "documents": [{"id": str(i), "text": "x"} for i in range(size)]}).encode()
            if size == 256:
                self.assertEqual(len(parse_documents(payload)), 256)
            else:
                with self.assertRaises(BoundaryError):
                    parse_documents(payload)


class RegularSnapshots(unittest.TestCase):
    def test_exact_limit_and_empty(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input"
            path.write_bytes(b"abc")
            self.assertEqual(read_regular(str(path), 3, exact=3), b"abc")
            for kwargs in ({"maximum": 2}, {"maximum": 4, "exact": 4}):
                with self.assertRaises(BoundaryError):
                    read_regular(str(path), **kwargs)
            path.write_bytes(b"")
            self.assertEqual(read_regular(str(path), 0, exact=0), b"")

    def test_special_files_do_not_block_or_follow_links(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "regular").write_bytes(b"a")
            (root / "link").symlink_to(root / "regular")
            os.mkfifo(root / "pipe")
            for path in (root, root / "link", root / "pipe"):
                with self.subTest(path=path.name), self.assertRaises(BoundaryError):
                    read_regular(str(path), 10)

    def test_real_concurrent_growth_is_not_success(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input"
            path.write_bytes(b"abc")
            original = os.read
            changed = False

            def grow(fd, count):
                nonlocal changed
                if not changed:
                    changed = True
                    with path.open("ab") as output:
                        output.write(b"d")
                return original(fd, count)

            with mock.patch("local_semantic_atlas._io_boundary.os.read", side_effect=grow):
                with self.assertRaises(BoundaryError) as failure:
                    read_regular(str(path), 3)
            self.assertEqual(failure.exception.code, "FILE_SIZE")
            self.assertEqual(path.read_bytes(), b"abcd")

    def test_invalid_paths_and_operational_failure(self):
        for value in ("", "a\x00b", "\ud800", "a" * 4097):
            with self.assertRaises(BoundaryError) as failure:
                read_regular(value, 1)
            self.assertFalse(failure.exception.operational)
        with self.assertRaises(BoundaryError) as failure:
            read_regular("/this-original-fixture-path-does-not-exist-92841", 1)
        self.assertTrue(failure.exception.operational)


if __name__ == "__main__":
    unittest.main()
