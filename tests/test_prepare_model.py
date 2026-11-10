"""Preparation transaction tests use tiny literal HTTPS responses, never inference."""
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import urllib.request

from local_semantic_atlas import Status
from local_semantic_atlas import prepare_model as module


class Response(io.BytesIO):
    status = 200
    url = "https://cdn.example.invalid/fixed"


class Preparation(unittest.TestCase):
    def test_exclusive_staging_complete_manifest_and_fixed_url(self):
        content = b"literal bytes"
        records = (("README.md", len(content), hashlib.sha256(content).hexdigest()),)
        class Opener:
            def open(self, url, timeout):
                self.url = url; self.timeout = timeout
                return Response(content)
        opener = Opener()
        with tempfile.TemporaryDirectory() as parent, patch.object(module, "ARTIFACTS", records), patch.object(module.urllib.request, "build_opener", return_value=opener):
            target = Path(parent) / "new"
            result = module.prepare_model(str(target))
            self.assertEqual(result.status, Status.COMPLETE)
            self.assertEqual((target / "README.md").read_bytes(), content)
            self.assertEqual(json.loads((target / "READY.json").read_bytes())["files"][0]["sha256"], records[0][2])
            self.assertEqual(opener.url, module._BASE + "README.md")
            self.assertEqual(module.prepare_model(str(target)).code, "DESTINATION_EXISTS")
            self.assertEqual((target / "README.md").read_bytes(), content)

    def test_size_hash_and_interrupted_io_never_publish_ready(self):
        good = b"literal bytes"; digest = hashlib.sha256(good).hexdigest()
        class Opener:
            def __init__(self, content): self.content = content
            def open(self, url, timeout): return Response(self.content)
        for content, expected in ((good[:-1], "DOWNLOAD_SIZE"), (good + b"x", "DOWNLOAD_SIZE"),
                                  (b"x" * len(good), "DOWNLOAD_HASH")):
            with tempfile.TemporaryDirectory() as parent, patch.object(module, "ARTIFACTS", (("README.md", len(good), digest),)), patch.object(module.urllib.request, "build_opener", return_value=Opener(content)):
                target = Path(parent) / "new"
                result = module.prepare_model(str(target))
                self.assertEqual((result.status, result.code), (Status.INCOMPLETE, expected))
                self.assertFalse((target / "READY.json").exists())
        with tempfile.TemporaryDirectory() as parent, patch.object(module, "_download", side_effect=OSError("SECRET")):
            target = Path(parent) / "new"
            self.assertEqual(module.prepare_model(str(target)).code, "PREPARATION_IO")
            self.assertFalse((target / "READY.json").exists())

    def test_redirects_cannot_downgrade_tls(self):
        request = urllib.request.Request(module._BASE + "README.md")
        with self.assertRaises(module._Preparation):
            module._HTTPSRedirect().redirect_request(request, None, 302, "", {}, "http://example.invalid/data")

    def test_fixed_manifest_matches_qualified_hashes(self):
        self.assertEqual(len(module.ARTIFACTS), 6)
        self.assertEqual(module.ARTIFACTS[0], ("onnx/model_qint8_avx512_vnni.onnx", 118346824,
            "dd476dd0c2514e9b9be83aeb3853fac0763e0bdf4a71645407587d77c48a2d88"))
        self.assertEqual(module.ARTIFACTS[1], ("onnx/tokenizer.json", 17082730,
            "0b44a9d7b51c3c62626640cda0e2c2f70fdacdc25bbbd68038369d14ebdf4c39"))


if __name__ == "__main__": unittest.main()
