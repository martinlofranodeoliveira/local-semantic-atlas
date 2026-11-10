"""Explicit fixed-URL preparation; this module is never used during inference.

An exclusively created 0700 directory is staging until READY.json exists. That
manifest is linked into place without replacement only after all six files have
been hash/length checked and fsynced. Failure leaves an unready private directory.
"""
from __future__ import annotations

import hashlib
import http.client
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request

from . import Result, Status

_BASE = "https://huggingface.co/intfloat/multilingual-e5-small/resolve/614241f622f53c4eeff9890bdc4f31cfecc418b3/"
# Names, sizes and SHA-256 from the actual qualified immutable manifest. No URL,
# revision, credentials, artifact name or checksum can be supplied by the caller.
ARTIFACTS = (
    ("onnx/model_qint8_avx512_vnni.onnx", 118346824, "dd476dd0c2514e9b9be83aeb3853fac0763e0bdf4a71645407587d77c48a2d88"),
    ("onnx/tokenizer.json", 17082730, "0b44a9d7b51c3c62626640cda0e2c2f70fdacdc25bbbd68038369d14ebdf4c39"),
    ("onnx/config.json", 653, "bbb7c1333fc4b3e27fbc9cd5d2070aabcc1d4dfb99917c3633e772f97545a6b6"),
    ("onnx/tokenizer_config.json", 443, "a1d6bc8734a6f635dc158508bef000f8e2e5a759c7d92f984b2c86e5ff53425b"),
    ("onnx/special_tokens_map.json", 167, "d05497f1da52c5e09554c0cd874037a083e1dc1b9cfd48034d1c717f1afc07a7"),
    ("README.md", 497538, "0038de97aee16258cecbad7ffda4b4febd6953e747a00e0ddbc8e6ed241e9c1c"),
)


class _Preparation(Exception):
    def __init__(self, code: str):
        self.code = code


class _HTTPSRedirect(urllib.request.HTTPRedirectHandler):
    max_redirections = 5

    def redirect_request(self, request, response, code, message, headers, newurl):
        # The immutable upstream uses signed CDN redirects; all must retain TLS.
        if urllib.parse.urlsplit(newurl).scheme != "https":
            raise _Preparation("DOWNLOAD_REDIRECT")
        return super().redirect_request(request, response, code, message, headers, newurl)


def _write_all(descriptor: int, payload: bytes) -> None:
    view = memoryview(payload)
    while view:
        count = os.write(descriptor, view)
        if count <= 0:
            raise OSError("short file write")
        view = view[count:]


def _new_file(directory: int, name: str) -> int:
    return os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
                   0o600, dir_fd=directory)


def _download(opener, directory: int, path: str, size: int, expected: str) -> None:
    digest = hashlib.sha256()
    total = 0
    deadline = time.monotonic() + 300
    descriptor = _new_file(directory, path.rsplit("/", 1)[-1])
    try:
        with opener.open(_BASE + path, timeout=20) as response:
            if response.status != 200 or urllib.parse.urlsplit(response.url).scheme != "https":
                raise _Preparation("DOWNLOAD_HTTP")
            while True:
                if time.monotonic() > deadline:
                    raise _Preparation("DOWNLOAD_TIMEOUT")
                chunk = response.read(min(65_536, size + 1 - total))
                if not chunk:
                    break
                total += len(chunk)
                if total > size:
                    raise _Preparation("DOWNLOAD_SIZE")
                digest.update(chunk)
                _write_all(descriptor, chunk)
        if total != size:
            raise _Preparation("DOWNLOAD_SIZE")
        if digest.hexdigest() != expected:
            raise _Preparation("DOWNLOAD_HASH")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def prepare_model(destination: str) -> Result:
    """Explicit network command, exclusive staging; never replaces existing data."""
    if type(destination) is not str or not destination or "\x00" in destination:
        return Result(Status.INVALID_INPUT, "INVALID_DESTINATION")
    try:
        if len(destination.encode("utf-8", "strict")) > 4096:
            return Result(Status.INVALID_INPUT, "INVALID_DESTINATION")
    except UnicodeError:
        return Result(Status.INVALID_INPUT, "INVALID_DESTINATION")
    directory = None
    try:
        try:
            os.mkdir(destination, 0o700)
        except FileExistsError:
            return Result(Status.INVALID_INPUT, "DESTINATION_EXISTS")
        directory = os.open(destination, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
        # No proxy/environment credentials or authentication handlers. This sole
        # explicit command makes HTTPS requests; inference never calls it.
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _HTTPSRedirect())
        for path, size, expected in ARTIFACTS:
            _download(opener, directory, path, size, expected)
        manifest = {"schema_version": 1, "status": "COMPLETE", "code": "MODEL_PREPARED",
                    "model": "intfloat/multilingual-e5-small",
                    "revision": "614241f622f53c4eeff9890bdc4f31cfecc418b3",
                    "files": [{"name": path.rsplit("/", 1)[-1], "bytes": size, "sha256": sha}
                              for path, size, sha in ARTIFACTS]}
        frame = (json.dumps(manifest, ensure_ascii=True, separators=(",", ":")) + "\n").encode("ascii")
        temporary = ".READY.json.staging"
        descriptor = _new_file(directory, temporary)
        try:
            _write_all(descriptor, frame)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        # link is atomic and refuses an existing READY.json, unlike replace().
        os.link(temporary, "READY.json", src_dir_fd=directory, dst_dir_fd=directory, follow_symlinks=False)
        os.unlink(temporary, dir_fd=directory)
        os.fsync(directory)
        return Result(Status.COMPLETE, "MODEL_PREPARED",
                      b'{"schema_version":1,"status":"COMPLETE","code":"MODEL_PREPARED"}\n')
    except _Preparation as error:
        return Result(Status.INCOMPLETE, error.code)
    except (urllib.error.URLError, http.client.HTTPException, TimeoutError, OSError, ValueError):
        return Result(Status.INCOMPLETE, "PREPARATION_IO")
    finally:
        if directory is not None:
            try:
                os.close(directory)
            except OSError:
                pass
