"""Actual pinned E5 inference and an independent scalar numerical consumer.

Run with prepared model artifacts, outside the source checkout. These three
original bilingual examples are examples, not a model quality benchmark.
"""
from __future__ import annotations

from dataclasses import FrozenInstanceError
import hashlib
import importlib.metadata
import importlib.util
import json
import math
from pathlib import Path
import resource
import shutil
import sys
from unittest.mock import patch

import numpy as np
import local_semantic_atlas as atlas
from local_semantic_atlas.engine import _ENCODER_REGISTRY, _INDEX_REGISTRY


TEXTS = {
    "stock": "O controle de estoque calcula o ponto de reposição das peças. Quando o saldo fica abaixo do mínimo, o almoxarifado solicita uma nova compra.",
    "maintenance": "The maintenance schedule specifies periodic inspection and lubrication of the machine bearings to prevent unexpected stoppages.",
    "backup": "Backup copies are stored in a separate location and restoration is tested regularly. A backup in the same failed disk cannot restore the data.",
}
QUERIES = [
    ("Quando devemos comprar mais peças para o almoxarifado?", "stock"),
    ("How often should bearings be serviced?", "maintenance"),
    ("Where should a backup copy be stored?", "backup"),
]


def scalar_embedding(backend, prefixed_text):
    """Reference pooling/norm: scalar math.fsum, never production numerics."""
    encoding = backend.tokenizer.encode(prefixed_text, add_special_tokens=True)
    feeds = {"input_ids": np.asarray([encoding.ids], dtype=np.int64),
             "attention_mask": np.asarray([encoding.attention_mask], dtype=np.int64),
             "token_type_ids": np.asarray([encoding.type_ids], dtype=np.int64)}
    hidden = backend.session.run(["last_hidden_state"], feeds)[0]
    positions = [i for i, flag in enumerate(encoding.attention_mask) if flag == 1]
    values = [math.fsum(float(hidden[0, i, j]) for i in positions) / len(positions)
              for j in range(384)]
    norm = math.sqrt(math.fsum(x * x for x in values))
    return [x / norm for x in values]


def main():
    assert len(sys.argv) == 2, "one prepared model directory required"
    root = Path(sys.argv[1])
    installed = Path(atlas.__file__).resolve()
    assert "site-packages" in installed.parts and installed.is_relative_to(Path(sys.prefix))
    assert not Path("/project").exists() and not Path("/build-source").exists()
    assert importlib.util.find_spec("build") is None
    assert importlib.util.find_spec("setuptools") is None
    assert shutil.which("gcc") is None
    model = (root / "model_qint8_avx512_vnni.onnx").read_bytes()
    tokenizer = (root / "tokenizer.json").read_bytes()
    assert hashlib.sha256(model).hexdigest() == atlas.MODEL_SHA256
    assert hashlib.sha256(tokenizer).hexdigest() == atlas.TOKENIZER_SHA256
    prepared = atlas.ApprovedEncoder.from_snapshots(model, tokenizer)
    assert prepared.status is atlas.Status.COMPLETE, (prepared.status, prepared.code)
    encoder = prepared.value
    del model, tokenizer, prepared
    assert not hasattr(encoder, "__dict__")
    backend = _ENCODER_REGISTRY[encoder]
    documents = [atlas.Document(key, value) for key, value in TEXTS.items()]
    built = atlas.build_index(documents, encoder)
    assert built.status is atlas.Status.COMPLETE, (built.status, built.code)
    index = built.value
    assert index.document_ids == ("backup", "maintenance", "stock")
    assert not hasattr(index, "__dict__")
    try:
        index.document_count = 999
        raise AssertionError("mutable index metadata")
    except (FrozenInstanceError, AttributeError):
        pass
    matrix = _INDEX_REGISTRY[index].vectors
    buffer = matrix
    while isinstance(buffer, np.ndarray):
        assert not buffer.flags.writeable
        try:
            buffer.setflags(write=True)
            raise AssertionError("mutable vector backing")
        except ValueError:
            pass
        buffer = buffer.base
    assert type(buffer) is bytes
    reference_documents = {key: scalar_embedding(backend, "passage: " + text)
                           for key, text in TEXTS.items()}
    results = []
    for query, expected in QUERIES:
        found = atlas.search(index, query, top_k=3)
        assert found.status is atlas.Status.COMPLETE, (found.status, found.code)
        assert found.value.hits[0].id == expected
        reference_query = scalar_embedding(backend, "query: " + query)
        reference_scores = {key: math.fsum(a * b for a, b in zip(reference_query, vec))
                            for key, vec in reference_documents.items()}
        assert [hit.id for hit in found.value.hits] == sorted(reference_scores, key=lambda key: (-reference_scores[key], key))
        assert max(abs(hit.score - reference_scores[hit.id]) for hit in found.value.hits) < 1e-10
        encoded = atlas.report_json(found)
        assert encoded.status is atlas.Status.COMPLETE
        assert query.encode() not in encoded.value
        assert not any(text.encode() in encoded.value for text in TEXTS.values())
        results.append({"expected_example_top_id": expected, "actual_top_id": found.value.hits[0].id,
                        "scores": [{"id": hit.id, "score": hit.score} for hit in found.value.hits]})
    empty = atlas.build_index([], encoder, atlas.WorkBudget(0, 0)).value
    assert atlas.search(empty, "valid query", budget=atlas.WorkBudget(0, 0)).status is atlas.Status.INCOMPLETE
    assert atlas.search(empty, "valid query", budget=atlas.WorkBudget(0, 256)).status is atlas.Status.COMPLETE_EMPTY
    assert atlas.search(empty, "").status is atlas.Status.INVALID_INPUT
    boundaries = {}
    for repetitions in range(300):
        text = "x " * repetitions
        count = len(backend.tokenizer.encode("query: " + text, add_special_tokens=True).ids)
        if count in (256, 257) and count not in boundaries:
            boundaries[count] = (text, repetitions)
    assert set(boundaries) == {256, 257}
    with patch.object(backend.session, "run", wraps=backend.session.run) as native_run:
        over = atlas.search(index, boundaries[257][0])
        assert over.status is atlas.Status.INVALID_INPUT and over.code == "TOKEN_LIMIT"
        native_run.assert_not_called()
        budget = atlas.search(index, boundaries[256][0], budget=atlas.WorkBudget(256, 255))
        assert budget.status is atlas.Status.INCOMPLETE and budget.code == "TOKEN_BUDGET_EXCEEDED"
        native_run.assert_not_called()
        exact = atlas.search(index, boundaries[256][0], budget=atlas.WorkBudget(256, 256))
        assert exact.status is atlas.Status.COMPLETE and exact.value.query_token_count == 256
        assert native_run.call_count == 1
    print(json.dumps({"status": "PASS", "scope": "actual installed-package native E5 examples and independent scalar numerical consumer; not a quality benchmark",
                      "package": str(installed), "versions": {name: importlib.metadata.version(name) for name in ("local-semantic-atlas", "numpy", "onnxruntime", "tokenizers")},
                      "examples": results, "independent_scalar_cosine_tolerance": 1e-10,
                      "actual_query_boundary_repetitions": {str(k): v[1] for k, v in boundaries.items()},
                      "exact256_tokens_native_inference": True, "257_tokens_and255_budget_before_native_call": True,
                      "empty_corpus_still_embeds_valid_query": True, "source_checkout_available": False,
                      "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}, ensure_ascii=False))


if __name__ == "__main__":
    main()
