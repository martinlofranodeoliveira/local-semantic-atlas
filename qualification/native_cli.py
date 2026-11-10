"""Actual installed CLI neural integration outside source checkout."""
from pathlib import Path
import json
import os
import resource
import shutil
import subprocess
import sys
import tempfile

with tempfile.TemporaryDirectory(prefix="atlas-cli-native-") as temporary:
    root = Path(temporary)
    documents = {"schema_version": 1, "documents": [
        {"id": "stock", "text": "O controle de estoque calcula o ponto de reposição das peças. Quando o saldo fica abaixo do mínimo, o almoxarifado solicita uma nova compra."},
        {"id": "maintenance", "text": "The maintenance schedule specifies periodic inspection and lubrication of the machine bearings to prevent unexpected stoppages."},
        {"id": "backup", "text": "Backup copies are stored in a separate location and restoration is tested regularly. A backup in the same failed disk cannot restore the data."}]}
    query = "Quando devemos comprar mais peças para o almoxarifado?\n"
    (root / "documents.json").write_text(json.dumps(documents), encoding="utf-8")
    (root / "query.txt").write_text(query, encoding="utf-8")
    (root / "empty.json").write_text('{"schema_version":1,"documents":[]}', encoding="utf-8")
    base = ["local-semantic-atlas", "search",
            "--documents", "/opt/examples/documents.json", "--query-file", "/opt/examples/query.pt.txt",
            "--model", "/opt/model/model_qint8_avx512_vnni.onnx", "--tokenizer", "/opt/model/tokenizer.json"]
    def run(argv):
        return subprocess.run(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=90)
    results = []
    artifacts = {}
    for extra in ([], ["--format", "html"], ["--format", "html", "--locale", "en"]):
        process = run(base + extra)
        assert process.returncode == 0, (process.returncode, process.stderr.decode())
        assert query.encode() not in process.stdout
        assert str(root).encode() not in process.stdout
        assert all(row["text"].encode() not in process.stdout for row in documents["documents"])
        if not extra:
            frame = json.loads(process.stdout)
            assert frame["status"] == "COMPLETE" and frame["hits"][0]["id"] == "stock"
            assert len(frame["hits"]) == 3
            results.append({"format": "json", "status": frame["status"], "actual_top_id": frame["hits"][0]["id"]})
            artifacts["report.json"] = process.stdout.decode("utf-8")
        else:
            language = "en" if extra[-1] == "en" else "pt-BR"
            assert f'<html lang="{language}">'.encode() in process.stdout
            assert b"<table>" in process.stdout and b"<script" not in process.stdout
            results.append({"format": "html", "locale": language, "bytes": len(process.stdout)})
            artifacts[f"report.{language}.html"] = process.stdout.decode("utf-8")
    empty = list(base); empty[empty.index("--documents") + 1] = str(root / "empty.json")
    process = run(empty)
    assert process.returncode == 0
    frame = json.loads(process.stdout)
    assert frame["status"] == "COMPLETE_EMPTY" and frame["query_token_count"] > 0 and frame["hits"] == []
    process = run(base + ["--max-tokens", "0"])
    assert process.returncode == 3
    assert json.loads(process.stdout)["status"] == "INCOMPLETE"
    (root / "bad-query").write_bytes(b"\xff")
    bad = list(base); bad[bad.index("--query-file") + 1] = str(root / "bad-query")
    process = run(bad)
    assert process.returncode == 2 and json.loads(process.stdout)["code"] == "QUERY_UTF8"
    with open("/dev/full", "wb", buffering=0) as full:
        process = subprocess.run(base, stdout=full, stderr=full, timeout=90)
    assert process.returncode == 3
    print(json.dumps({"status": "PASS", "scope": "actual installed CLI outside checkout; actual approved encoder, not a model quality benchmark",
        "examples": results, "empty_still_encodes_query": True, "zero_token_budget_exit": 3,
        "invalid_query_utf8_exit": 2, "actual_inference_both_streams_full_exit": 3,
        "reports_omit_query_document_text_and_paths": True,
        "child_peak_rss_kib": resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
        "artifacts": artifacts}, indent=2))
