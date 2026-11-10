"""Escaped, self-contained static HTML over the SDK's closed report schema."""
from __future__ import annotations

from html import escape
import json

from . import Result, Status, report_json

MAX_HTML_BYTES = 262_144
_TEXT = {
    "pt-BR": {
        "title": "Relatório de busca semântica", "results": "Documentos classificados",
        "rank": "Posição", "document": "ID do documento", "score": "Similaridade de cosseno",
        "notice": "Similaridade relativa entre vetores. Os valores não são probabilidades nem garantias de relevância.",
        "empty": "O corpus está vazio. A consulta foi codificada pelo modelo aprovado; não há documentos para classificar.",
        "failure": "A operação não foi concluída. Nenhum resultado parcial foi publicado.",
        "recovery": "Consulte o código abaixo e os limites documentados antes de executar novamente.",
        "provenance": "Proveniência e limites", "model": "Modelo", "revision": "Revisão",
        "model_hash": "SHA-256 do modelo", "tokenizer_hash": "SHA-256 do tokenizador",
        "dataset": "SHA-256 do corpus", "count": "Documentos no corpus",
        "doc_tokens": "Tokens dos documentos", "query_tokens": "Tokens da consulta",
        "top_k": "Limite de resultados", "privacy": "Este relatório contém IDs, escores e proveniência. O texto da consulta e dos documentos foi omitido.",
        "status": "Estado", "code": "Código",
    },
    "en": {
        "title": "Semantic search report", "results": "Ranked documents",
        "rank": "Rank", "document": "Document ID", "score": "Cosine similarity",
        "notice": "Relative vector similarity. Values are not probabilities or guarantees of relevance.",
        "empty": "The corpus is empty. The approved model encoded the query; there are no documents to rank.",
        "failure": "The operation did not complete. No partial result was published.",
        "recovery": "Check the code below and the documented limits before running again.",
        "provenance": "Provenance and limits", "model": "Model", "revision": "Revision",
        "model_hash": "Model SHA-256", "tokenizer_hash": "Tokenizer SHA-256",
        "dataset": "Corpus SHA-256", "count": "Corpus documents",
        "doc_tokens": "Document tokens", "query_tokens": "Query tokens",
        "top_k": "Result limit", "privacy": "This report contains IDs, scores and provenance. Query and document text are omitted.",
        "status": "Status", "code": "Code",
    },
}

_CSS = """
:root{color-scheme:light;--paper:#fbfaf6;--ink:#172f35;--muted:#405a60;--rule:#b3c5c3;--accent:#145e58}
*{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);font:1rem/1.6 system-ui,sans-serif}
main{max-width:64rem;margin:3rem auto;padding:0 1.5rem 3rem}h1{max-width:24ch;font:600 clamp(2rem,5vw,3.25rem)/1.15 Georgia,serif;letter-spacing:-.025em;margin:0 0 1.25rem;text-wrap:balance}
h2{font-size:1.25rem;margin:2.5rem 0 .75rem}p{max-width:72ch;margin:.75rem 0}.identity{display:flex;flex-wrap:wrap;gap:.35rem 1.5rem;color:var(--muted);margin-bottom:1.5rem}
.status{font-weight:700;color:var(--accent)}.table-wrap{overflow-x:auto}table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}caption{text-align:left;font-weight:650;padding:0 0 .75rem}
th,td{text-align:left;padding:.9rem .75rem;border-bottom:1px solid var(--rule);vertical-align:top}thead{border-top:2px solid var(--ink)}th{font-size:.9rem}td:first-child{width:5rem;color:var(--muted)}th:last-child,td:last-child{text-align:right}code{font-family:ui-monospace,monospace;font-size:.9em;overflow-wrap:anywhere}td code{white-space:normal}
.note,footer{color:var(--muted);font-size:.92rem}dl{margin:0}dl>div{display:grid;grid-template-columns:minmax(10rem,1fr) minmax(0,3fr);gap:.25rem 1.5rem;padding:.55rem 0;border-bottom:1px solid var(--rule)}dt{color:var(--muted)}dd{margin:0;overflow-wrap:anywhere;font-variant-numeric:tabular-nums}footer{margin-top:2.5rem;padding-top:1rem;border-top:1px solid var(--ink)}
::selection{background:#c4e4d8;color:#102f2c}:focus-visible{outline:3px solid var(--accent);outline-offset:3px}
@media(max-width:36rem){main{margin:1.5rem auto;padding-inline:1rem}dl>div{grid-template-columns:1fr;gap:0}th,td{padding:.75rem .4rem}table{min-width:28rem}}
@media print{body{background:white;color:black}main{max-width:none;margin:0;padding:0}h1{font-size:24pt}tr,dl>div{break-inside:avoid}.table-wrap{overflow:visible}thead{display:table-header-group}table{min-width:0}footer{font-size:9pt}}
"""


def _encoded_html(result: Result, locale: str, document: dict) -> bytes:
    text = _TEXT[locale]
    safe = lambda value: escape(str(value), quote=True)
    status = safe(document["status"])
    code = safe(document["code"])
    header = (f'<div class="identity"><span class="status">{text["status"]}: {status}</span>'
              f'<span>{text["code"]}: <code>{code}</code></span></div>')
    if result.status not in (Status.COMPLETE, Status.COMPLETE_EMPTY):
        body = f'<p>{text["failure"]}</p><p>{text["recovery"]}</p>'
    else:
        if result.status == Status.COMPLETE_EMPTY:
            body = f'<p>{text["empty"]}</p>'
        else:
            rows = "".join(f'<tr><td>{rank}</td><td><code>{safe(hit["id"])}</code></td>'
                           f'<td>{safe(repr(hit["score"]))}</td></tr>'
                           for rank, hit in enumerate(document["hits"], 1))
            body = (f'<div class="table-wrap"><table><caption>{text["results"]}</caption><thead><tr>'
                    f'<th scope="col">{text["rank"]}</th><th scope="col">{text["document"]}</th>'
                    f'<th scope="col">{text["score"]}</th></tr></thead><tbody>{rows}</tbody></table></div>')
        body += f'<p class="note">{text["notice"]}</p><h2>{text["provenance"]}</h2><dl>'
        model = document["model"]
        fields = (("model", model["id"]), ("revision", model["revision"]),
                  ("model_hash", model["model_sha256"]), ("tokenizer_hash", model["tokenizer_sha256"]),
                  ("dataset", document["dataset_digest"]), ("count", document["corpus_count"]),
                  ("doc_tokens", document["document_token_count"]), ("query_tokens", document["query_token_count"]),
                  ("top_k", document["top_k"]))
        for label, value in fields:
            body += f'<div><dt>{text[label]}</dt><dd><code>{safe(value)}</code></dd></div>'
        body += "</dl>"
    return (f'<!doctype html>\n<html lang="{locale}"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width, initial-scale=1">'
            '<meta name="referrer" content="no-referrer">'
            '<meta http-equiv="Content-Security-Policy" content="default-src &apos;none&apos;; style-src &apos;unsafe-inline&apos;; base-uri &apos;none&apos;; form-action &apos;none&apos;">'
            f'<title>{text["title"]} · Local Semantic Atlas</title><style>{_CSS}</style></head>'
            f'<body><main><h1>{text["title"]}</h1>{header}{body}'
            f'<footer><p>{text["privacy"]}</p></footer></main></body></html>\n').encode("utf-8")


def report_html(result: object, *, locale: str = "pt-BR") -> Result:
    """Validate through the SDK before rendering; serialization is not search success."""
    if type(locale) is not str or locale not in _TEXT:
        return Result(Status.INVALID_INPUT, "INVALID_LOCALE")
    closed = report_json(result)
    if closed.status != Status.COMPLETE:
        return closed
    document = json.loads(closed.value)
    frame = _encoded_html(result, locale, document)
    if len(frame) > MAX_HTML_BYTES:
        return Result(Status.INCOMPLETE, "HTML_SIZE_EXCEEDED")
    return Result(Status.COMPLETE, "OK", frame)
