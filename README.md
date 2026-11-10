# Local Semantic Atlas

Local multilingual neural retrieval with an explicit model revision, bounded input and inspectable provenance. The approved E5 encoder embeds documents and queries; the SDK pools and normalizes actual hidden states, then ranks by float64 cosine similarity. It does not generate answers. Scores are relative similarities, not probabilities or relevance guarantees.

[Português](docs/README.pt-BR.md)

The qualified stack is Python 3.13.18, NumPy 2.5.4, ONNX Runtime 1.31.1 and tokenizers 0.23.3 on the observed Linux x86_64 CPU host. The selected upstream model is the qint8 `avx512_vnni` variant. Other CPUs, platforms and dependency versions are unqualified; an upstream variant name is not a portability guarantee.

## Prepare artifacts explicitly

Use the two-stage Docker workflow below, or install the built wheel after restoring the exact runtime lock. The installed `local-semantic-atlas` entry point calls the same bounded SDK.

```sh
local-semantic-atlas prepare-model --destination ./approved-model
```

This is the only network command. It downloads six fixed HTTPS URLs at revision `614241f622f53c4eeff9890bdc4f31cfecc418b3` of `intfloat/multilingual-e5-small`, verifies each exact byte length and SHA-256, and keeps the upstream README and configuration alongside the two runtime artifacts. There is no URL/revision/provider/authentication override. HTTPS CDN redirects are allowed; TLS downgrade is rejected; proxy environment configuration is not used. Each artifact has a 300-second checked deadline plus at most one 20-second blocking read interval.

The destination must not exist, and its parent must be controlled by the operator. Creation is exclusive with mode 0700. During download it is a private staging directory. `READY.json` is linked into place atomically without replacing an existing name after every artifact has been verified, closed and fsynced. It records the exact six hashes and lengths; it is a completion marker, not a signature. Failure returns nonzero and leaves staging data without a completed preparation claim; inspect or remove that newly created directory before choosing another destination. The tool does not resume, overwrite existing directories or remove user data. A failed final directory fsync may leave the completed marker present while the command still returns nonzero; use both exit status and the verified manifest when recording preparation.

Model files total about 136 MB. Weights and dependency wheels are fetched separately under their upstream terms; they are not project source. Preparation has controlled-response tests and an observed fresh live download through this exact downloader: all six fixed artifacts passed byte/hash checks. The resulting newly prepared snapshots feed the offline package/runtime gates.

## Search literal file snapshots

```sh
local-semantic-atlas search \
  --documents examples/documents.json \
  --query-file examples/query.pt.txt \
  --model approved-model/model_qint8_avx512_vnni.onnx \
  --tokenizer approved-model/tokenizer.json
```

The document file is closed JSON: exactly `schema_version: 1` and a `documents` array of objects with exactly string `id` and `text` fields. Duplicate object keys, floats, unknown fields and more than 256 rows are rejected. IDs are unique ASCII identifiers up to 64 bytes (`[A-Za-z0-9][A-Za-z0-9_.-]{0,63}`). Query and document text are literal: no whitespace stripping, Unicode normalization, chunking or truncation. Query files include any final LF in their contents. Input paths must open as regular files; final symlinks, FIFOs and devices are rejected. Reads capture bounded owned bytes and detect observed file identity/size/metadata changes; this is not a hostile-filesystem sandbox or a persistent path lock.

Search never downloads artifacts. The CLI checks lengths while reading, and the SDK verifies both hashes before any native tokenizer or model construction. Each invocation rebuilds its in-memory index and encodes the query; it does not save vectors or reuse a persistent cache. JSON stdout contains ranked IDs, scores, model/hash provenance and counts. It omits query/document text and source paths. IDs and a corpus digest can still identify a dataset; treat the report accordingly.

```sh
local-semantic-atlas search \
  --documents examples/documents.json --query-file examples/query.pt.txt \
  --model approved-model/model_qint8_avx512_vnni.onnx \
  --tokenizer approved-model/tokenizer.json \
  --format html > report.pt.html
```

HTML defaults to PT-BR; add `--locale en` for English. The self-contained printable report uses a semantic ranked table, an explicit empty/error state and a provenance list. It loads no script, image, font or other external resource. HTML is validated through the SDK's closed report schema and escaped before emission. Browser layout, assistive-technology behavior and browser accessibility conformance have not been observed; only source and native output checks are recorded.

## Limits and exits

| Boundary | Maximum |
| --- | --- |
| Documents / returned hits | 256 / `min(corpus_count, top_k)` |
| Individual query or document | 8,192 strict UTF-8 bytes, nonempty, no NUL/surrogates |
| Total document text | 1,048,576 UTF-8 bytes |
| Document JSON file | 2,097,152 bytes |
| Encoded passage/query, including prefix and special tokens | 256 tokens; excess is rejected before inference |
| Indexing token budget | 65,536 tokens |
| JSON / HTML report | 262,144 UTF-8 bytes each, including final LF |

`--top-k` defaults to 5 (1–256), `--max-documents` to 256 (0–256), and `--max-tokens` to 65,536 (0–65,536). A budget is a separate ceiling for each indexing/search operation, not one shared total across both. Empty corpora still encode an admitted query before reporting `COMPLETE_EMPTY`. No partial index/hits are published on exhaustion.

| Exit | Meaning |
| --- | --- |
| 0 | `COMPLETE` or `COMPLETE_EMPTY`; help/preparation also return 0 on success |
| 2 | Invalid input, artifact or command usage |
| 3 | Incomplete operation or input/output/preparation failure |
| 4 | Numerically indeterminate result |

Errors use stable codes and closed JSON without exception text or tracebacks. The CLI derives its exit from the operation, not from successful error serialization. Entire output bytes are staged before writing; a downstream write failure returns 3, with a best-effort diagnostic. As with any byte stream, the operating system may have accepted a prefix before an I/O failure: discard stdout unless the command exits 0. Redirection can create an empty/truncated destination even if the command fails.

ORT telemetry events are disabled before session creation. The pinned native library can still issue a read-only device-ID persistence warning during library import; zero native import telemetry behavior is not claimed. Actual checks ran with network disabled, one model thread per ORT setting and explicit resource caps. Results are not claimed bit-identical across hardware.

## Evidence scope

The 40 checks preserve all 15 current core checks and the 8 original file-boundary checks, adding 17 routing, HTML, preparation and adversarial checks. Actual approved-model CLI runs exercise JSON and both HTML locales, empty corpus query inference, token-budget failure, invalid UTF-8 and successful inference with both standard streams failing. Source-integration checks preceded the separate installed-package gates. They are not a retrieval-quality benchmark. Local package evidence and hosted exact-source CI remain distinct; independent audit and hosted CI are not inferred from local checks.


## Package workflow

Run `sh scripts/ci.sh`. The explicit network build (`Dockerfile.prepare`) restores the exact 24 development wheels from official PyPI with hashes and runs the fixed six-artifact downloader. The separate `docker build --network none` consumes that completed local image, builds a real wheel/sdist, verifies all package code/data, and rebuilds a wheel from the sdist. Its artifact-only runtime installs only the 20-entry runtime lock and that rebuilt wheel. There is no checkout, preparation source, compiler, build frontend or backend. The final no-bind container runs all checks, the independent scalar numerical consumer and actual installed CLI examples with the script's limits. The prepared local image must exist before the offline build; no cross-job cache is assumed.

Manual wheel installation: `python -m pip install --require-hashes -r requirements-runtime.lock`, then `python -m pip install --no-deps dist/local_semantic_atlas-0.1.0-py3-none-any.whl`. For offline installation add `--no-index --find-links=...` pointing to the explicitly prepared wheelhouse. Build tools use the separate `requirements-dev.lock`. Binary hashes cover only the qualified Linux x86_64/Python3.13 tuple.

See [security](SECURITY.md), [contributing](CONTRIBUTING.md) and [third-party provenance](NOTICE.md). The original code is MIT; upstream terms are separate. Published by [LOFRANO TECNOLOGIA](https://lofranotecnologia.com.br).


### Observed literal example

The committed `examples/query.pt.txt` includes a final LF, which is preserved as input. On the qualified host the actual installed CLI returned `COMPLETE`, 99 document tokens and 20 query tokens, ranking `stock` (0.8803328171865816), `maintenance` (0.7418338083175413), then `backup` (0.7131352326394726). [Full observed JSON](docs/observed-demo.json) records the model and corpus hashes. These are original synthetic maintenance/stock/backup examples, not a retrieval-quality benchmark. Removing the LF changes the literal query and can change its embedding and scores; no cross-hardware bit stability is promised.
