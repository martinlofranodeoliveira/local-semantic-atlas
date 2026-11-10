# Security and trust boundary

Please use GitHub private vulnerability reporting/security advisories for sensitive reports when the public repository enables them. Do not put private documents, model files, credentials or exploit data in public issues.

The SDK admits only the two pinned byte snapshots before constructing maintained native parsers. It does not sandbox native ONNX/tokenizer code or hostile Python introspection. Input files are cooperative bounded regular-file captures, not a hostile-filesystem boundary. Reports omit literal text and paths, but IDs and hashes can still reveal dataset identity. Never treat cosine scores as probabilities or evidence for high-stakes decisions.

Only explicit preparation makes network requests. Search uses approved owned snapshots with no implicit download/provider fallback. ORT events are disabled before session creation; the native import can still issue a device-ID persistence warning. The observed offline container prevents egress during validation; zero import telemetry behavior is not claimed.

The new preparation command accepts only an exclusive new directory in an operator-controlled parent. A failed preparation leaves staging files; a completion marker is not a signature, and consumers still hash-admit snapshots. Discard report stdout on any nonzero exit because downstream I/O can accept a prefix before failing.
