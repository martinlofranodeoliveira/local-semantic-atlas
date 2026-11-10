# Contributing

Use Python 3.13 and the exact Linux x86_64 locks. `sh scripts/ci.sh` performs separate network preparation, offline wheel/sdist builds and fresh artifact-only validation. It can download approximately 136 MB of fixed model artifacts plus dependency wheels. No keys are needed. Do not mount a home directory or forward environment credentials into containers.

Preserve admission before inference, actual hidden-state pooling, typed complete/incomplete outcomes, fixed artifact provenance, immutable vectors and privacy. Keep every existing check, add counterexamples for substantive defects, and distinguish model examples from quality benchmarks. Do not commit weights, wheel files or caches. Changes to model/provider/CPU scope require new native qualification and exact artifact review.

Static HTML checks do not establish browser accessibility or visual conformance. Document new observations at their actual scope. Dependency and native component notices belong to their upstream authors; preserve complete terms and actual provenance rather than replacing them with a license summary.
