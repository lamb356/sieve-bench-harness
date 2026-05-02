# semantic-hard-v1

Canonical data-driven semantic-hard manifests for the SIEVE benchmark harness.

Each `{language}.jsonl` file contains exactly 300 rows selected from the full eval corpus for that language. A row is included only when the harness `RipgrepRetriever` missed the paired code document in the top 5 (`ripgrep.recall@5 == 0.0`) against the full-language eval corpus.

Supported languages:

- `python` — CoIR/CodeSearchNet Python test qrels
- `typescript` — pinned TypeScript tree-sitter dedupe test split
- `go` — CoIR/CodeSearchNet Go test qrels
- `rust` — pinned Rust tree-sitter dedupe test split

Required row fields:

- `schema_version`: `1`
- `selection`: `semantic-hard-v1`
- `language`
- `source`, `corpus_id`, `dataset_id`, `dataset_revision`, `eval_split`
- `query_id`, `ground_truth_document_id`, `ground_truth_path`
- `query`, `ground_truth_code`
- `query_sha256`, `ground_truth_code_sha256`
- `ripgrep.recall@5`: must be `0.0`
- `provenance.source_metadata`: source loader metadata for traceability

The loader validates language, hashes, duplicate query/document/code identities, minimum row count, and corpus drift before constructing a `LoadedBenchmark`.
