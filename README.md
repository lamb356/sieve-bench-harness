# sieve-bench-harness

Production-grade benchmark harness for public multi-language code retrieval evaluation.

Current status: Phase A implemented.

Phase A scope
- Python only
- CoIR primary benchmark loader
- ripgrep retriever
- retrieval metrics (Recall@1/5/10, MRR@10, NDCG@10)
- mandatory contamination gate via normalized SHA-256 + Bloom filter
- one-command quickcheck output for the Phase A launch gate

Repo layout
- `bench/loaders/` benchmark source loaders
- `bench/retrievers/` pluggable retrievers
- `bench/metrics/` retrieval and performance metrics
- `bench/contamination/` code normalization and Bloom filter logic
- `bench/runners/` benchmark orchestration
- `bench/report/` report generation and templates
- `bench-results/` generated outputs (gitignored)
- `bench/cache/` local cache and Bloom filters (gitignored)

Pinned public benchmark provenance
- CoIR dataset id: `CoIR-Retrieval/CodeSearchNet`
- CoIR pinned revision: `25e0292562b7bee26dd9b2d83a03981795862c77`
- Phase A language: Python only
- Phase A loader uses official `python-qrels/test-00000-of-00001.parquet`

Audit note on licensing/provenance
- The Hugging Face card for `CoIR-Retrieval/CodeSearchNet` does not currently expose strong license metadata in `card_data`.
- This is documented explicitly rather than treated as a route blocker for Phase A.
- Phase A consumes only the public benchmark parquet artifacts hosted at the pinned revision and does not pull any private Carson/lamb356 repositories.
- Phase A synthetic contamination fixture is public and local-only.

Contamination checks
Contamination checks are mandatory. The runner will hard-fail if the Bloom filter file is missing.

Phase A quickcheck uses a fixture Bloom filter built from a canary normalized hash so the contamination gate is exercised end-to-end without waiting for the full 200M-example CoRNStack filter build. This is a harness-verification filter, not the final production contamination audit artifact. The quickcheck benchmark itself uses a deterministic 100-query CoIR-Python sample.

Install
```bash
python -m pip install -e '.[dev]'
```

Run tests
```bash
PYTHONPATH=. pytest -q
```

Phase A quickcheck
```bash
make bench-python-quickcheck
```

Outputs
- `bench-results/phase-a-python-quickcheck/results.json`
- `bench-results/phase-a-python-quickcheck/hero-table.md`

Expected quickcheck behavior
- ripgrep must achieve non-zero Recall@5 on CoIR-Python
- contamination filter must be active
- any run without a Bloom filter must fail before benchmarking

Notes on CoIR Python mapping for Phase A
The CoIR CodeSearchNet parquet layout is not directly packaged as natural-language-query -> code corpus. For Phase A the loader derives a code-retrieval view from the official public test qrels by:
- indexing Python code documents from the test-partition query parquet
- using the matched public corpus text row as the natural-language query
- preserving the exact qrels mapping in metadata for auditing

This gives a deterministic public code-retrieval harness for the Python Phase A gate while keeping the official test-pairing source of truth.

Development commands
```bash
make test
make fixture-bloom
make bench-python-quickcheck
make bench-all
```
