# sieve-bench-harness

## Methodology

SIEVE is designed as a local-first, small-model code retrieval system for real AI coding-agent deployment: fast local search, consumer-GPU/CPU practicality, a tiny binary/parameter footprint, and sub-millisecond-class query latency for the eventual SIEVE implementation. The benchmark therefore evaluates the quality × latency × memory × size Pareto frontier instead of treating raw Recall@5 against arbitrarily large models as the only objective.

The hero table is intentionally size-matched. LateOn-Code-edge is the primary retrieval-trained competitor because it is a 17M-parameter PyLate/ColBERT model, close enough to SIEVE's 4.2M-parameter class to make a deployment-relevant comparison. Larger systems can be useful references, but they are not the pitch surface for a local-first agent retriever.

LateOn-Code-edge, LateOn-Code, and SIEVE share a training-recipe family: CoRNStack-style pretraining plus NV-Retriever-style hard-negative mining. Comparisons against the LateOn family should therefore be read as architecture-to-architecture comparisons under a related recipe family, not as independent-baseline claims.

The extended table exists for reference only. CodeBERT is a NULL BASELINE: base `microsoft/codebert-base` features without retrieval fine-tuning, scored as an independent dual encoder with cosine similarity, and expected to stay near random. LateOn-Code is a 149M-parameter SOTA-at-larger-scale ceiling reference and is not a fair apples-to-apples competitor for SIEVE's 4.2M-parameter deployment class.

Benchmark quality claims rely on the hero table, not cherry-picked rows from the extended table. The Role column exists to make each row's purpose explicit for reviewers.

LateOn-Code-edge and LateOn-Code are evaluated in Phase B v3 with brute-force multi-vector MaxSim over the Phase B corpus. This is intentionally simple and auditable at this corpus size, but it is much slower than single-vector cosine; production LateOn deployments would use PLAID/FastPLAID indexing, which is not benchmarked here.

Diagnostic tooling used to validate retriever implementations lives under `bench/diagnostics/`. The `retriever_health.py` diagnostic was used to inspect CodeBERT/UniXcoder tokenization and caught the UniXcoder missing-`<encoder-only>` formatting bug that Phase B v2 fixes. Phase B v3 keeps the v2 retriever quality surface intact while fixing per-retriever CPU memory reporting with isolated subprocess delta-RSS measurement.

Production-grade benchmark harness for public multi-language code retrieval evaluation.

Current status: Phase B v3 implemented; Phase B.5 adds the raw-surface/full-eval CodeSearchNet Python route.

Phase B v3 scope
- Python only on the same CoIR benchmark surface as Phase A/B v1/v2
- Seven retrievers: ripgrep, BM25, CodeBERT null baseline, UniXcoder with `<encoder-only>` formatting, LateOn-Code-edge, LateOn-Code, SIEVE
- Retrieval metrics: Recall@1/5/10, MRR@10, NDCG@10
- Performance metrics: p50/p95/p99 latency, throughput, index build time, per-retriever memory footprint
- CPU memory is reported as isolated subprocess delta RSS; CUDA retrievers keep `torch.cuda.max_memory_allocated()` measurement
- Full report outputs: JSON, markdown hero + extended tables, CSV audit data, optional HTML view
- Normalized-code benchmark surface: all Phase B v3 retrievers index `document.index_text`

Phase B.5 scope
- Python CodeSearchNet full eval distribution using the same retriever set, metrics, and memory methodology as Phase B v3
- Raw-surface query distribution: mixed semantic-hard and raw/literal queries rather than the Phase B v3 hard-slice interpretation
- Separate output directory: `bench-results/phase-b5-python-full/`
- Separate ripgrep index cache so Phase B v3 artifacts remain reproducible
- Phase B.5 records raw-surface findings as observational and does not apply Phase B v3 semantic-hard ordering gates

Phase B v1/v2/v3/B.5 audit trail
- Phase B v1 artifacts remain at `bench-results/phase-b-python-full/` when present locally.
- Phase B v2 artifacts remain at `bench-results/phase-b-v2-python-full/` and are not overwritten by v3.
- Phase B v3 writes artifacts to `bench-results/phase-b-v3-python-full/`.
- Phase B.5 writes artifacts to `bench-results/phase-b5-python-full/`.

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

Phase B v3 full Python benchmark
```bash
make bench-python
```

Phase B v3 outputs
- `bench-results/phase-b-v3-python-full/results.json`
- `bench-results/phase-b-v3-python-full/benchmark-table.md`
- `bench-results/phase-b-v3-python-full/benchmark-full.csv`
- `bench-results/phase-b-v3-python-full/interactive.html`

Phase B.5 full-eval Python benchmark target
```bash
make bench-python-b5
```
The Make target matches Phase B v3's deterministic 100-query run for practical validation while drawing from the full eval distribution. To run the exhaustive 14,702-query full eval, invoke `python -m bench.runners.run_benchmark phase-b5-python-full` without `--sample-size`.

Phase B.5 outputs
- `bench-results/phase-b5-python-full/results.json`
- `bench-results/phase-b5-python-full/benchmark-table.md`
- `bench-results/phase-b5-python-full/benchmark-full.csv`
- `bench-results/phase-b5-python-full/interactive.html`

Retriever notes
- Model downloads are cached under `bench/cache/models/`.
- CodeBERT pin: `microsoft/codebert-base@3b0952feddeffad0063f274080e3c23d75e7eb39`; this is a null baseline using base pretrained features only.
- UniXcoder pin: `microsoft/unixcoder-base@5604afdc964f6c53782a6813140ade5216b99006`; Phase B v3 keeps the Phase B v2 `<encoder-only>` query/document wrapper.
- LateOn-Code-edge pin: `lightonai/LateOn-Code-edge@07ef20f406c86badca122464808f4cac2f6e4b25`.
- LateOn-Code pin: `lightonai/LateOn-Code@734b659a57935ef50562d79581c3ff1f8d825c93`.
- CodeBERT and UniXcoder use 512-token max context, document head+tail truncation, query head-only truncation, mean pooling, and cosine similarity.
- LateOn retrievers use PyLate multi-vector embeddings and brute-force MaxSim scoring in Phase B v3.

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
