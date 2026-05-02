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

Current status: Phase B v3 implemented; Phase B.5 adds full-eval CodeSearchNet Python route coverage; canonical TypeScript Phase B v3/B.5 routes are available through a pinned public TypeScript `.ts` code/docstring dataset; Go Phase B v3/B.5 routes use official CoIR/CodeSearchNet Go test qrels; Rust Phase B v3/B.5 routes are shipped as a caveated pinned `.rs` docstring/code-pair eval because no official CodeSearchNet/CoIR/CornStack Rust retrieval qrels were identifiable.

Phase B v3 scope
- Python on the same CoIR benchmark surface as Phase A/B v1/v2
- Canonical TypeScript route on `Shuu12121/typescript-treesitter-dedupe-filtered-datasetsV2` when invoked via `phase-b-typescript-full` / `make bench-typescript-b3`; no upstream CodeSearchNet/CoIR-style semantic-hard TypeScript split was identifiable, so this route preserves Phase B v3 retriever methodology over the public TypeScript test split. The repo-local generated semantic-hard subset lives under `bench/data/semantic-hard-v1/`.
- Go route on official CoIR/CodeSearchNet Go test qrels when invoked via `phase-b-go-full` / `make bench-go-b3`
- Rust route on `Shuu12121/rust-treesitter-dedupe-filtered-datasetsV2` when invoked via `phase-b-rust-full` / `make bench-rust-b3`; this is a caveated docstring-pair eval, not official retrieval-qrels coverage
- Seven retrievers: ripgrep, BM25, CodeBERT null baseline, UniXcoder with `<encoder-only>` formatting, LateOn-Code-edge, LateOn-Code, SIEVE
- Retrieval metrics: Recall@1/5/10, MRR@10, NDCG@10
- Performance metrics: p50/p95/p99 latency, throughput, index build time, per-retriever memory footprint
- CPU memory is reported as isolated subprocess delta RSS; CUDA retrievers keep `torch.cuda.max_memory_allocated()` measurement
- Full report outputs: JSON, markdown hero + extended tables, CSV audit data, optional HTML view
- Raw-code benchmark surface: all Phase B v3/B.5 retrievers index `document.code`; loader `document.index_text` remains metadata/cache material only.

Phase B.5 scope
- Python CodeSearchNet full eval distribution using the same retriever set, metrics, and memory methodology as Phase B v3
- Canonical TypeScript full eval distribution using the same retriever set, metrics, and memory methodology as Phase B v3/B.5 Python
- Go full eval distribution using official CoIR/CodeSearchNet Go test qrels with the same retriever set, metrics, and memory methodology
- Rust full eval distribution using pinned Rust `.rs` docstring/code pairs with explicit methodology caveat because official Rust retrieval qrels were not identifiable
- Full-eval query distribution: mixed semantic-hard and raw/literal queries rather than the Phase B v3 hard-slice interpretation
- Separate Python output directory: `bench-results/phase-b5-python-full/`
- Separate TypeScript output directory: `bench-results/phase-b5-typescript-full/`
- Separate Go output directory: `bench-results/phase-b5-go-full/`
- Separate Rust output directory: `bench-results/phase-b5-rust-full/`
- Separate ripgrep index caches so Phase B v3 artifacts remain reproducible
- Phase B.5 records full-eval findings as observational and does not apply Phase B v3 semantic-hard ordering gates

Phase B v1/v2/v3/B.5 audit trail
- Phase B v1 artifacts remain at `bench-results/phase-b-python-full/` when present locally.
- Phase B v2 artifacts remain at `bench-results/phase-b-v2-python-full/` and are not overwritten by v3.
- Phase B v3 writes artifacts to `bench-results/phase-b-v3-python-full/`.
- Phase B TypeScript writes artifacts to `bench-results/phase-b-typescript-full/`.
- Phase B Go writes artifacts to `bench-results/phase-b-go-full/`.
- Phase B Rust writes artifacts to `bench-results/phase-b-rust-full/`.
- Phase B.5 Python writes artifacts to `bench-results/phase-b5-python-full/`.
- Phase B.5 TypeScript writes artifacts to `bench-results/phase-b5-typescript-full/`.
- Phase B.5 Go writes artifacts to `bench-results/phase-b5-go-full/`.
- Phase B.5 Rust writes artifacts to `bench-results/phase-b5-rust-full/`.

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
- Go loader uses official CoIR/CodeSearchNet `go-qrels/test-00000-of-00001.parquet` with 8,122 qrel rows over the pinned revision
- TypeScript dataset id: `Shuu12121/typescript-treesitter-dedupe-filtered-datasetsV2`
- TypeScript pinned revision: `1e2fcd3764fb9126a33eaea58961925e667769f0`
- TypeScript source language: canonical TypeScript (`.ts`) with docstring/code pairs from permissively licensed public repositories
- TypeScript eval surface: test split only, 11,579 non-empty docstring/code query-document pairs; official split counts are train 328,457; validation 4,493; test 11,579
- TypeScript dataset license/provenance: Hugging Face card license `apache-2.0`; row-level licenses in the pinned test split are Apache-2.0, MIT, BSD-3-Clause, and ISC
- Rust dataset id: `Shuu12121/rust-treesitter-dedupe-filtered-datasetsV2`
- Rust pinned revision: `c0331761290f11fb428d7ac74cccda9fbac81fc2`
- Rust eval surface: test split only, 8,868 non-empty Rust `.rs` docstring/code query-document pairs; this is a caveated constructed eval because CodeSearchNet/CoIR/CornStack Rust retrieval qrels were not found
- Rust dataset license/provenance: Hugging Face card license `apache-2.0`; row-level licenses are preserved in metadata and reports

Canonical semantic-hard-v1 manifests
- Source of truth: `bench/data/semantic-hard-v1/{python,typescript,go,rust}.jsonl`.
- Each file contains exactly 300 full-eval rows where the harness `RipgrepRetriever` missed the paired code document in the top 5 (`ripgrep.recall@5 == 0.0`) against the full-language eval corpus.
- The schema records dataset/source provenance, query/document identifiers, query/code SHA-256 hashes, ripgrep metrics, and `provenance.source_metadata`.
- `bench.loaders.semantic_hard.load_semantic_hard_benchmark()` validates language, hashes, duplicate query/document/code/query identities, minimum row count, provenance fields, and corpus/query drift before constructing a `LoadedBenchmark`.

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

Phase B v3 TypeScript benchmark target
```bash
make bench-typescript-b3
```
The route uses the pinned `Shuu12121/typescript-treesitter-dedupe-filtered-datasetsV2` test split because no upstream CodeSearchNet/CoIR-style semantic-hard TypeScript qrel split was identifiable. The Make target uses a deterministic 100-query sample from the 11,579-query canonical TypeScript test split and bounds the corpus to sampled positives plus deterministic negatives for practical validation; omit `--corpus-sample-size` for a full-corpus run. For the generated repo-local semantic-hard subset, see `bench/data/semantic-hard-v1/typescript.jsonl`.

Phase B.5 TypeScript full-eval benchmark target
```bash
make bench-typescript-b5
```
The Make target draws a deterministic 100-query sample from the pinned TypeScript test split and bounds the corpus to sampled positives plus deterministic negatives for practical validation. To run the exhaustive 11,579-query/full-corpus eval, invoke `python -m bench.runners.run_benchmark phase-b5-typescript-full` without `--sample-size` or `--corpus-sample-size`.

Phase B v3 Go benchmark target
```bash
make bench-go-b3
```
The route uses the official CoIR/CodeSearchNet Go test qrels at the pinned CoIR revision. The Make target uses a deterministic 100-query sample from the 8,122-qrel Go test surface and bounds the corpus to sampled positives plus deterministic negatives for practical validation. Use `phase-b5-go-full` without `--sample-size` or `--corpus-sample-size` for the exhaustive 8,122-query/full-corpus Go run.

Phase B.5 Go full-eval benchmark target
```bash
make bench-go-b5
```
The Phase B.5 Go route uses the same official Go qrels and marks ordering gates observational, matching the Python/TypeScript B.5 convention.

Phase B v3 Rust benchmark target
```bash
make bench-rust-b3
```
The route uses pinned Rust `.rs` docstring/code pairs because no official CodeSearchNet/CoIR/CornStack Rust retrieval qrels were identifiable. Treat this as a cross-language signal and harness-validation track, not as an official retrieval-qrels benchmark.

Phase B.5 Rust full-eval benchmark target
```bash
make bench-rust-b5
```
The Make target draws a deterministic 100-query sample from the 8,868-row Rust test split and bounds the corpus to sampled positives plus deterministic negatives for practical validation. Invoke `python -m bench.runners.run_benchmark phase-b5-rust-full` without `--sample-size` or `--corpus-sample-size` for the exhaustive 8,868-query/full-corpus Rust run.

Phase B.5 outputs
- `bench-results/phase-b5-python-full/results.json`
- `bench-results/phase-b5-python-full/benchmark-table.md`
- `bench-results/phase-b5-python-full/benchmark-full.csv`
- `bench-results/phase-b5-python-full/interactive.html`
- `bench-results/phase-b-typescript-full/results.json`
- `bench-results/phase-b-typescript-full/benchmark-table.md`
- `bench-results/phase-b-typescript-full/benchmark-full.csv`
- `bench-results/phase-b-typescript-full/interactive.html`
- `bench-results/phase-b5-typescript-full/results.json`
- `bench-results/phase-b5-typescript-full/benchmark-table.md`
- `bench-results/phase-b5-typescript-full/benchmark-full.csv`
- `bench-results/phase-b5-typescript-full/interactive.html`
- `bench-results/phase-b-go-full/results.json`
- `bench-results/phase-b-go-full/benchmark-table.md`
- `bench-results/phase-b-go-full/benchmark-full.csv`
- `bench-results/phase-b-go-full/interactive.html`
- `bench-results/phase-b5-go-full/results.json`
- `bench-results/phase-b5-go-full/benchmark-table.md`
- `bench-results/phase-b5-go-full/benchmark-full.csv`
- `bench-results/phase-b5-go-full/interactive.html`
- `bench-results/phase-b-rust-full/results.json`
- `bench-results/phase-b-rust-full/benchmark-table.md`
- `bench-results/phase-b-rust-full/benchmark-full.csv`
- `bench-results/phase-b-rust-full/interactive.html`
- `bench-results/phase-b5-rust-full/results.json`
- `bench-results/phase-b5-rust-full/benchmark-table.md`
- `bench-results/phase-b5-rust-full/benchmark-full.csv`
- `bench-results/phase-b5-rust-full/interactive.html`

Retriever notes
- Model downloads are cached under `bench/cache/models/`.
- CodeBERT pin: `microsoft/codebert-base@3b0952feddeffad0063f274080e3c23d75e7eb39`; this is a null baseline using base pretrained features only.
- UniXcoder pin: `microsoft/unixcoder-base@5604afdc964f6c53782a6813140ade5216b99006`; Phase B v3 keeps the Phase B v2 `<encoder-only>` query/document wrapper.
- LateOn-Code-edge pin: `lightonai/LateOn-Code-edge@07ef20f406c86badca122464808f4cac2f6e4b25`.
- LateOn-Code pin: `lightonai/LateOn-Code@734b659a57935ef50562d79581c3ff1f8d825c93`.
- CodeBERT and UniXcoder use 512-token max context, document head+tail truncation, query head-only truncation, mean pooling, and cosine similarity.
- LateOn retrievers use PyLate multi-vector embeddings and brute-force MaxSim scoring in Phase B v3.
- TypeScript retriever config is intentionally minimal: ripgrep is text-agnostic; BM25 uses the same punctuation-aware raw-code tokenization; UniXcoder and LateOn-Code variants use their multilingual/code pretrained encoders without a language flag; CodeBERT remains a null baseline; SIEVE is labeled Phase 1 weights pending until real trained ONNX exports are supplied.
- Go retriever config follows the TypeScript/Python baseline set over official CoIR/CodeSearchNet Go qrels; UniXcoder is expected to be meaningful on Go because Go is one of its trained CodeSearchNet languages.
- Rust retriever config follows the same baseline set but is caveated: UniXcoder and LateOn may degrade because the public Rust route is not official CodeSearchNet/CoIR retrieval-qrels coverage, and SIEVE remains Phase 1 weights pending.

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
make bench-typescript-b3
make bench-typescript-b5
make bench-go-b3
make bench-go-b5
make bench-rust-b3
make bench-rust-b5
make bench-all
```
