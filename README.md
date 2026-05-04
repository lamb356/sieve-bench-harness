# SIEVE Bench Harness

Benchmark harness for evaluating SIEVE retrieval quality, latency, and metadata contracts across code-search datasets.

The harness keeps benchmark claims tied to explicit datasets, retriever configurations, and reproducible commands. It is not the product surface; it is the verification surface for SIEVE retrieval behavior.

## Current status

- Default dense backend for v1 reporting: `bge-small` / `BAAI/bge-small-en-v1.5`.
- Canonical semantic-hard-v1 manifests are included under `bench/data/semantic-hard-v1/`.
- The SIEVE CLI adapter preserves `retrieval_sources` from JSON output and derives legacy `layer` metadata from the first retrieval source when needed.
- An experimental custom encoder retriever is preserved in [`docs/custom-encoder-archive.md`](https://github.com/lamb356/sieve/blob/main/docs/custom-encoder-archive.md). v1 ships with bge-small as the default dense backend.

## MCP server

MCP server: design complete, implementation forthcoming. See [`docs/mcp-server-v1-design.md`](https://github.com/lamb356/sieve/blob/main/docs/mcp-server-v1-design.md) for the v1 surface.

## Retriever surface

Default reporting distinguishes:

- lexical baselines: ripgrep, BM25
- SIEVE CLI retriever
- default dense backend: bge-small-en-v1.5
- diagnostic/null baselines where configured: CodeBERT, UniXcoder
- larger/reference retrievers where configured: LateOn-Code-edge, LateOn-Code

## Run tests

```bash
uv run pytest -q
```

The latest cleanup gate for the pushed branch passed at `104 passed, 1 skipped`.

## Notes

- Benchmark rows should not be treated as interchangeable product claims. Each row has a role: lexical baseline, default dense backend, null/diagnostic baseline, or reference retriever.
- Do not publish local checkpoint paths or checkpoint artifacts from benchmark runs.
