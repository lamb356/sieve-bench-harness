from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
BENCH_RESULTS_DIR = REPO_ROOT / "bench-results"
BENCH_CACHE_DIR = REPO_ROOT / "bench" / "cache"
CORNSTACK_BLOOM_PATH = BENCH_CACHE_DIR / "cornstack_bloom.bin"
PHASE_A_RESULTS_DIR = BENCH_RESULTS_DIR / "phase-a-python-quickcheck"

GLOBAL_RANDOM_SEED = 1337
QUICKCHECK_SAMPLE_SIZE = 100
QUICKCHECK_TOP_K = 5
QUICKCHECK_OUTPUT_JSON = PHASE_A_RESULTS_DIR / "results.json"
QUICKCHECK_OUTPUT_MD = PHASE_A_RESULTS_DIR / "hero-table.md"
RIPGREP_INDEX_DIR = BENCH_CACHE_DIR / "indexes" / "ripgrep" / "coir-python-quickcheck"

COIR_SOURCE_NAME = "coir"
COIR_DATASET_ID = "CoIR-Retrieval/CodeSearchNet"
COIR_DATASET_REVISION = "25e0292562b7bee26dd9b2d83a03981795862c77"
COIR_LANGUAGE = "python"
COIR_CORPUS_ID = "coir-python-test-corpus"
COIR_QUERIES_PATH = "python-queries/queries-00000-of-00001.parquet"
COIR_CORPUS_PATH = "python-corpus/corpus-00000-of-00001.parquet"
COIR_QRELS_TEST_PATH = "python-qrels/test-00000-of-00001.parquet"

BLOOM_EXPECTED_ITEMS = 200_000_000
BLOOM_FALSE_POSITIVE_RATE = 0.01
BLOOM_CANARY_CODE = "def contamination_fixture_canary() -> int:\n    return 7\n"

RECALL_KS: tuple[int, ...] = (1, 5, 10)
NDCG_K = 10
MRR_K = 10
