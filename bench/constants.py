from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
BENCH_RESULTS_DIR = REPO_ROOT / "bench-results"
BENCH_CACHE_DIR = REPO_ROOT / "bench" / "cache"
BENCH_MODEL_CACHE_DIR = BENCH_CACHE_DIR / "models"
CORNSTACK_BLOOM_PATH = BENCH_CACHE_DIR / "cornstack_bloom.bin"
PHASE_A_RESULTS_DIR = BENCH_RESULTS_DIR / "phase-a-python-quickcheck"
PHASE_B_RESULTS_DIR = BENCH_RESULTS_DIR / "phase-b-python-full"
PHASE_B_V2_RESULTS_DIR = BENCH_RESULTS_DIR / "phase-b-v2-python-full"
PHASE_B_V3_RESULTS_DIR = BENCH_RESULTS_DIR / "phase-b-v3-python-full"
PHASE_B5_RESULTS_DIR = BENCH_RESULTS_DIR / "phase-b5-python-full"
PHASE_B_TYPESCRIPT_RESULTS_DIR = BENCH_RESULTS_DIR / "phase-b-typescript-full"
PHASE_B5_TYPESCRIPT_RESULTS_DIR = BENCH_RESULTS_DIR / "phase-b5-typescript-full"
PHASE_B5_CPU_TIMEOUT_SECONDS = 3_600.0

GLOBAL_RANDOM_SEED = 1337
QUICKCHECK_SAMPLE_SIZE = 100
QUICKCHECK_TOP_K = 5
QUICKCHECK_OUTPUT_JSON = PHASE_A_RESULTS_DIR / "results.json"
QUICKCHECK_OUTPUT_MD = PHASE_A_RESULTS_DIR / "hero-table.md"
RIPGREP_INDEX_DIR = BENCH_CACHE_DIR / "indexes" / "ripgrep" / "coir-python-quickcheck"
PHASE_B_RIPGREP_INDEX_DIR = BENCH_CACHE_DIR / "indexes" / "ripgrep" / "coir-python-full"
PHASE_B5_RIPGREP_INDEX_DIR = BENCH_CACHE_DIR / "indexes" / "ripgrep" / "coir-python-eval-full"
PHASE_B_TYPESCRIPT_RIPGREP_INDEX_DIR = BENCH_CACHE_DIR / "indexes" / "ripgrep" / "arkts-typescript-full"
PHASE_B5_TYPESCRIPT_RIPGREP_INDEX_DIR = BENCH_CACHE_DIR / "indexes" / "ripgrep" / "arkts-typescript-b5-full"

CODEBERT_MODEL_ID = "microsoft/codebert-base"
CODEBERT_MODEL_REVISION = "3b0952feddeffad0063f274080e3c23d75e7eb39"
UNIXCODER_MODEL_ID = "microsoft/unixcoder-base"
UNIXCODER_MODEL_REVISION = "5604afdc964f6c53782a6813140ade5216b99006"
LATEON_CODE_EDGE_MODEL_ID = "lightonai/LateOn-Code-edge"
LATEON_CODE_EDGE_MODEL_REVISION = "07ef20f406c86badca122464808f4cac2f6e4b25"
LATEON_CODE_MODEL_ID = "lightonai/LateOn-Code"
LATEON_CODE_MODEL_REVISION = "734b659a57935ef50562d79581c3ff1f8d825c93"

COIR_SOURCE_NAME = "coir"
COIR_DATASET_ID = "CoIR-Retrieval/CodeSearchNet"
COIR_DATASET_REVISION = "25e0292562b7bee26dd9b2d83a03981795862c77"
COIR_LANGUAGE = "python"
COIR_CORPUS_ID = "coir-python-test-corpus"
COIR_QUERIES_PATH = "python-queries/queries-00000-of-00001.parquet"
COIR_CORPUS_PATH = "python-corpus/corpus-00000-of-00001.parquet"
COIR_QRELS_TEST_PATH = "python-qrels/test-00000-of-00001.parquet"
PYTHON_EVAL_FULL = "python-eval-full"
PYTHON_EVAL_FULL_QUERY_COUNT = 14_702

TYPESCRIPT_SOURCE_NAME = "arkts-codesearch"
TYPESCRIPT_DATASET_ID = "hreyulog/arkts-code-docstring"
TYPESCRIPT_DATASET_REVISION = "b10cf6c85767455aef80fc02557614a408c183c1"
TYPESCRIPT_LANGUAGE = "typescript"
TYPESCRIPT_DATASET_LANGUAGE = "arkts"
TYPESCRIPT_FAMILY_NAME = "ArkTS"
TYPESCRIPT_CORPUS_ID = "arkts-codesearch-full-corpus"
TYPESCRIPT_EVAL_FULL = "typescript-arkts-full"
TYPESCRIPT_EVAL_FULL_QUERY_COUNT = 24_452
TYPESCRIPT_SPLIT_COUNTS = {"train": 19_561, "validation": 2_445, "test": 2_446}

BLOOM_EXPECTED_ITEMS = 200_000_000
BLOOM_FALSE_POSITIVE_RATE = 0.01
BLOOM_CANARY_CODE = "def contamination_fixture_canary() -> int:\n    return 7\n"

RECALL_KS: tuple[int, ...] = (1, 5, 10)
NDCG_K = 10
MRR_K = 10
