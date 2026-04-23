from pathlib import Path

import pytest

from bench.contamination.bloom import BloomFilter
from bench.runners.run_benchmark import run_phase_a_quickcheck


def test_phase_a_quickcheck_requires_contamination_filter(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing.bin"

    with pytest.raises(FileNotFoundError, match="Contamination checks are mandatory"):
        run_phase_a_quickcheck(bloom_path=missing_path, sample_size=1, top_k=5, output_dir=tmp_path / "out")


def test_phase_a_quickcheck_rejects_bloom_without_canary(tmp_path: Path) -> None:
    wrong_bloom = BloomFilter.create(expected_items=10, false_positive_rate=0.01)
    wrong_bloom.add("deadbeef" * 8)
    bloom_path = tmp_path / "wrong.bin"
    wrong_bloom.save(bloom_path)

    with pytest.raises(ValueError, match="canary"):
        run_phase_a_quickcheck(bloom_path=bloom_path, sample_size=1, top_k=5, output_dir=tmp_path / "out")
