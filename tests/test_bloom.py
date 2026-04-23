from bench.contamination.bloom import BloomFilter


KNOWN_HASHES = [
    "0" * 64,
    "1" * 64,
    "2" * 64,
]


def test_bloom_filter_contains_known_fixture_hashes() -> None:
    bloom = BloomFilter.create(expected_items=3, false_positive_rate=0.01)
    for item in KNOWN_HASHES:
        bloom.add(item)

    for item in KNOWN_HASHES:
        assert item in bloom

    assert ("3" * 64) not in bloom


def test_bloom_filter_round_trips_to_disk(tmp_path) -> None:
    bloom = BloomFilter.create(expected_items=3, false_positive_rate=0.01)
    for item in KNOWN_HASHES:
        bloom.add(item)

    bloom_path = tmp_path / "fixture.bin"
    bloom.save(bloom_path)
    reloaded = BloomFilter.load(bloom_path)

    for item in KNOWN_HASHES:
        assert item in reloaded
