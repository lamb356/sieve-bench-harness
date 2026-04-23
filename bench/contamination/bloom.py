from __future__ import annotations

import hashlib
import json
import math
import struct
from dataclasses import dataclass
from pathlib import Path

from bitarray import bitarray

from bench.contamination.normalize import normalize_code


@dataclass
class BloomFilter:
    expected_items: int
    false_positive_rate: float
    num_bits: int
    num_hashes: int
    bits: bitarray

    @classmethod
    def create(cls, *, expected_items: int, false_positive_rate: float) -> "BloomFilter":
        if expected_items <= 0:
            raise ValueError("expected_items must be positive")
        if not (0.0 < false_positive_rate < 1.0):
            raise ValueError("false_positive_rate must be between 0 and 1")
        num_bits = math.ceil(-(expected_items * math.log(false_positive_rate)) / (math.log(2) ** 2))
        num_hashes = max(1, round((num_bits / expected_items) * math.log(2)))
        bits = bitarray(num_bits)
        bits.setall(0)
        return cls(
            expected_items=expected_items,
            false_positive_rate=false_positive_rate,
            num_bits=num_bits,
            num_hashes=num_hashes,
            bits=bits,
        )

    def _hash_indexes(self, value: str) -> list[int]:
        digest = hashlib.sha256(value.encode("utf-8")).digest()
        head = int.from_bytes(digest[:16], "big")
        tail = int.from_bytes(digest[16:], "big") or 1
        return [int((head + offset * tail) % self.num_bits) for offset in range(self.num_hashes)]

    def add(self, value: str) -> None:
        for index in self._hash_indexes(value):
            self.bits[index] = 1

    def __contains__(self, value: str) -> bool:
        return all(self.bits[index] for index in self._hash_indexes(value))

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        header = json.dumps(
            {
                "expected_items": self.expected_items,
                "false_positive_rate": self.false_positive_rate,
                "num_bits": self.num_bits,
                "num_hashes": self.num_hashes,
            },
            sort_keys=True,
        ).encode("utf-8")
        with path.open("wb") as handle:
            handle.write(struct.pack(">I", len(header)))
            handle.write(header)
            self.bits.tofile(handle)

    @classmethod
    def load(cls, path: Path) -> "BloomFilter":
        with path.open("rb") as handle:
            header_size = struct.unpack(">I", handle.read(4))[0]
            header = json.loads(handle.read(header_size).decode("utf-8"))
            bits = bitarray()
            bits.fromfile(handle)
        expected_num_bits = int(header["num_bits"])
        if len(bits) < expected_num_bits:
            raise ValueError("Bloom filter file is truncated")
        if len(bits) > expected_num_bits:
            del bits[expected_num_bits:]
        return cls(
            expected_items=int(header["expected_items"]),
            false_positive_rate=float(header["false_positive_rate"]),
            num_bits=expected_num_bits,
            num_hashes=int(header["num_hashes"]),
            bits=bits,
        )


def normalized_code_hash(code: str, *, language: str) -> str:
    normalized = normalize_code(code, language=language)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def assert_canary_membership(bloom: BloomFilter, *, language: str, canary_code: str) -> None:
    canary_hash = normalized_code_hash(canary_code, language=language)
    if canary_hash not in bloom:
        raise ValueError("Bloom filter verification failed: expected canary hash is missing")


def build_fixture_bloom(path: Path, *, language: str, code_samples: list[str], expected_items: int = 3) -> BloomFilter:
    bloom = BloomFilter.create(expected_items=max(expected_items, len(code_samples)), false_positive_rate=0.01)
    for sample in code_samples:
        bloom.add(normalized_code_hash(sample, language=language))
    bloom.save(path)
    return bloom
