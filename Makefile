UV ?= uv
PYTHON ?= $(UV) run --extra dev python
PIP ?= $(PYTHON) -m pip

.PHONY: bootstrap test fixture-bloom bench-python-quickcheck bench-python bench-python-b5 bench-typescript-b3 bench-typescript-b5 bench-all clean

bootstrap:
	$(PIP) install -e '.[dev]'

fixture-bloom:
	PYTHONPATH=. $(PYTHON) -c "from bench.constants import BLOOM_CANARY_CODE, CORNSTACK_BLOOM_PATH; from bench.contamination.bloom import build_fixture_bloom; build_fixture_bloom(CORNSTACK_BLOOM_PATH, language='python', code_samples=[BLOOM_CANARY_CODE], expected_items=10000); print(f'fixture bloom ready at {CORNSTACK_BLOOM_PATH}')"

test:
	PYTHONPATH=. $(PYTHON) -m pytest -q

bench-python-quickcheck: fixture-bloom
	PYTHONPATH=. $(PYTHON) -m bench.runners.run_benchmark phase-a-quickcheck --bloom-path bench/cache/cornstack_bloom.bin --sample-size 100 --top-k 5 --output-dir bench-results/phase-a-python-quickcheck

bench-python: fixture-bloom
	PYTHONPATH=. $(PYTHON) -m bench.runners.run_benchmark phase-b-python-full --bloom-path bench/cache/cornstack_bloom.bin --sample-size 100 --top-k 10 --output-dir bench-results/phase-b-v3-python-full

bench-python-b5: fixture-bloom
	PYTHONPATH=. $(PYTHON) -m bench.runners.run_benchmark phase-b5-python-full --bloom-path bench/cache/cornstack_bloom.bin --sample-size 100 --top-k 10 --output-dir bench-results/phase-b5-python-full

bench-typescript-b3: fixture-bloom
	PYTHONPATH=. $(PYTHON) -m bench.runners.run_benchmark phase-b-typescript-full --bloom-path bench/cache/cornstack_bloom.bin --sample-size 100 --corpus-sample-size 1000 --top-k 10 --output-dir bench-results/phase-b-typescript-full

bench-typescript-b5: fixture-bloom
	PYTHONPATH=. $(PYTHON) -m bench.runners.run_benchmark phase-b5-typescript-full --bloom-path bench/cache/cornstack_bloom.bin --sample-size 100 --corpus-sample-size 1000 --top-k 10 --output-dir bench-results/phase-b5-typescript-full

bench-all: bench-python

clean:
	rm -rf .pytest_cache bench-results bench/cache
