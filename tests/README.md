# M2M test suite

What is actually in this directory and how to run it. The previous version of this file
described a `unit/`, `integration/` and `benchmarks/` tree, LlamaIndex/MCP tests, a
`pytest discover` command and ">90% coverage", none of which existed [audit r3a-13].

## Structure

Flat: 36 `test_*.py` files in `tests/`, no subdirectories. `tests/test_audit_*.py` are the
regression tests of the 2026-11 audit fix pass. `tests/validate_vulkan.py` is a manual script
and is not collected by pytest.

## Running

From the repository root. The pytest configuration lives in `pyproject.toml`
(`[tool.pytest.ini_options]`) and puts `src/` on `sys.path`, so the package does not need to
be installed. `--strict-markers` and `--strict-config` are on: an unregistered marker fails
the run.

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1   # ~2x faster here; BLAS spin-waits otherwise
pytest -q -m "not slow and not integration and not gpu and not embeddings"
```

That `-m` expression is exactly what CI runs (`.github/workflows/ci.yml`, job `test`).

Measured on 2026-11-27 on the dev machine (Linux/WSL2, Python 3.12, 1 BLAS thread):

```
619 passed / 22 skipped / 41 deselected / 0 failed      (8.6 min wall time)
```

### Markers

| marker        | meaning                                                      | in CI      |
| ------------- | ------------------------------------------------------------ | ---------- |
| `slow`        | large ingestion / throughput tests (seconds to minutes)      | deselected |
| `integration` | needs optional dependencies or a running service             | deselected |
| `gpu`         | needs an NVIDIA GPU (CUDA backend)                           | deselected |
| `embeddings`  | needs embedding models (sentence-transformers, torch)         | deselected |

### The 22 skips (machine without GPU or local datasets)

| count | where                                                     | reason (`-ra` output)                                     |
| ----- | --------------------------------------------------------- | --------------------------------------------------------- |
| 13    | `test_cuda_backend.py`                                    | `CUDA not available (no NVIDIA GPU or PyTorch not installed)` |
| 5     | `test_search_batch_lsh.py` (4), `test_hnsw_diversity.py` (1) | `dataset real no disponible` (`/tmp/m2mvs/embeddings.npy`) |
| 2     | `test_gate_select_k.py`                                   | `dataset de validación ausente` (`/tmp/m2mvs/embeddings.npy`) |
| 2     | `test_sq8_spectral.py`                                    | `corpus real no disponible`                               |

All of them are explicit `skipif` / `skip` with a reason: nothing fails (or passes vacuously)
when a dataset or a GPU is missing.

**LangChain** (`test_langchain.py`, `test_langchain_full.py`, 11 tests): they run when
`langchain-core` is installed, as in the measurement above. GitHub CI installs `.[dev]`,
which does not include `langchain-core` yet, so there they are skipped through
`pytest.importorskip("langchain_core")`: 11 extra skips in CI until `langchain-core` is
added to the `dev` extra.

### The 41 deselected

Tests marked `slow`, `integration`, `gpu` or `embeddings`, among them the whole
`test_rag_dataset.py` module (`pytestmark = [pytest.mark.embeddings, ...]`) and the
`TestPerformance` class of `test_m2m_advanced.py` (`slow`). Run them on demand with
`pytest -m slow`, `pytest -m gpu`, etc.

## Coverage

```bash
pytest -m "not slow and not integration and not gpu and not embeddings" \
    --cov=m2m --cov-report=term
```

Measured with the CI selection: **70.57 %** (branch coverage; configuration in
`[tool.coverage.*]` of `pyproject.toml`). CI fails below **68 %** (`--cov-fail-under`
in `ci.yml` = the measured value rounded down, minus 2 points).

## Timing-dependent tests

None in the CI selection. `test_lsh_speedup` asserted a speedup ratio measured with
`time.time()` and failed under host load (it measured -1.35x once); it was replaced by
`test_lsh_matches_exact_scan`, a deterministic parity check of LSH against the exact scan.
The measurement lives in `benchmarks/bench_lsh.py` and runs weekly in
`.github/workflows/benchmark.yml`, which uploads the JSON as an artifact and sets no
thresholds (CI runners are too noisy for timing gates).
`TestPerformance::test_search_performance` still asserts `QPS > 20`, but it is marked `slow`
and never runs in CI.

## Lint and static analysis

```bash
pre-commit run --all-files        # black, isort, flake8 (.flake8), bandit (-c pyproject.toml), hygiene hooks
pre-commit run mypy --all-files   # manual only: mypy is not a gate yet
```

CI runs the same tools in the `lint` and `security` jobs of `.github/workflows/ci.yml`.
