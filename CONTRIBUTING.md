# Contributing to M2M Vector Search

Thank you for your interest in contributing! This guide covers everything you need to get started.

## Getting Started

### Prerequisites
- Python 3.11+
- pip
- Git
- (Optional) Vulkan-compatible GPU or NVIDIA GPU with CUDA

### Setup

```bash
git clone https://github.com/brianschwabauer/m2m-vector-search.git
cd m2m-vector-search
pip install -e ".[dev]"
```

### Run Tests

```bash
# All tests
pytest tests/ -v

# Specific test file
pytest tests/test_crud.py -v

# With coverage
pytest tests/ --cov=m2m --cov-report=term-missing
```

## Development Workflow

1. **Fork** the repository
2. **Create a branch** from `main`: `git checkout -b feature/my-feature`
3. **Make changes** with tests
4. **Run tests**: `pytest tests/ -v` (all 53 must pass)
5. **Run benchmarks** (if performance-related): `python -m benchmarks.run`
6. **Commit** with conventional commits format
7. **Push** and open a Pull Request

## Code Style

- Follow PEP 8
- Type hints are encouraged but not required
- Docstrings for public functions/classes
- No `print()` in production code — use `logging`
- No `__pycache__` in commits

## Commit Messages

Use [Conventional Commits](https://www.conventionalcommits.org/):
```
feat: add CUDA backend support
fix: resolve k=0 crash in query
docs: update README installation guide
test: add chaos testing for edge cases
perf: optimize HRM2 query with einsum
```

## Project Structure

```
src/m2m/
├── __init__.py          # SimpleVectorDB, AdvancedVectorDB
├── splats.py            # Gaussian Splat storage
├── hrm2_engine.py       # HRM2 index engine
├── gpu_vector_index.py  # Vulkan/CUDA GPU backend
├── storage.py           # Persistence layer
├── config.py            # Configuration
├── api/
│   ├── edge_api.py      # Edge node REST API
│   └── coordinator_api.py # Coordinator REST API
└── cluster/
    ├── router.py        # Query routing
    ├── aggregator.py    # Result aggregation (RRF)
    └── protocol.py      # Cluster protocol messages
```

## Reporting Issues

- Use GitHub Issues with the provided templates
- Include: OS, Python version, GPU, reproduction steps
- For bugs: include minimal reproducible code
- For features: describe the use case

## Benchmarking

If you change anything performance-related, run benchmarks and report results:

```bash
python -m benchmarks.run
```

**Important:** Only report real measured data. Never fabricate or estimate benchmark results.

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
