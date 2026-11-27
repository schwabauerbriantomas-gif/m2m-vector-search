# Contributing

## Setup

```bash
pip install -e ".[dev]"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1   # small BLAS ops spin-wait otherwise
```

`.[dev]` installs the `api` and `langchain` extras and the tooling. `torch` and the Vulkan
bindings are **not** installed; the tests that need them are marked and deselected.

## Tests

```bash
pytest tests/ -m "not slow and not integration and not gpu and not embeddings"
```

That is the exact selection CI runs (`.github/workflows/ci.yml`). Do not add `-q`:
`addopts` already contains it and `-qq` suppresses the `N passed` line.

- New markers must be registered in `[tool.pytest.ini_options]` (`--strict-markers` is on).
- A bug fix comes with the regression test that fails without it. The audit regression
  tests live in `tests/test_audit_*.py` and cite the audit id in the test name or docstring.
- A behaviour change is documented in `CHANGELOG.md` under `[Unreleased]`.

## Style

```bash
black src/ tests/
isort src/ tests/
flake8 src/ tests/                     # config in .flake8
bandit -c pyproject.toml -r src/m2m -ll
pre-commit run --all-files             # mypy is manual
```

Line length is 100. `flake8 --select=E9,F63,F7,F82,F401` must stay clean on the files you
touch; the repository carries pre-existing warnings elsewhere.

## Public API rules

- Additive only: new methods, fields and optional parameters are fine; existing names keep
  their meaning unless the change is listed in `CHANGELOG.md`.
- New exceptions only for inputs that were silently invalid before (NaN, unknown operators).
- Never change authentication, CORS or bind-address defaults silently.

## Documentation

Every code block in `README.md` and `docs/*.md` must run as written. If you change a
signature, update `docs/API.md` and `docs/CONFIGURATION.md` in the same change.

## Pull requests

One topic per branch, commits prefixed with `fix(<scope>):`, `feat(<scope>):`,
`docs(<scope>):` or `test(<scope>):`, and a description that says how the change was
verified. Fill in the pull request template.
