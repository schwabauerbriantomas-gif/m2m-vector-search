# Contributing to M2M Vector Search

First off, thank you for considering contributing to M2M! 🎉

M2M is an open-source project, and we welcome contributions from the community. Whether you're fixing a bug, adding a feature, improving documentation, or suggesting optimizations, your help is appreciated.

---

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How Can I Contribute?](#how-can-i-contribute)
- [Development Setup](#development-setup)
- [Coding Standards](#coding-standards)
- [Pull Request Process](#pull-request-process)
- [Testing Guidelines](#testing-guidelines)
- [Documentation](#documentation)
- [Community](#community)

---

## 🤝 Code of Conduct

This project and everyone participating in it is governed by our Code of Conduct. By participating, you are expected to uphold this code. Please report unacceptable behavior to [maintainers](mailto:contact@neuralmemory.ai).

### Our Standards

- Be respectful and inclusive
- Welcome newcomers and help them get started
- Focus on what is best for the community
- Show empathy towards other community members
- Accept constructive criticism gracefully

---

## 🚀 How Can I Contribute?

### Reporting Bugs

Before creating bug reports, please check the issue list to avoid duplicates. When creating a bug report, please include:

**Bug Report Template**:
```markdown
**Description**
A clear description of what the bug is.

**To Reproduce**
Steps to reproduce the behavior:
1. Initialize M2M with config '...'
2. Add splats '....'
3. Run search '....'
4. See error

**Expected Behavior**
What you expected to happen.

**Actual Behavior**
What actually happened.

**Environment**
- OS: [e.g., Windows 10, Ubuntu 22.04]
- Python: [e.g., 3.12]
- PyTorch: [e.g., 2.0.1]
- Vulkan SDK: [e.g., 1.3.268]
- GPU: [e.g., AMD RX 6650 XT]

**Code Sample**
```python
# Minimal reproducible example
from m2m import M2MConfig, create_m2m
# ...
```

**Additional Context**
Add any other context about the problem here.
```

---

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. When creating an enhancement suggestion, include:

**Feature Request Template**:
```markdown
**Is your feature request related to a problem?**
A clear description of what the problem is.

**Describe the solution you'd like**
A clear description of what you want to happen.

**Describe alternatives you've considered**
Any alternative solutions or features you've considered.

**Use Case**
Why would this feature be useful? Who would benefit?

**Additional Context**
Add any other context, code examples, or references.
```

---

### Pull Requests

- Fill in the required template
- Do not include issue numbers in the PR title
- Include screenshots and animated GIFs in your pull request whenever possible
- Follow the coding standards
- Include tests for new functionality
- Update documentation for changed functionality
- End all files with a newline

---

## 💻 Development Setup

### Prerequisites

```bash
# Python 3.8+
python --version

# Git
git --version

# Vulkan SDK (optional, for GPU acceleration)
# https://vulkan.lunarg.com/
```

### Fork and Clone

```bash
# Fork the repository on GitHub, then:
git clone https://github.com/YOUR_USERNAME/m2m-vector-search.git
cd m2m-vector-search

# Add upstream remote
git remote add upstream https://github.com/schwabauerbriantomas-gif/m2m-vector-search.git
```

### Create Virtual Environment

```bash
# Create venv
python -m venv venv

# Activate
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install dev dependencies
pip install -r requirements-dev.txt
```

### Install Pre-commit Hooks

```bash
# Install pre-commit
pip install pre-commit

# Install hooks
pre-commit install
```

### Run Tests

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest --cov=m2m tests/

# Run specific test
pytest tests/test_splats.py -v
```

### Generate Charts (for documentation updates)

```bash
# Generate performance charts
python scripts/generate_charts.py
```

---

## 📏 Coding Standards

### Python Style Guide

We follow PEP 8 with some modifications:

```python
# Use 4 spaces for indentation
def function(arg1, arg2):
    pass

# Use descriptive variable names
splat_count = 10000  # Good
x = 10000  # Bad

# Use type hints
def search(query: torch.Tensor, k: int = 64) -> Tuple[torch.Tensor, torch.Tensor]:
    """Search for k nearest neighbors."""
    pass

# Use docstrings (Google style)
def add_splats(embeddings: torch.Tensor) -> int:
    """Add Gaussian splats to the store.
    
    Args:
        embeddings: Tensor of shape (N, 640) with embeddings.
    
    Returns:
        Number of splats successfully added.
    
    Raises:
        ValueError: If embeddings are not 640-dimensional.
    """
    pass

# Maximum line length: 100 characters
long_variable_name = some_function(arg1, arg2, arg3, arg4, arg5, 
                                    arg6, arg7, arg8)

# Use f-strings for formatting
message = f"Added {n_splats} splats to store"
```

### Code Formatting

We use the following tools:

```bash
# Format code with Black
black m2m/ tests/

# Sort imports with isort
isort m2m/ tests/

# Lint with flake8
flake8 m2m/ tests/

# Type check with mypy
mypy m2m/
```

### File Organization

```
m2m/
├── __init__.py          # Public API exports
├── config.py             # Configuration dataclass
├── splats.py             # SplatStore implementation
├── hrm2_engine.py        # Hierarchical search engine
├── memory.py             # 3-tier memory manager
├── gpu_vector_index.py   # GPU acceleration (GPUVectorIndex, HierarchicalGPUSearch)
├── geometry.py           # Riemannian operations
└── energy.py             # Energy functions

tests/
├── test_splats.py        # SplatStore tests
├── test_hrm2.py          # HRM2 engine tests
├── test_memory.py        # Memory manager tests
└── test_integration.py   # Integration tests
```

---

## 🔄 Pull Request Process

### 1. Create a Branch

```bash
# Update main
git checkout main
git pull upstream main

# Create feature branch
git checkout -b feature/your-feature-name

# Or for bug fixes
git checkout -b fix/issue-number-description
```

### 2. Make Changes

```bash
# Make your changes
# ...

# Run tests
pytest tests/

# Format code
black m2m/ tests/
isort m2m/ tests/

# Check linting
flake8 m2m/ tests/
```

### 3. Commit Changes

We follow [Conventional Commits](https://www.conventionalcommits.org/):

```bash
# Feature
git commit -m "feat: add FAISS-GPU support for HRM2"

# Bug fix
git commit -m "fix: correct Riemannian gradient projection"

# Documentation
git commit -m "docs: update installation instructions"

# Performance
git commit -m "perf: optimize KNN search with Numba JIT"

# Refactor
git commit -m "refactor: simplify memory hierarchy logic"

# Test
git commit -m "test: add edge cases for splat normalization"

# Chore
git commit -m "chore: update dependencies"
```

### 4. Push and Create PR

```bash
# Push to your fork
git push origin feature/your-feature-name

# Create Pull Request on GitHub
```

### 5. PR Checklist

Before submitting, ensure:

- [ ] Code follows the style guidelines
- [ ] All tests pass
- [ ] New tests added for new functionality
- [ ] Documentation updated
- [ ] Commit messages follow conventional commits
- [ ] PR description clearly describes changes
- [ ] Linked to relevant issues

### 6. Review Process

1. Maintainers will review your PR
2. Address review feedback
3. Once approved, a maintainer will merge your PR

---

## 🧪 Testing Guidelines

### Test Structure

```python
# tests/test_splats.py
import pytest
import torch
from m2m import M2MConfig, SplatStore

class TestSplatStore:
    """Tests for SplatStore functionality."""
    
    @pytest.fixture
    def config(self):
        return M2MConfig(device='cpu', max_splats=1000)
    
    @pytest.fixture
    def store(self, config):
        return SplatStore(config)
    
    def test_add_splat(self, store):
        """Test adding a single splat."""
        embedding = torch.randn(640)
        result = store.add_splat(embedding)
        assert result == True
        assert store.n_active == 1
    
    def test_add_splat_capacity(self, config):
        """Test splat capacity limit."""
        config.max_splats = 10
        store = SplatStore(config)
        
        for i in range(15):
            embedding = torch.randn(640)
            store.add_splat(embedding)
        
        assert store.n_active == 10
    
    @pytest.mark.parametrize("k", [1, 10, 64])
    def test_search_k_values(self, store, k):
        """Test search with different k values."""
        embeddings = torch.randn(100, 640)
        store.add_splats(embeddings)
        
        query = torch.randn(1, 640)
        results = store.find_neighbors(query, k=k)
        
        assert len(results) == k
```

### Run Tests

```bash
# All tests
pytest tests/

# Specific test file
pytest tests/test_splats.py

# Specific test class
pytest tests/test_splats.py::TestSplatStore

# Specific test
pytest tests/test_splats.py::TestSplatStore::test_add_splat

# With coverage
pytest --cov=m2m --cov-report=html tests/

# Verbose
pytest -v tests/

# Parallel
pytest -n 4 tests/
```

### Benchmark Tests

```bash
# Run benchmarks
python benchmarks/benchmark_m2m.py --n-splats 100000

# Generate report
python benchmarks/benchmark_m2m.py --report
```

---

## 📚 Documentation

### Code Documentation

- All public functions must have docstrings
- Use Google-style docstrings
- Include type hints
- Provide usage examples

```python
def search(query: torch.Tensor, k: int = 64) -> SearchResults:
    """Search for k nearest neighbors.
    
    Performs hierarchical search through HRM2 clustering
    to find the k most similar splats to the query vector.
    
    Args:
        query: Query embedding of shape (1, 640) or (640,).
            Must be L2-normalized to unit sphere.
        k: Number of nearest neighbors to retrieve.
            Must be positive and <= max_splats.
    
    Returns:
        SearchResults containing:
            - neighbors_mu: Neighbor embeddings (k, 640)
            - neighbors_alpha: Neighbor amplitudes (k,)
            - neighbors_kappa: Neighbor concentrations (k,)
    
    Raises:
        ValueError: If query is not 640-dimensional.
        ValueError: If k <= 0 or k > max_splats.
    
    Example:
        >>> config = M2MConfig(device='cpu')
        >>> m2m = M2MEngine(config)
        >>> embeddings = torch.randn(1000, 640)
        >>> m2m.add_splats(embeddings)
        >>> query = torch.randn(1, 640)
        >>> results = m2m.search(query, k=10)
        >>> print(results.neighbors_mu.shape)
        torch.Size([10, 640])
    
    Note:
        Search is performed using HRM2 hierarchical clustering
        for O(log N) complexity instead of O(N) linear search.
    """
    pass
```

### README Updates

If you add or change functionality, update the relevant sections in README.md:

- Features
- API Reference
- Benchmarks (if performance-related)
- Quick Start (if usage changes)

### Generating Charts

If you modify benchmark-related code, regenerate charts:

```bash
python scripts/generate_charts.py
```

---

## 🌍 Community

### Getting Help

- **GitHub Issues**: For bug reports and feature requests
- **Discussions**: For questions and general discussion
- **Discord**: [Join our community](https://discord.com/invite/clawd)

### Recognition

Contributors are recognized in:

- GitHub Contributors page
- Release notes for significant contributions
- Community highlights

---

## 📄 License

By contributing to M2M, you agree that your contributions will be licensed under the Apache License 2.0.

---

## 🙏 Thank You!

Thank you for taking the time to contribute! Your efforts help make M2M better for everyone.

---

**Questions?** Feel free to reach out to the maintainers or open a discussion on GitHub.

*Last updated: 2026-02-23*
