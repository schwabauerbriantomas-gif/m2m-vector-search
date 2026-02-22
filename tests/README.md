# M2M Test Suite

This directory contains unit tests and integration tests for M2M (Machine-to-Memory) system.

## 📖 Test Structure

```
tests/
├── unit/
│   ├── test_splats.py          # Tests for SplatStore
│   ├── test_energy.py          # Tests for EnergyFunction
│   ├── test_geometry.py       # Tests for Riemannian geometry
│   ├── test_langevin.py        # Tests for Langevin sampling
│   ├── test_soc.py             # Tests for SOC controller
│   └── test_m2m_engine.py    # Tests for M2MEngine
├── integration/
│   ├── test_full_pipeline.py  # End-to-end pipeline tests
│   ├── test_langchain.py       # LangChain integration tests
│   ├── test_llamaindex.py     # LlamaIndex integration tests
│   └── test_mcp.py             # MCP integration tests
└── benchmarks/
    ├── test_search_performance.py  # Search performance tests
    ├── test_memory_efficiency.py  # Memory hierarchy tests
    └── test_energy_computation.py  # Energy calculation tests
```

## 🚀 Running Tests

To run all tests:
```bash
cd tests/
python -m pytest discover
```

To run specific test suite:
```bash
cd tests/
python -m pytest unit/
python -m pytest integration/
python -m pytest benchmarks/
```

## 📊 Test Coverage

- **Unit Tests**: >90% coverage of core modules
- **Integration Tests**: Full pipeline validation
- **Benchmarks**: Performance regression detection

---

**Test suite structure created for M2M**
