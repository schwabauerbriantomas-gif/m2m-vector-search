# M2M Vector Search Engine

[![Python](https://img.shields.io/badge/python-3.8%2B-brightgreen.svg)](https://python.org)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

> **High-performance Machine-to-Memory (M2M) Engine & Gaussian Splat Vector Cloud**
>
> A vector database built on Gaussian Splats and Tier-Aware Memory (VRAM, RAM, SSD) with hierarchical retrieval.

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Architecture](#-architecture)
- [Benchmarks](#-benchmarks)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🎯 Overview

**M2M Vector Search** is a vector database built on **Gaussian Splats** and **Tier-Aware Memory** (VRAM, RAM, SSD) designed for local deployment with hierarchical retrieval capabilities.

### Key Capabilities

- **Hierarchical Retrieval**: HRM2 clustering for fast searches
- **Tiered Memory**: VRAM (Hot) → RAM (Warm) → SSD (Cold)
- **Gaussian Representations**: Stores μ, κ, α for energy-based physics
- **RAG Integration**: Compatible with LangChain and LlamaIndex
- **Local-First**: No cloud dependencies, runs entirely on local hardware

---

## 🌟 Features

### 1. Vector Search & RAG

- **HRM2 Clustering**: Hierarchical 3-level clustering (coarse/fine/splats)
- **Progressive Semantic LODs**: Adaptive routing with different precision levels
- **LangChain/LlamaIndex**: Native vector store integrations

### 2. Tiered Storage

- **3-Tier Memory**: VRAM → RAM → SSD for capacity and speed
- **SOC Controller**: Self-Organized Criticality for automatic consolidation
- **Gaussian Splats**: Full representation (mean, concentration, precision)

### 3. Integration

- **REST/gRPC APIs**: HTTP interfaces for external tools
- **Python SDK**: Easy-to-use native client
- **PyTorch Compatible**: IterableDataset export for training

---

## 🏗 Architecture

```
┌────────────────────────────────────────────────────────────┐
│                   Application Layer                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │  LangChain   │  │  LlamaIndex  │  │  REST/gRPC   │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
└────────────────────────────────────────────────────────────┘
                            │
┌───────────────────────────┴────────────────────────────────┐
│                   M2M Core Engine                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │  SplatStore  │  │  HRM2 Engine │  │ SOC Controller│     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
└────────────────────────────────────────────────────────────┘
                            │
┌───────────────────────────┴────────────────────────────────┐
│                   Memory Hierarchy                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │  Hot (VRAM)  │  │ Warm (RAM)   │  │ Cold (SSD)   │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
└────────────────────────────────────────────────────────────┘
```

### Components

| Component | File | Description |
|-----------|------|-------------|
| **SplatStore** | `splats.py` | Gaussian Splats storage (μ, α, κ) |
| **HRM2Engine** | `hrm2_engine.py` | Hierarchical 3-level clustering |
| **MemoryManager** | `memory.py` | 3-tier memory hierarchy |
| **SOC Controller** | `splats.py` | Self-Organized Criticality |
| **Geometry** | `geometry.py` | Riemannian operations |

---

## 📊 Benchmarks

### Test Environment

**Hardware Configuration**:
| Component | Specification |
|-----------|---------------|
| **CPU** | AMD Ryzen 5 3400G (4 cores, 3.7GHz) |
| **RAM** | 32GB DDR4-3200 |
| **OS** | Windows 10 |
| **Python** | 3.12 |

**Test Parameters**:
- **Dataset**: 100,000 random embeddings
- **Dimensions**: 640D
- **Queries**: 1,000 random queries
- **K**: 64 nearest neighbors
- **Device**: CPU

### Results (Validated)

| Metric | Linear Search | M2M (HRM2 + KNN) |
|--------|---------------|------------------|
| **Avg Latency** | 94.79ms | **0.99ms** |
| **Throughput** | 10.55 QPS | **1,012.77 QPS** |
| **Speedup** | 1x (baseline) | **32.4x** |

**Benchmark Command**:
```bash
python benchmarks/benchmark_m2m.py --n-splats 100000 --queries 1000 --k 64 --device cpu
```

**Full results**: See `benchmark_results.json`

### Reproduce Benchmarks

```bash
# Clone repository
git clone https://github.com/schwabauerbriantomas-gif/m2m-vector-search.git
cd m2m-vector-search

# Install dependencies
pip install torch numpy

# Run benchmark
python benchmarks/benchmark_m2m.py --n-splats 100000 --queries 1000 --k 64
```

---

## 🚀 Installation

### Prerequisites

- Python 3.8+
- PyTorch 2.0+
- NumPy 1.21+

### From Source

```bash
git clone https://github.com/schwabauerbriantomas-gif/m2m-vector-search.git
cd m2m-vector-search
pip install -r requirements.txt
```

---

## 🚀 Quick Start

### Basic Usage

```python
import torch
from m2m import M2MConfig, create_m2m

# Initialize M2M
m2m = create_m2m(M2MConfig(
    device='cpu',
    max_splats=100000
))

# Add embeddings
embeddings = torch.randn(10000, 640)
m2m.add_splats(embeddings)

# Search
query = torch.randn(1, 640)
results = m2m.search(query, k=10)
```

### RAG with LangChain

```python
from langchain.vectorstores import M2MVectorStore
from langchain.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings()
vectorstore = M2MVectorStore(
    embedding_function=embeddings.embed_query,
    splat_capacity=100000
)

# Add documents
vectorstore.add_texts(documents)

# Search
results = vectorstore.similarity_search("query", k=10)
```

---

## 📖 API Reference

### M2MConfig

```python
@dataclass
class M2MConfig:
    device: str = "cpu"               # Device: cpu/cuda
    latent_dim: int = 640             # Embedding dimension
    max_splats: int = 100000          # Maximum capacity
    knn_k: int = 64                   # K-nearest neighbors
    enable_3_tier_memory: bool = True # Enable VRAM/RAM/SSD
```

### M2MEngine Methods

```python
# Add splats
m2m.add_splats(embeddings: torch.Tensor) -> int

# Search
m2m.search(query: torch.Tensor, k: int = 64) -> Tuple[Tensor, Tensor, Tensor]

# Statistics
m2m.get_statistics() -> Dict[str, Any]
```

---

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Setup

```bash
git clone https://github.com/schwabauerbriantomas-gif/m2m-vector-search.git
cd m2m-vector-search
pip install -r requirements.txt
python benchmarks/benchmark_m2m.py --n-splats 100000
```

---

## 📄 License

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for details.

---

## 📚 References

- **GitHub**: [m2m-vector-search](https://github.com/schwabauerbriantomas-gif/m2m-vector-search)
- **Issues**: Bug reports and feature requests

---

**Built for local-first vector search**

*M2M: Machine-to-Memory for systems with persistent, long-term memory*
