# M2M Vector Search Engine

[![Python](https://img.shields.io/badge/python-3.8%2B-brightgreen.svg)](https://python.org)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

> **Machine-to-Memory (M2M) Engine & Gaussian Splat Vector Store**
>
> A vector database with hierarchical retrieval for local-first applications.

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Architecture](#-architecture)
- [Benchmarks](#-benchmarks)
- [When to Use](#-when-to-use)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [API Reference](#-api-reference)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🎯 Overview

**M2M Vector Search** is a vector database built on Gaussian Splats with hierarchical retrieval (HRM2) and 3-tier memory management.

### Key Features

- **Hierarchical Retrieval**: HRM2 clustering for efficient searches
- **3-Tier Memory**: VRAM (Hot) → RAM (Warm) → SSD (Cold)
- **Gaussian Splats**: Full representation (μ, α, κ)
- **RAG Compatible**: LangChain and LlamaIndex integrations
- **Local-First**: No cloud dependencies

---

## ✅ When to Use

### Works Well With

| Data Type | Characteristics | Expected Performance |
|-----------|-----------------|---------------------|
| **Images** | Natural clusters in feature space | Good speedup |
| **Geolocation** | Spatial clustering | Good speedup |
| **Audio features** | Pattern-based clustering | Moderate speedup |

### Does NOT Work Well With

| Data Type | Characteristics | Recommendation |
|-----------|-----------------|----------------|
| **Text embeddings** (uniform) | No natural clusters | Use Linear Scan or FAISS |
| **GloVe/Word2Vec** | Gaussian distribution | Use Linear Scan or FAISS |
| **Sentence embeddings** | Uniform in hypersphere | Use Linear Scan or FAISS |

> ⚠️ **Important**: For uniform embeddings like DBpedia, Linear Scan is faster than any indexing method. See [METHODOLOGY_CONCLUSIONS.md](METHODOLOGY_CONCLUSIONS.md) for full analysis.

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

| Parameter | Value |
|-----------|-------|
| **CPU** | AMD Ryzen 5 3400G |
| **RAM** | 32GB DDR4-3200 |
| **Dataset** | DBpedia (OpenAI embeddings) |
| **Vectors** | 10,000 |
| **Dimensions** | 640D |

### Results Summary

> ⚠️ **For uniform text embeddings, Linear Scan is the best option.**

| Method | Recall | Latency | Throughput | Verdict |
|--------|--------|---------|------------|---------|
| **Linear Scan** | 100% | ~24ms | ~42 QPS | ✅ **Best for text** |
| HETD | 100% | ~46ms | ~22 QPS | ❌ Slower |
| Enhanced Transformer | 95% | ~46ms | ~22 QPS | ❌ Slower |
| M2M Resonant | 46% | ~8ms | ~125 QPS | ❌ Low recall |

### Dataset Analysis (DBpedia)

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **Silhouette Score** | -0.0048 | No natural clusters |
| **Coefficient of Variation** | 0.085 | Very uniform distribution |
| **Cluster Overlap** | 5.5x | Complete overlap |

**Conclusion**: DBpedia embeddings are uniformly distributed with no cluster structure. Indexing methods add overhead without benefit.

---

## 📈 Performance by Data Type

### Structured Data (Recommended)

For data with natural clusters (images, geolocation, etc.):

```
Expected: 2-10x speedup with >95% recall
```

### Uniform Data (Not Recommended)

For text embeddings without cluster structure:

```
Expected: 0.5-1.0x speedup (often slower)
Recommendation: Use Linear Scan or FAISS/HNSW
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

### Check if M2M is Right for Your Data

```python
from sklearn.metrics import silhouette_score
import numpy as np

def should_use_m2m(vectors, sample_size=1000):
    """Check if M2M is appropriate for your data."""
    
    # Sample for speed
    idx = np.random.choice(len(vectors), min(sample_size, len(vectors)), replace=False)
    sample = vectors[idx]
    
    # Quick clustering
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=20, n_init=1)
    labels = kmeans.fit_predict(sample)
    
    # Calculate silhouette
    score = silhouette_score(sample, labels)
    
    # Calculate distance variance
    from scipy.spatial.distance import pdist
    distances = pdist(sample[:100])
    cv = np.std(distances) / np.mean(distances)
    
    print(f"Silhouette Score: {score:.4f}")
    print(f"CV of distances: {cv:.4f}")
    
    if score > 0.2 and cv > 0.2:
        print("✅ M2M should work well")
        return True
    else:
        print("❌ Use Linear Scan or FAISS instead")
        return False
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
    enable_vulkan: bool = False       # Enable GPU acceleration
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

---

## 📄 License

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for details.

---

## 📚 References

- **GitHub**: [m2m-vector-search](https://github.com/schwabauerbriantomas-gif/m2m-vector-search)
- **Methodology Conclusions**: [METHODOLOGY_CONCLUSIONS.md](METHODOLOGY_CONCLUSIONS.md)
- **Issues**: Bug reports and feature requests

---

## 🔬 Methodology

All benchmarks documented with:

- **Test environment**: CPU, Python 3.12
- **Dataset**: DBpedia (OpenAI text-embedding-3-large)
- **Reproducibility**: Scripts in `tests/`
- **Honest reporting**: Both successes and limitations

---

**Built for local-first vector search**

*M2M: Machine-to-Memory*
