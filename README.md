<p align="center">
  <img src="https://img.shields.io/badge/version-2.2.0-blue" alt="Version">
  <img src="https://img.shields.io/badge/python-3.10%2B-green" alt="Python">
  <img src="https://img.shields.io/badge/license-AGPL--3.0-orange" alt="License">
  <img src="https://img.shields.io/badge/tests-394%20passed-success" alt="Tests">
  <img src="https://img.shields.io/badge/backends-CPU%20%7C%20CUDA%20%7C%20Vulkan-purple" alt="Backends">
</p>

<h1 align="center">M2M Vector Search</h1>

<p align="center">
  <strong>Machine-to-Memory</strong> &mdash; A vector search engine with probabilistic Gaussian Splats, online learning via feedback, and multi-backend GPU acceleration.
</p>

<p align="center">
  <a href="#quick-start">Quick Start</a> &bull;
  <a href="#features">Features</a> &bull;
  <a href="#architecture">Architecture</a> &bull;
  <a href="#semantic-memory-system">Memory</a> &bull;
  <a href="#benchmarks">Benchmarks</a>
</p>

---

## Features

| Feature | Description |
|---------|-------------|
| 🔮 **Gaussian Splats** | Probabilistic vector representation: `score(x, i) = αᵢ · exp(-κᵢ · ‖x − μᵢ‖²)` |
| 📈 **Online Learning** | Hebbian update rules adapt α, κ, μ from user feedback (relevant / not_relevant) |
| ⚡ **Multi-GPU Backend** | CPU, NVIDIA CUDA, and AMD Vulkan via a single API |
| 🧠 **Energy-Based Model** | Native uncertainty quantification via energy landscape |
| 🔍 **HRM2 Engine** | Hierarchical Routing with Mixture Models and adaptive probing |
| 🧹 **SOC Consolidation** | Self-Organized Criticality prunes low-α splats automatically |
| 🗄️ **Semantic Memory** | Hybrid BM25 + vector search with Reciprocal Rank Fusion |
| 🔗 **LangChain Ready** | Native Retriever interface |

---

## Quick Start

### Install

```bash
pip install m2m-vector-search
```

### Minimal Example

```python
from m2m import SimpleVectorDB
import numpy as np

# Create a vector database
db = SimpleVectorDB(latent_dim=128)

# Add vectors
vectors = np.random.randn(1000, 128).astype(np.float32)
db.add(vectors=vectors, ids=[f"doc_{i}" for i in range(1000)])

# Search
query = np.random.randn(128).astype(np.float32)
results = db.search(query, k=10)

for r in results:
    print(f"  {r.id}: score={r.score:.4f}")
```

### Gaussian Splats with Online Learning

This is the core differentiator. Each vector is a Gaussian Splat with three learnable parameters:

- **μ** (mean): position in embedding space
- **κ** (concentration): how sharply the splat responds — higher κ = more precise match required
- **α** (amplitude): how "important" the memory is — grows with positive feedback, decays with negative

```python
from m2m import AdvancedVectorDB
import numpy as np

db = AdvancedVectorDB(latent_dim=128, use_gaussian_splats=True)
vectors = np.random.randn(500, 128).astype(np.float32)
db.add(vectors=vectors, ids=[f"item_{i}" for i in range(500)])

# Search uses Gaussian scoring: α·exp(-κ·‖x−μ‖²)
query = np.random.randn(128).astype(np.float32)
results = db.search(query, k=10)

# Provide feedback — the system learns from it
db.feedback(
    query=query,
    relevant_ids=[results[0].id, results[1].id],
    irrelevant_ids=[results[8].id, results[9].id],
)

# Next search adapts: strong splats get promoted, weak ones demoted
results2 = db.search(query, k=10)
```

**Update rules (Hebbian + temporal decay):**

| Event | α (importance) | κ (concentration) | μ (position) |
|-------|:-:|:-:|:-:|
| Relevant feedback | `α += lr_α · α` | `κ += lr_κ · ‖x − μ‖⁻²` | `μ += lr_μ · (x − μ)` (drift toward query) |
| Irrelevant feedback | `α *= (1 − lr_α)` | `κ -= 0.5 · lr_κ` | — |
| Temporal decay | `α *= exp(-λ·Δt)` | — | — |

---

## Architecture

```
┌─────────────────────────────────────────────────┐
│                   REST API (FastAPI)             │
│            Collections · CRUD · Search           │
├─────────────────────────────────────────────────┤
│           SemanticMemoryDB / VectorDB            │
│      Hybrid Search · Fusion · Temporal Decay     │
├──────────┬──────────┬───────────┬───────────────┤
│  Splats  │  HRM2    │  EBM      │  SOC          │
│  (μ,κ,α) │  Engine  │  Energy   │  Consolidate  │
├──────────┴──────────┴───────────┴───────────────┤
│              Backend Layer (pluggable)           │
├─────────┬──────────┬──────────┬─────────────────┤
│   CPU   │  CUDA    │  Vulkan  │  Transformed    │
├─────────┴──────────┴──────────┴─────────────────┤
│              Storage Layer                       │
├─────────┬─────────────────┬─────────────────────┤
│  WAL    │  Persistence    │  GPUVectorIndex     │
└─────────┴─────────────────┴─────────────────────┘
```

---

## Semantic Memory System

```python
from m2m import SemanticMemoryDB
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("BAAI/bge-small-en-v1.5")
encoder = lambda text: model.encode(text, show_progress_bar=False)

mem = SemanticMemoryDB(
    encoder=encoder,
    latent_dim=384,
    fusion_method="rrf",           # Reciprocal Rank Fusion
    temporal_decay=True,            # Recent memories rank higher
    temporal_half_life_days=30.0,
    auto_categorize=True,
)

mem.store("User prefers dark mode for coding", metadata={"category": "preference"})
mem.store("We decided to use Qdrant for production", metadata={"category": "decision"})

results = mem.search("what did we decide about databases?", k=5)
```

### Hybrid Search

| Method | Tuning Required | Best For |
|--------|:-:|----------|
| **RRF** | No | General-purpose (recommended) |
| **Weighted** | Yes | Domain-specific with known priorities |
| `vector_only` | No | Pure semantic search |
| `bm25_only` | No | Pure keyword search |

---

## Benchmarks

> All data below is measured. No synthetic or estimated numbers.

**System:** AMD Ryzen 5 3400G, 16 GB RAM, NVIDIA RTX 3090, Python 3.12

### CPU Scale Progression (dim=640, k=64)

| Splats (N) | Linear Scan (ms) | M2M HRM2 (ms) | Speedup | QPS |
|:-:|:-:|:-:|:-:|:-:|
| 100 | 0.12 | — | — | 8,337 |
| 1,000 | 1.45 | — | — | 691 |
| 10,000 | 10.04 | — | — | 100 |
| 100,000 | 94.79 | 0.99 | **32.4x** | 1,013 |

At 100K splats, HRM2 hierarchical routing achieves **32x speedup** over linear scan.

### Gaussian Scoring Overhead

The Gaussian re-ranking step (`α·exp(-κ·d²)`) adds ~0.1ms on top of the candidate retrieval, and promotes high-α splats in the ranking.

---

## Development

```bash
git clone https://github.com/schwabauerbriantomas-gif/m2m-vector-search.git
cd m2m-vector-search
pip install -e ".[all]"

# Run tests
pytest tests/ -v  # 394 tests

# Code quality
black src/ tests/
flake8 src/ tests/
```

---

## License

GNU Affero General Public License v3.0 — see [LICENSE](LICENSE) for details.
