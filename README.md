<p align="center">
  <img src="https://img.shields.io/badge/version-2.1.0-blue" alt="Version">
  <img src="https://img.shields.io/badge/python-3.10%2B-green" alt="Python">
  <img src="https://img.shields.io/badge/license-Apache%202.0-orange" alt="License">
  <img src="https://img.shields.io/badge/tests-53%20passed-success" alt="Tests">
  <img src="https://img.shields.io/badge/backends-CPU%20%7C%20CUDA%20%7C%20Vulkan-purple" alt="Backends">
</p>

<h1 align="center">🔬 M2M Vector Search</h1>

<p align="center">
  <strong>Machine-to-Memory</strong> — Búsqueda vectorial con Gaussian Splats, Modelos Basados en Energía y GPU multi-backend
</p>

<p align="center">
  <a href="#quick-start">Quick Start</a> •
  <a href="#features">Features</a> •
  <a href="#architecture">Architecture</a> •
  <a href="#benchmarks">Benchmarks</a> •
  <a href="#comparison">Comparison</a>
</p>

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🔮 **Gaussian Splats** | Representación probabilística de vectores con μ, κ, α |
| ⚡ **Multi-GPU Backend** | CPU, NVIDIA CUDA, y AMD Vulkan con una sola API |
| 🧠 **Energy-Based Models** | Quantificación de incertidumbre nativa |
| 🔍 **HRM2 Engine** | Hierarchical Routing with Mixture Models |
| 🌐 **Distributed Cluster** | Arquitectura Edge/Coordinator con sharding |
| 📊 **SOC Consolidation** | Self-Organized Criticality para memoria |
| 🗄️ **Full CRUD API** | REST API con colecciones, metadata, documentos |
| 🔗 **LangChain Ready** | Retriever interface nativo |
| 📈 **Query Optimizer** | LRU cache y batch optimization |

## Quick Start

```bash
pip install m2m-vector-search
```

```python
from m2m import SimpleVectorDB
import numpy as np

# 1. Crear base de datos
db = SimpleVectorDB(latent_dim=768)

# 2. Agregar vectores
vectors = np.random.randn(1000, 768).astype(np.float32)
db.add(vectors=vectors, ids=[f"doc_{i}" for i in range(1000)])

# 3. Buscar
query = np.random.randn(768).astype(np.float32)
results = db.search(query, k=10)

for r in results:
    print(f"  {r.id}: score={r.score:.4f}")
```

## Advanced (EBM)

```python
from m2m import AdvancedVectorDB

db = AdvancedVectorDB(latent_dim=768, enable_energy_features=True)
db.add(vectors=vectors, ids=ids, metadata=[{"topic": "ml"} for _ in range(1000)])

# Search with energy
result = db.search_with_energy(query, k=10)
print(f"Confidence: {result.total_confidence:.2%}")
print(f"Uncertainty regions: {len(result.uncertainty_regions)}")

# Explore unknown areas
suggestions = db.suggest_exploration(n=3)
for s in suggestions:
    print(f"  💡 {s.description} (energy={s.region.energy:.2f})")
```

## Architecture

```
┌─────────────────────────────────────────────────┐
│                   REST API (FastAPI)             │
├─────────────────────────────────────────────────┤
│  CollectionManager  │  CRUD  │  Search  │  EBM  │
├─────────────────────────────────────────────────┤
│         SimpleVectorDB / AdvancedVectorDB        │
├──────────┬──────────┬───────────┬───────────────┤
│  Splats  │  HRM2    │  EBM      │  SOC          │
│ (μ,κ,α)  │  Engine  │  Energy   │  Consolidate  │
├──────────┴──────────┴───────────┴───────────────┤
│              Backend Layer (pluggable)           │
├─────────┬──────────┬──────────┬─────────────────┤
│   CPU   │  CUDA    │  Vulkan  │  Transformed    │
├─────────┴──────────┴──────────┴─────────────────┤
│              Storage Layer                       │
├─────────┬─────────────────┬─────────────────────┤
│  WAL    │  Persistence    │  GPUVectorIndex     │
└─────────┴─────────────────┴─────────────────────┘

         Distributed Cluster Architecture

    ┌──────────────┐
    │ Coordinator  │ ← Routes queries, aggregates results (RRF)
    └──────┬───────┘
           │ HTTP
    ┌──────┴───────┬──────────────┐
    │   Edge 1     │   Edge 2     │  ... Edge N
    │ (Shard A)    │ (Shard B)    │
    └──────────────┴──────────────┘
```

## Benchmarks

> ⚠️ **Todos los datos son mediciones reales.** Ver `benchmark_stats.md` para el análisis completo.

**Sistema:** AMD Ryzen 5 3400G, 16GB RAM, NVIDIA RTX 3090, Python 3.12  
**Config:** 10K splats, 1K queries, k=10, dim=640

| Backend | Latencia (ms) | Throughput (QPS) | vs Linear |
|---------|:-------------:|:----------------:|:---------:|
| Linear (numpy) | 24.21 | 41.31 | 1.00x |
| M2M CPU | 32.93 | 30.37 | 0.74x |
| M2M Vulkan | 32.78 | 30.51 | 0.74x |
| M2M CUDA | 26.54 | 37.68 | 0.91x |

**Nota:** M2M no supera a linear search para N≤10K. El overhead de HRM2 se compensa en datasets más grandes. Ver `optimization_report.md` para el plan de mejora.

## Comparison

| Feature | M2M | FAISS | hnswlib | Milvus | Weaviate |
|---------|:---:|:-----:|:-------:|:------:|:--------:|
| GPU (CUDA) | ✅ | ✅ | ❌ | ✅ | ✅ |
| GPU (Vulkan/AMD) | ✅ | ❌ | ❌ | ❌ | ❌ |
| Uncertainty Quant. | ✅ | ❌ | ❌ | ❌ | ❌ |
| Energy-Based Search | ✅ | ❌ | ❌ | ❌ | ❌ |
| Exploration Suggest. | ✅ | ❌ | ❌ | ❌ | ✅ |
| Distributed | ✅ | Parcial | ❌ | ✅ | ✅ |
| Persistence | ✅ | Manual | ❌ | ✅ | ✅ |
| REST API | ✅ | ❌ | ❌ | ✅ | ✅ |
| LangChain | ✅ | ✅ | ✅ | ✅ | ✅ |
| Pure Python | ✅ | ❌ | ❌ | ❌ | ❌ |
| Dependencies | numpy, sklearn | faiss-cpu | hnswlib | Docker | Docker |

## Use Cases

- 🔍 **Semantic Search:** RAG pipelines con embeddings
- 🧠 **Uncertainty-Aware Retrieval:** Saber cuándo no saber
- 🌐 **Edge Deployment:** Vulkan para GPUs AMD/Intel
- 📊 **Research:** Energy-Based Models para vector spaces
- 🔗 **Multi-Agent Memory:** Memory backend para sistemas multi-agente

## Installation

```bash
# Core (CPU only)
pip install m2m-vector-search

# With GPU backends
pip install m2m-vector-search[vulkan]  # AMD/Intel Vulkan
pip install m2m-vector-search[cuda]    # NVIDIA CUDA

# Development
pip install m2m-vector-search[all]
```

## Development

```bash
git clone https://github.com/schwabauerbriantomas-gif/m2m-vector-search.git
cd m2m-vector-search
pip install -e ".[all]"
pytest tests/ -v  # 53 tests
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Roadmap

- [x] v2.0: Open source release with CPU/Vulkan
- [x] v2.1: CUDA backend, EBM features, cluster mode
- [ ] v2.2: Incremental index updates
- [ ] v2.3: Benchmark N=100K-1M, crossover validation
- [ ] v3.0: Learned indices, quantization, disk-based mode

## Citation

```bibtex
@software{m2m_vector_search,
  author = {Schwabauer, Brian},
  title = {M2M Vector Search: Gaussian Splat-based Vector Database with Energy-Based Models},
  year = {2026},
  url = {https://github.com/schwabauerbriantomas-gif/m2m-vector-search},
  license = {Apache-2.0}
}
```

## License

Apache License 2.0 — see [LICENSE](LICENSE) for details.

## Acknowledgments

- [FAISS](https://github.com/facebookresearch/faiss) — Inspiration for GPU vector search
- [HNSW](https://arxiv.org/abs/1603.09320) — Graph-based ANN reference
- [3D Gaussian Splatting](https://arxiv.org/abs/2308.04079) — Gaussian representation paradigm
- [Energy-Based Learning](https://www.cs.nyu.edu/~yann/talks/lecun-20060928.pdf) — EBM theoretical foundation
