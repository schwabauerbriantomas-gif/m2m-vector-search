<p align="center">
  <img src="https://img.shields.io/badge/version-2.1.0-blue" alt="Version">
  <img src="https://img.shields.io/badge/python-3.10%2B-green" alt="Python">
  <img src="https://img.shields.io/badge/license-AGPL-3.0-orange" alt="License">
  <img src="https://img.shields.io/badge/tests-166%20passed-success" alt="Tests">
  <img src="https://img.shields.io/badge/backends-CPU%20%7C%20CUDA%20%7C%20Vulkan-purple" alt="Backends">
</p>

<h1 align="center">🔬 M2M Vector Search</h1>

<p align="center">
  <strong>Machine-to-Memory</strong> - Búsqueda vectorial con Gaussian Splats, Modelos Basados en Energía y GPU multi-backend
</p>

<p align="center">
  <a href="#quick-start">Quick Start</a> •
  <a href="#alfred-memory">Alfred Memory</a> •
  <a href="#features">Features</a> •
  <a href="#architecture">Architecture</a> •
  <a href="#benchmarks">Benchmarks</a>
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

## 🎩 Alfred Memory (v2.2)

**AlfredMemoryDB** - Semantic memory system purpose-built for AI assistants.

```python
from m2m import AlfredMemoryDB
from sentence_transformers import SentenceTransformer

# 1. Setup encoder (bge-small-en-v1.5: best all-rounder per research)
model = SentenceTransformer("BAAI/bge-small-en-v1.5")
encoder = lambda text: model.encode(text, show_progress_bar=False)

# 2. Create memory DB with Phase 2 features
mem = AlfredMemoryDB(
    encoder=encoder,
    latent_dim=384,
    storage_path="./alfred_memory",
    fusion_method="rrf",          # "rrf", "weighted", "vector_only", "bm25_only"
    temporal_decay=True,           # Recent memories rank higher
    temporal_half_life_days=30.0,  # Half-life in days
    auto_categorize=True,          # Auto-infer category from text
)

# 3. Store memories
mem.store("User prefers dark mode for coding", metadata={"category": "preference"})
mem.store("We decided to use Qdrant for production", metadata={"category": "decision"})

# 4. Search
results = mem.search("what did we decide about databases?", k=5)
for r in results:
    print(f"[{r.score:.3f}] [{r.metadata.get('category', '?')}] {r.document[:80]}")
```

### AlfredMemoryDB API Reference

| Method | Description |
|--------|-------------|
| `store(text, metadata, doc_id)` | Store a memory with auto-embedding |
| `store_with_vector(text, vector, metadata, doc_id)` | Store with pre-computed vector |
| `batch_store(texts, metadatas, ids)` | Store multiple memories efficiently |
| `search(query, k, filter, hybrid)` | Search with hybrid fusion |
| `delete(id, ids, filter)` | Delete memories |
| `get(doc_id)` | Retrieve a specific memory |
| `save()` | Persist to disk |
| `stats()` | System statistics and health |
| `clear()` | Clear all memories |

### Fusion Methods

Based on research via Z.AI tools (RRF used by Elasticsearch, OpenSearch, Pinecone):

| Method | Tuning | Accuracy | Latency | Use Case |
|--------|--------|----------|---------|----------|
| **RRF** | None (k=60) | Good | Instant | General-purpose (recommended) |
| **Weighted** | α weight needed | Better (if tuned) | Instant | Domain-specific with known priorities |
| **vector_only** | None | Good | Instant | Pure semantic search |
| **bm25_only** | None | Good | Instant | Pure keyword search |

### Phase 2 Features (Research-Backed)

**Temporal Decay** - Exponential decay: `boost = exp(-λ × age)` where `λ = ln(2) / half_life`
- Recent memories receive higher scores (configurable half-life)
- Based on: temporal relevance in information retrieval research

**Auto-Categorization** - Keyword-based classification into 10 categories:
`decision`, `preference`, `project`, `error`, `learning`, `question`, `conversation`, `task`, `config`

**Auto-Date** - Automatically sets `date` metadata to current date if not provided

### Workspace Indexing

```bash
# Index all Alfred workspace files
python scripts/index_alfred_workspace.py --storage-path ./alfred_indexed_memory

# Force re-index
python scripts/index_alfred_workspace.py --reindex
```

Indexes: `~/.openclaw/workspace/` (all .md, .py, .json) + M2M source code.

## Quick Start (Core Vector DB)

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

            AlfredMemoryDB Architecture (v2.1)

    ┌──────────────────────────────────────┐
    │         AlfredMemoryDB               │
    ├──────────────────────────────────────┤
    │  store() → encoder → vector + BM25   │
    │  search() → fusion → ranked results  │
    ├──────────┬───────────────────────────┤
    │  Vector  │  BM25      │  Fusion      │
    │  Search  │  Keyword   │  Engine      │
    │  (M2M)   │  Index     │              │
    ├──────────┴────────────┤  RRF /       │
    │  Temporal Decay       │  Weighted    │
    │  Auto-Categorize      │  Vector-Only │
    │  Auto-Date            │  BM25-Only   │
    └───────────────────────┴──────────────┘
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

## Development

```bash
git clone https://github.com/schwabauerbriantomas-gif/m2m-vector-search.git
cd m2m-vector-search
pip install -e ".[all]"
pytest tests/ -v  # 166 tests
```

## Research Notes (Phase 2)

Sources consulted via Z.AI tools (GLM-5 web_search + web_read):

1. **Hybrid Search Fusion** - RRF is industry standard (ES, OpenSearch, Pinecone). Score-agnostic, no normalization needed, k=60 default. Cross-encoder reranking for highest accuracy (RRF → reranker pattern).

2. **Embedding Models** - bge-small-en-v1.5: best all-rounder. gte-small: highest MTEB accuracy. all-MiniLM-L6-v2: fastest but aging. All 384D. all-MiniLM-L12-v2: avoid (slower, not better).

3. **Vector Compression** - Scalar Quantization (int8): 4x memory reduction, <1% recall loss. Binary Quantization: 32x reduction but needs oversampling+reranking. Matryoshka embeddings: store 1024D, index 256D.

4. **Index Algorithms** - HNSW: best speed/accuracy (30-50% memory overhead). IVF: good balance, tunable via nprobe. PQ: memory compression only, lossy.

5. **Production Architecture** - Separate storage/compute. Tenant-based sharding. Time-based sharding for RAG. Mutability requirements in 2025.

## License

GNU Affero General Public License v3.0 - see [LICENSE](LICENSE) for details.
CENSE) for details.
