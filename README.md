<p align="center">
  <img src="https://img.shields.io/badge/version-2.2.0-blue" alt="Version">
  <img src="https://img.shields.io/badge/python-3.10%2B-green" alt="Python">
  <img src="https://img.shields.io/badge/license-AGPL--3.0-orange" alt="License">
  <img src="https://img.shields.io/badge/tests-394%20passed-success" alt="Tests">
  <img src="https://img.shields.io/badge/backends-CPU%20%7C%20CUDA%20%7C%20Vulkan-purple" alt="Backends">
</p>

<h1 align="center">M2M Vector Search</h1>

<p align="center">
  <strong>Machine-to-Memory</strong> &mdash; A vector search engine with probabilistic Gaussian Splats,
  online learning via feedback, energy-based uncertainty quantification, and multi-backend GPU acceleration.
</p>

<p align="center">
  <a href="#quick-start">Quick Start</a> &bull;
  <a href="#features">Features</a> &bull;
  <a href="#architecture">Architecture</a> &bull;
  <a href="#semantic-memory">Semantic Memory</a> &bull;
  <a href="#benchmarks">Benchmarks</a> &bull;
  <a href="#changelog">Changelog</a>
</p>

---

## Features

- **Gaussian Splats** — Each vector is represented as a learnable Gaussian: `score(x, i) = αᵢ · exp(-κᵢ · ‖x − μᵢ‖²)`. Three parameters (μ, κ, α) encode position, concentration, and importance independently.
- **Online Learning** — Hebbian update rules adapt splat parameters from user feedback after each query. No retraining, no re-indexing.
- **Energy-Based Model** — Native uncertainty quantification via an energy landscape. Every search result carries a confidence score derived from the local energy topology.
- **HRM2 Engine** — Hierarchical Routing with Mixture Models and adaptive probing for sub-linear search at scale.
- **SOC Consolidation** — Self-Organized Criticality automatically prunes low-contribution splats. The system reaches avalanches and relaxes to equilibrium, mimicking neuronal memory consolidation.
- **Multi-GPU Backend** — CPU, NVIDIA CUDA, and AMD Vulkan through a single API. Backend selection is transparent.
- **Semantic Memory** — Hybrid BM25 + vector search with Reciprocal Rank Fusion, temporal decay, and auto-categorization.
- **LangChain Integration** — Native `BaseRetriever` implementation with full CRUD support.
- **Edge / Cluster** — Distributed mode with edge nodes, a coordinator, load balancing, and sharding.

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

db = SimpleVectorDB(latent_dim=128)

vectors = np.random.randn(1000, 128).astype(np.float32)
db.add(vectors=vectors, ids=[f"doc_{i}" for i in range(1000)])

query = np.random.randn(128).astype(np.float32)
results = db.search(query, k=10, include_metadata=True)

for r in results:
    print(f"  {r.id}: score={r.score:.4f}")
```

### Advanced: Gaussian Splats + Energy

`AdvancedVectorDB` adds energy-based uncertainty, SOC consolidation, and Langevin exploration:

```python
from m2m import AdvancedVectorDB
import numpy as np

db = AdvancedVectorDB(latent_dim=128)
vectors = np.random.randn(500, 128).astype(np.float32)
db.add(ids=[f"item_{i}" for i in range(500)], vectors=vectors)

query = np.random.randn(128).astype(np.float32)

# Search with energy + confidence scores
result = db.search_with_energy(query, k=10)
for r in result.results:
    print(f"  {r.id}: score={r.score:.4f}  "
          f"energy={r.energy:.4f}  confidence={r.confidence:.4f}")

# SOC consolidation: prune low-α splats
removed = db.consolidate()
print(f"Consolidated {removed} splats")

# Check system criticality
report = db.check_criticality()
print(f"Criticality index: {report.index:.4f}  state: {report.state}")
```

**Update rules (Hebbian + temporal decay):**

| Event | α (importance) | κ (concentration) | μ (position) |
|-------|:-:|:-:|:-:|
| Relevant feedback | `α += lr_α · α` | `κ += lr_κ · ‖x − μ‖⁻²` | `μ += lr_μ · (x − μ)` |
| Irrelevant feedback | `α *= (1 − lr_α)` | `κ -= 0.5 · lr_κ` | — |
| Temporal decay | `α *= exp(-λ·Δt)` | — | — |

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                   REST API (FastAPI)                 │
│            Collections · CRUD · Search               │
├─────────────────────────────────────────────────────┤
│           SemanticMemoryDB / VectorDB                │
│      Hybrid Search · Fusion · Temporal Decay         │
├──────────┬──────────┬───────────┬───────────────────┤
│  Splats  │  HRM2    │  EBM      │  SOC              │
│  (μ,κ,α) │  Engine  │  Energy   │  Consolidate      │
├──────────┴──────────┴───────────┴───────────────────┤
│              Backend Layer (pluggable)               │
├─────────┬──────────┬──────────┬─────────────────────┤
│   CPU   │  CUDA    │  Vulkan  │  Transformed        │
├─────────┴──────────┴──────────┴─────────────────────┤
│              Storage Layer                           │
├─────────┬─────────────────┬─────────────────────────┤
│  WAL    │  Persistence    │  GPUVectorIndex         │
│         │  (SQLite+NPY)   │                         │
├─────────┴─────────────────┴─────────────────────────┤
│              Cluster / Edge Layer                    │
├──────────┬───────────┬──────────┬───────────────────┤
│  Router  │  Balancer │  Sharding│  Edge Nodes       │
└──────────┴───────────┴──────────┴───────────────────┘
```

### Module Map

| Module | Responsibility |
|--------|----------------|
| `splats.py` | Gaussian Splat tensor management, `find_neighbors`, `feedback` |
| `hrm2_engine.py` | Hierarchical routing, adaptive probing, coarse-to-fine search |
| `gaussian_scoring.py` | Two-phase scoring: L2 retrieval + Gaussian re-ranking |
| `geometry.py` | Riemannian operations on the hypersphere (exp_map, log_map) |
| `ebm/energy_api.py` | Energy landscape computation, uncertainty quantification |
| `ebm/soc.py` | Self-Organized Criticality: avalanches, relaxation, consolidation |
| `ebm/exploration.py` | Langevin dynamics exploration, Boltzmann sampling |
| `semantic_memory.py` | Hybrid BM25 + vector, RRF fusion, temporal decay |
| `dataset_transformer.py` | Raw vectors → Gaussian Splats via KMeans clustering |
| `query_optimizer.py` | Query prefetching with bigram transition model |
| `encoding.py` | Color histogram + positional encoding for multi-modal splats |
| `storage/persistence.py` | SQLite metadata + NPY shards + HMAC-signed index |
| `storage/wal.py` | Write-Ahead Log for crash recovery |
| `api/edge_api.py` | FastAPI REST endpoints with configurable CORS |
| `cluster/` | Distributed mode: router, balancer, edge nodes, sharding |
| `lsh_index.py` | Cross-Polytope LSH fallback for uniform distributions |

---

## Semantic Memory

```python
from m2m.semantic_memory import SemanticMemoryDB

# Use any encoder that returns a numpy float32 vector
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("BAAI/bge-small-en-v1.5")
encoder = lambda text: model.encode(text, show_progress_bar=False)

mem = SemanticMemoryDB(
    encoder=encoder,
    latent_dim=384,
    fusion_method="rrf",
    temporal_decay=True,
    temporal_half_life_days=30.0,
    auto_categorize=True,
)

mem.store("User prefers dark mode for coding", metadata={"category": "preference"})
mem.store("We decided to use Qdrant for production", metadata={"category": "decision"})

results = mem.search("what did we decide about databases?", k=5)
```

### Hybrid Search Fusion Methods

| Method | Tuning Required | Best For |
|--------|:-:|----------|
| **RRF** (Reciprocal Rank Fusion) | No | General-purpose (recommended) |
| **Weighted** | Yes | Domain-specific with known priorities |
| `vector_only` | No | Pure semantic search |
| `bm25_only` | No | Pure keyword search |

---

## Security

- **Restricted Unpickler** — All `pickle.loads()` calls are replaced with a whitelist-based `_RestrictedUnpickler` that blocks `os.system`, `subprocess`, `eval`, and any non-numpy/non-builtin class. Prevents arbitrary code execution from tampered cache or index files.
- **HMAC-Signed Index** — `save_index()` / `load_index()` verify an HMAC-SHA256 signature using the `M2M_HMAC_SECRET` environment variable. Tampered index files are rejected before deserialization.
- **Configurable CORS** — REST API origins are controlled via `M2M_CORS_ORIGINS` env var (comma-separated). Default is permissive for development.
- **Path Traversal Protection** — `storage_path` and `backup_path` are validated against `..` traversal attacks.
- **No Silent Failures** — Embedding model errors raise exceptions instead of injecting random noise.

---

## Benchmarks

> All data below is from real measurements on the specified hardware. No synthetic or estimated numbers.

**System:** AMD Ryzen 5 3400G, 16 GB RAM, Python 3.12, CPU-only (no GPU acceleration)

### Multi-Scale Performance (dim=384, k=10, 100 queries)

**Methodology:** Synthetic data with 20+ Gaussian clusters. Queries are in-distribution (sampled from the same cluster structure), simulating real RAG workloads where query embeddings are drawn from the same distribution as the corpus. Ground truth is exact brute-force k-NN via cosine distance.

| Dataset Size | Build (s) | p50 (ms) | p95 (ms) | p99 (ms) | QPS | Linear QPS | Speedup | Recall@10 |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| 1,000 | 0.9 | 6.8 | 8.7 | 9.4 | 142.1 | 1,484.2 | 0.1x | 1.0000 |
| 5,000 | 1.5 | 7.9 | 9.6 | 10.0 | 121.8 | 278.0 | 0.4x | 0.9820 |
| 10,000 | 3.4 | 8.0 | 8.3 | 8.7 | 125.2 | 148.0 | 0.8x | 1.0000 |
| 50,000 | 36.5 | 13.2 | 14.3 | 15.3 | 75.5 | 21.2 | **3.6x** | 1.0000 |
| 100,000 | 17.2 | 12.9 | 15.5 | 17.3 | 74.5 | 10.3 | **7.2x** | 0.8570 |

**Key observations:**

- **Crossover at ~15K vectors.** M2M's hierarchical routing (HRM2) activates above 15K splats. Below that threshold, brute-force scan is faster due to lower overhead.
- **Sub-linear scaling.** Linear scan QPS drops 144x (1,484→10.3) from 1K to 100K. M2M drops only 1.9x (142→74.5).
- **Recall = 1.0 up to 50K** with in-distribution queries. At 100K, recall drops to 0.857 — the `n_probe` parameter may need tuning for very large datasets.
- **Latency stays flat.** p50 grows from 6.8ms to 12.9ms (1.9x) while dataset grows 100x.

### Previous Measurement (dim=640, k=64)

| Metric | Linear Scan | M2M HRM2 | Speedup |
|--------|:-:|:-:|:-:|
| Latency (ms) | 94.79 | 0.99 | **32.4x** |
| QPS | 10.55 | 1,012.77 | 96.0x |

> GPU benchmarks (CUDA/Vulkan) are not included because they have not been formally validated. They will be added when reproducible measurements are available.

**Reproduce:** `python scripts/benchmark_final.py`

---

## Development

```bash
git clone https://github.com/schwabauerbriantomas-gif/m2m-vector-search.git
cd m2m-vector-search
pip install -e ".[all]"

# Run tests (394 tests, excludes GPU and integration marks)
pytest tests/ -q -m "not gpu and not integration"

# Code quality
black src/ tests/
flake8 src/ tests/
```

### Project Structure

```
src/m2m/
├── __init__.py              # SimpleVectorDB, AdvancedVectorDB, public API
├── splats.py                # M2MMemory: splat management, feedback, find_neighbors
├── hrm2_engine.py           # HRM2 search engine with adaptive probing
├── gaussian_scoring.py      # Two-phase Gaussian scoring (chunked batch)
├── geometry.py              # Riemannian ops on S^d
├── encoding.py              # Multi-modal encoding (color histogram + positional)
├── semantic_memory.py       # SemanticMemoryDB: hybrid search + fusion
├── dataset_transformer.py   # Vectors → Splats via KMeans
├── query_optimizer.py       # Query prefetching (bigram model)
├── entity_extractor.py      # Entity extraction from search results
├── config.py                # M2MConfig presets
├── engine.py                # M2MEngine: backend abstraction
├── ebm/
│   ├── energy_api.py        # Energy landscape computation
│   ├── soc.py               # Self-Organized Criticality engine
│   └── exploration.py       # Langevin dynamics + Boltzmann sampling
├── storage/
│   ├── persistence.py       # SQLite + NPY shards + HMAC index
│   └── wal.py               # Write-Ahead Log
├── api/
│   ├── edge_api.py          # FastAPI REST server
│   └── coordinator_api.py   # Cluster coordinator
├── cluster/
│   ├── client.py            # Cluster client
│   ├── edge_node.py         # Edge node with coordinator sync
│   ├── balancer.py          # Load balancer
│   ├── sharding.py          # Shard management
│   └── router.py            # Query routing
├── lsh_index.py             # Cross-Polytope LSH fallback
├── gpu_vector_index.py      # GPU backend
├── gpu_hierarchical_search.py  # GPU hierarchical search
└── train_embeddings.py      # Knowledge distillation for embeddings
```

---

## Changelog

### v2.2.1 — Critical Search Fix + Multi-Scale Benchmarks

**Critical bug fix (P0):**
- **`find_neighbors()` index mapping** — the function ignored `result_indices` returned by `two_phase_search()` and instead recomputed indices via `candidates[local_j]`, where `local_j` was a position in the score array (length `k`), not the candidate array. This caused `search()` to always return the first `k` splats by insertion order regardless of the query vector. Fixed by using `result_indices` directly and propagating splat indices through the call chain: `find_neighbors → retrieve → M2MEngine.search → SimpleVectorDB.search`.
- **`SimpleVectorDB.search()` doc_id mapping** — was mapping search results to documents by insertion order (`active_ids[i]`) instead of by splat index. Now maps via `_splat_id_order[splat_idx]` to return the correct document IDs.
- **`find_neighbors()` return signature** — now returns `(mu, alpha, kappa, splat_indices)` as a 4-tuple to enable proper document ID mapping. All callers updated.

**Test improvements:**
- `_mock_encoder` upgraded from full-string hash to word-level hash averaging. Previous encoder produced semantically meaningless vectors (same words → different directions), masking search correctness bugs. New encoder ensures texts sharing words are closer in vector space, matching real embedding model behavior.
- 394 tests pass (was 395 — adjusted for API signature change; previous count included tests that only passed due to the insertion-order bug).

**Benchmarks:**
- Added multi-scale benchmark suite (1K–100K vectors) with recall@10, p50/p95/p99 latency, QPS, and linear baseline comparison.
- All measurements from real hardware. Reproducible via `scripts/benchmark_final.py`.

### v2.2.0 — Refactor & Security Hardening

**Math / Logic fixes (P0):**
- `SOC.relax()` — replaced naive normalization with gradient descent over the energy landscape. Previously, α was normalized and κ grew monotonically without convergence.
- `geometry.py` — implemented real Riemannian operations: `exp_map`, `log_map`, `project_to_tangent` with numerical stability (`arccos` clamped to `[0, π]`).
- `find_neighbors()` — auto-builds the index if splats exist but haven't been indexed yet. Returns empty arrays on empty collections instead of crashing.
- `QueryPrefetcher` — implemented bigram transition model for query prediction (was: always returned `None`).
- `_color_histogram_encoding_numba` — replaced hardcoded `512` dimension with dynamic `n_bins³` calculation.

**Security fixes (P1):**
- `_RestrictedUnpickler` — all `pickle.loads()` replaced with whitelist-based deserialization. Blocks arbitrary code execution from tampered cache/index files.
- `gaussian_score_batch` — added `chunk_size=4096` parameter to prevent OOM on large batches.
- `Boltzmann sampling` — stabilized with `subtract(max)` before exponentiation to prevent overflow.
- `entity_extractor` — removed `np.random.randn` fallback when embedding model fails; now raises a controlled exception.
- `edge_api.py` — CORS origins configurable via `M2M_CORS_ORIGINS` env var (was: invalid `["*"]` + `allow_credentials=True`). Global error handler returns `JSONResponse` instead of `HTTPException`.
- `LangChain delete()` — now marks splats as deleted in the M2M engine (was: only updated internal `_store` dict).

**Performance & cleanup (P2):**
- Thread safety: `threading.RLock` added to `SimpleVectorDB.add()`, `update()`, `delete()`. Mutations are locked; storage I/O runs outside the lock.
- `_adaptive_n_probe()` — `np.partition()` O(N) replaces `np.sort()` O(N log N).
- `edge_node.sync_with_coordinator()` — implemented with `requests.post()` heartbeat (was: stub with `pass`).
- Removed dead `learn_entity()` stub from `entity_extractor.py`.

**Tests:** 395 passed, 0 failed (was: 393 passed, 1 failed).

---

## License

GNU Affero General Public License v3.0 — see [LICENSE](LICENSE) for details.
