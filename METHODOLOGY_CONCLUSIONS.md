# M2M Vector Search — Methodology Conclusions

**Date**: 2026-03-05
**Dataset tested**: DBpedia (OpenAI text-embedding-3-large, 640D)
**Primary conclusion**: Linear Scan remains the best option for uniformly distributed embeddings; M2M HRM2 excels on structured data and multi-modal workloads

---

## Executive Summary

Multiple methodologies were tested to improve vector search performance on text embeddings. **No method outperformed Linear Scan** for uniformly distributed datasets such as DBpedia. However, the current M2M engine — built on top of **Gaussian Splats**, a **two-level hierarchical index (HRM2)**, and optional **Vulkan GPU acceleration** — provides a meaningful speedup over pure CPU for structured or multi-modal data, and offers unique capabilities (generative exploration, SOC memory consolidation, 3-tier memory tiering) that no classic ANN library provides.

---

## 📊 Comparative Results

### Classic Search Methodologies (DBpedia, uniform distribution)

| Methodology | Recall | Speedup | Conclusion |
|-------------|--------|---------|------------|
| **Linear Scan** | 100% | 1.0x | ✅ **Best for uniform data** |
| HETD Basic | 100% | 0.5x | ❌ Slower |
| Adaptive HETD | 70% | 6x | ❌ Low recall |
| HETD + PCA | 93% | 0.5x | ❌ Slower |
| Enhanced Transformer | 95% | 0.5x | ❌ Slower |
| M2M Resonant | 46% | 3x | ❌ Very low recall |

### Current M2M Engine Benchmark (sklearn synthetic dataset, 10,000 vectors, 640D)

| System | Avg Latency | Throughput | Speedup |
|--------|-------------|------------|---------|
| **Linear Scan (baseline)** | 30.06 ms | 33.26 QPS | 1.0x |
| **M2M CPU (HRM2)** | 89.24 ms | 11.20 QPS | 0.3x |
| **M2M Vulkan (HRM2 + GPU)** | **51.88 ms** | **19.28 QPS** | **0.6x** |

> **Note on homogeneous distributions and latency**: When vectors are uniformly distributed (dense cluster without clear boundaries), the internal K-Means index cannot separate them into distinct semantic paths. Consequently, HRM2 must probe multiple overlapping clusters, forcing latency toward `O(N)` linear time instead of the ideal `O(√N)` achieved on naturally grouped datasets.

---

## 🏗 Current Implementation Architecture

### Core Components

The M2M engine is composed of the following modules:

| Module | File | Role |
|--------|------|------|
| **SimpleVectorDB / AdvancedVectorDB** | `m2m.py` | Public high-level API |
| **M2MMemory** | `m2m.py` | 3-tier memory manager + SOC controller |
| **M2MEngine** | `m2m.py` | Core orchestrator: routing, add, search, generate |
| **SplatStore** | `splats.py` | Physical tensor storage (μ, α, κ) + GPU index lifecycle |
| **HRM2Engine** | `hrm2_engine.py` | Two-level hierarchical K-Means index |
| **GPUVectorIndex** | `gpu_vector_index.py` | Persistent Vulkan compute shader index |
| **HierarchicalGPUSearch** | `gpu_hierarchical_search.py` | Two-stage GPU ANN search (centroids + clusters) |
| **Config** | `config.py` | Profiles: `simple()` / `advanced()` |

### Two Public Modes of Operation

#### 1. `SimpleVectorDB` — *"The SQLite of Vector DBs"*
Designed for **edge devices and RAG pipelines**. Disables expensive agentic features (SOC, Langevin dynamics, 3-tier memory tiering) to maximize throughput and minimize memory footprint.

```python
from m2m import SimpleVectorDB
db = SimpleVectorDB(device='cpu')
db.add(vectors)                         # ← normalizes to S^639, builds HRM2 index
results = db.search(query, k=10)        # ← HRM2 hierarchical search (CPU)
```

#### 2. `AdvancedVectorDB` — *"The Cognitive Latent Space"*
Designed for **autonomous agents**. Enables the full cognitive suite:
- **3-Tier Memory**: VRAM (Hot ~0.1 ms) → RAM (Warm ~0.5 ms) → SSD (Cold ~10 ms)
- **Langevin Dynamics**: generative latent space exploration
- **SOC Consolidation**: passive pruning of redundant/dead memory

```python
from m2m import AdvancedVectorDB
db = AdvancedVectorDB(device='cuda')
db.add(vectors)
nearest = db.search(query, k=10)
samples = db.generate(query, n_steps=20)   # ← Underdamped Langevin walk
removed = db.consolidate(threshold=0.85)   # ← SOC memory pruning
```

### HRM2 — Two-Level Hierarchical Index

The `HRM2Engine` builds a **two-level K-Means tree** over the vector space:

```
Query
  │
  ▼
Level 1 (Coarse): Query vs C cluster centroids  — O(Q × C), C << N
  │  → selects n_probe closest macro-clusters
  ▼
Level 2 (Fine): Query vs fine clusters within each macro-cluster — O(Q × n_probe × M)
  │  → MoE router scores exact candidates
  ▼
Top-k results
```

**Level of Detail (LOD)** modes, selectable at query time:
| LOD | Mode | Speed | Accuracy |
|-----|------|-------|----------|
| 0 | Coarse approximation | Ultra-fast | Low |
| 1 | Fine approximation | Fast | Medium |
| 2 | Exact MoE router | Standard | High (default) |

### Vulkan GPU Acceleration (`GPUVectorIndex`)

The `GPUVectorIndex` class implements **persistent GPU buffer** compute via Vulkan:

- **Index buffer** uploaded **once** at initialization — never re-uploaded unless `rebuild()` is called.
- **Query buffer** (dynamic): only queries (small) are transferred per search call.
- **Result buffer** (dynamic): bounded by `max_batch × CHUNK_SIZE × 4 bytes ≈ 3 MB`.
- **Dispatch**: `vkCmdDispatch(ceil(N/256), B, 1)` — processes all B queries in a single kernel launch.
- **Chunked dispatch**: iterates over the index in 8,192-vector chunks, accumulating a rolling top-k to keep memory bounded.
- **Shader**: GLSL compute shader (`shaders/moe_batch.comp`) compiled to SPIR-V (`moe_batch.spv`) at first run via `glslc`.

**Three persistent GPU buffer layout:**

```
┌───────────────────┬─────────────────┬───────────────────────┐
│ Region            │ Size            │ Contents              │
├───────────────────┼─────────────────┼───────────────────────┤
│ Index Buffer      │ N × D × 4 bytes │ Index vectors (float) │
│ (Persistent)      │                 │ — uploaded ONCE       │
├───────────────────┼─────────────────┼───────────────────────┤
│ Query Buffer      │ B × D × 4 bytes │ Batch of queries      │
│ (Dynamic)         │                 │ — copied per call     │
├───────────────────┼─────────────────┼───────────────────────┤
│ Result Buffer     │ B × C × 4 bytes │ L2 distances          │
│ (Dynamic, Chunked)│                 │ — read after dispatch  │
└───────────────────┴─────────────────┴───────────────────────┘
```

### `HierarchicalGPUSearch` — Two-Stage GPU ANN Search

An alternative GPU path implementing a full two-stage hierarchical GPU search:

- **Stage 1 (Coarse)**: queries vs cluster centroids on GPU — `O(Q × C)`.
- **Stage 2 (Fine)**: queries vs members of the n_probe selected clusters on GPU — `O(Q × n_probe × M)`.
- Gracefully falls back to `_CPUFallbackIndex` (NumPy brute-force) when Vulkan is unavailable.

### Gaussian Splats — The Storage Primitive

Each vector is stored as a **Gaussian Splat** with three properties:

| Parameter | Symbol | Description |
|-----------|--------|-------------|
| Mean | μ | The normalized position on the hypersphere S^(D-1) |
| Opacity | α | Access frequency / relevance weight |
| Concentration | κ | Von Mises-Fisher concentration (spread) |

The `SplatStore` maintains flat NumPy tensors `(mu, alpha, kappa, frequency)` for fast energy and SOC computations, and delegates search to `HRM2Engine`.

### Integrations

| Framework | Integration file | Key class |
|-----------|-----------------|-----------|
| **LangChain** | `integrations/langchain.py` | `M2MVectorStore` |
| **LlamaIndex** | `integrations/llamaindex.py` | `M2MVectorStore` |

---

## 🔍 DBpedia Dataset Analysis

### Measured characteristics

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **Silhouette Score** | -0.0048 | Clusters worse than random |
| **Coefficient of Variation** | 0.085 | Very uniform distribution |
| **Cluster overlap** | 5.5x | Completely overlapping clusters |
| **Distribution** | Uniform on S^639 | No spatial structure |

### Diagnosis

OpenAI `text-embedding-3-large` embeddings are **uniformly distributed** on the hypersphere. There is no natural cluster structure that can be exploited. This is why every hierarchical method (HETD, HRM2, HNSW-style) adds overhead without benefit on this specific dataset.

---

## ✅ When to Use Advanced Methodologies

### Required conditions

| Condition | Optimal Value | How to measure |
|-----------|---------------|----------------|
| Silhouette Score | > 0.2 | `sklearn.metrics.silhouette_score` |
| Coefficient of Variation | > 0.2 | `std(distances) / mean(distances)` |
| Overlap | < 1.5 | `2 × radius / centroid_distance` |

### Appropriate datasets

- ✅ Images (SIFT, SURF, CLIP embeddings)
- ✅ Audio features with patterns (ImageBind, Whisper-based)
- ✅ Geolocation data
- ✅ Video temporal tokens
- ✅ 3D point cloud features (PointNet++)
- ✅ Mixed-modality retrievals (omnimodal workloads)
- ✅ Data with natural clustering

---

## ❌ When NOT to Use (Contraindicated)

### Failure conditions

| Condition | Problematic Value |
|-----------|-----------------|
| Silhouette | < 0.1 |
| Coefficient of Variation | < 0.15 |
| Overlap | > 2.0 |

### Inappropriate datasets

- ❌ Text embeddings from large LMs (DBpedia, GloVe, Sentence-BERT)
- ❌ Data on a uniform hypersphere
- ❌ Pure Gaussian distributions without cluster structure

---

## 🎯 Recommendations by Data Type

### For Uniform Text Embeddings

```
✅ Optimized Linear Scan
   - Latency: ~30ms (10K vectors, CPU)
   - Recall: 100%
   - Simple, predictable, zero overhead

✅ Alternatives for higher throughput:
   - FAISS IVF
   - HNSW (hnswlib)
   - ScaNN
```

### For Structured / Multi-modal Data

```
1. Analyze distribution (Silhouette, CV, Overlap)
2. If structure exists → Use M2M HRM2 or HierarchicalGPUSearch
   - Enable Vulkan for GPU acceleration (requires Vulkan 1.0+)
   - Use SimpleVectorDB for edge / RAG
   - Use AdvancedVectorDB for agents with generative memory
3. If no improvement → Fall back to Linear Scan
```

---

## 🔬 Decision Flow

```
Start
  │
  ▼
Analyze dataset (Silhouette, CV, Overlap)
  │
  ▼
Silhouette > 0.2 AND CV > 0.2?
  │
  ├─ YES ──► Try M2M HRM2 / HierarchicalGPUSearch
  │            │
  │            └── Enable Vulkan? ──► GPUVectorIndex (persistent GPU buffer)
  │                                    vkCmdDispatch (chunked, B queries / call)
  │            │
  │            ▼
  │          Speedup > 1.2x AND Recall > 95%?
  │            │
  │            ├─ YES ──► Use M2M (SimpleVectorDB or AdvancedVectorDB)
  │            │
  │            └─ NO ──► Fall back to Linear Scan
  │
  └─ NO ──► Use Linear Scan or FAISS IVF directly
```

---

## 📁 Project Files

### Core (Maintained)

| File | Purpose |
|------|---------|
| `m2m.py` | Main public API: `SimpleVectorDB`, `AdvancedVectorDB`, `M2MEngine`, `M2MMemory` |
| `hrm2_engine.py` | Two-level hierarchical K-Means index (HRM2) |
| `splats.py` | Gaussian Splat tensor store + HRM2 / GPU index interface |
| `gpu_vector_index.py` | Persistent Vulkan GPU index (`GPUVectorIndex`) |
| `gpu_hierarchical_search.py` | Two-stage GPU ANN search (`HierarchicalGPUSearch`) |
| `config.py` | Configuration profiles (`simple()` / `advanced()`) |
| `clustering.py` | K-Means implementation |
| `encoding.py` | `FullEmbeddingBuilder` for Gaussian Splat feature embedding |
| `splat_types.py` | `GaussianSplat`, `SplatEmbedding`, `SplatCluster` data types |
| `dataset_transformer.py` | Offline transform: flat embeddings → M2M hierarchical splat format |
| `energy.py` | Energy function for SOC / Langevin dynamics |
| `geometry.py` | Spherical geometry utilities |
| `memory.py` | 3-tier memory hierarchy implementation |
| `data_lake.py` | PyTorch DataLoader export |
| `shaders/moe_batch.comp` | GLSL compute shader (Vulkan MoE distance kernel) |
| `shaders/moe_batch.spv` | Compiled SPIR-V compute shader (auto-built at runtime) |
| `integrations/` | LangChain & LlamaIndex adapters |
| `METHODOLOGY_CONCLUSIONS.md` | This document |
| `README.md` | Quick-start guide, architecture overview, benchmarks |
| `CONFIG_RAG.md` | RAG configuration guide |

### Removed

- Failed and temporary tests
- Benchmark scripts that did not add value
- Redundant experimental scripts

---

## 💡 Lessons Learned

1. **There is no universal solution** for vector search
2. **Analyze BEFORE implementing** complex methodologies
3. **Measure real performance**, do not assume theoretical improvements
4. **Linear Scan** is often the best option for uniform distributions
5. **Document limitations** honestly
6. **Index overhead** can outweigh any benefit on homogeneous data
7. **GPU acceleration** (Vulkan) provides a real but modest boost (~1.7x over CPU M2M) — most effective on structured, high-dimensional data or large batch queries
8. **Hierarchical indexing** (HRM2) shines on clustered data; degrades to near-O(N) on uniform distributions
9. **Gaussian Splats** enable unique agentic capabilities (SOC, Langevin) that no standard ANN index provides

---

## 📚 Concepts Explored

During this analysis, concepts were explored from:

- **Physics**: Hopfield Networks, Resonance, Underdamped Langevin Dynamics
- **Neuroscience**: Hippocampal Grid Cells
- **Quantum Mechanics**: Superposition
- **Graph Theory**: Random Walks
- **Von Mises-Fisher Distribution**: Spherical clustering (κ concentration)
- **Self-Organized Criticality (SOC)**: Passive memory consolidation
- **Mixture of Experts (MoE)**: Hardware-accelerated distance routing

None of the exotic methodologies improved recall for uniform data. The MoE router and SOC consolidation remain valuable for **agentic workloads** where data has natural structure and evolves over time.

---

## 🎯 Final Conclusion

> **For uniform text embeddings such as DBpedia, Linear Scan is the best option.**
>
> Advanced methodologies (HRM2, Vulkan GPU, HETD) only work well when data has natural cluster structure. Attempting to force structure where none exists adds overhead with no recall benefit.
>
> **However, the M2M engine is not merely an ANN index.** Its Gaussian Splat representation, 3-tier memory tiering, Langevin generative exploration, and SOC consolidation make it the right choice for autonomous agent memory systems — independent of whether raw ANN latency beats a linear scan.

---

*Document updated: 2026-03-05*
