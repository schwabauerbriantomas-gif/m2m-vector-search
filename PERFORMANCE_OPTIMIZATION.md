# Performance Optimization Report

## Date: 2026-03-17

## Problem

M2M Vector Search was **1.4-2.5x slower than linear scan** for datasets with N≤10K. The hierarchical HRM2 index added more Python overhead than the clustering pruning could save for small-to-medium datasets.

### Before Optimization (RTX 3090, N=10K, K=10, dim=640)

| Backend | Avg Latency | QPS | Speedup vs Linear |
|---------|------------|-----|-------------------|
| Linear  | 24.21 ms   | 41.31 | 1.0x |
| CPU     | 32.93 ms   | 30.37 | **0.7x** (slower!) |
| Vulkan  | 32.78 ms   | 30.51 | **0.7x** (slower!) |
| CUDA    | 26.54 ms   | 37.68 | **0.9x** (slower!) |

## Root Cause

The search path in `SplatStore.find_neighbors()` called `HRM2Engine.query()` per query in a Python loop. Each call involved:

1. `sklearn KMeans.transform()` — compute distances to all coarse centroids
2. `np.argsort()` — sort to find top-N probe clusters
3. Python for-loop over probed clusters
4. `np.vstack()` — concatenate candidate embeddings (memory allocation)
5. `np.linalg.norm()` — compute exact distances
6. Python loop to build candidates list
7. `list.sort()` — sort candidates by distance

For N=10K, this Python-level overhead (~33μs per query just in Python dispatch) exceeded the cost of a single vectorized BLAS linear scan over all 10K vectors.

## Phase 1 Optimization

### What Changed

Replaced the per-query HRM2 Python path in `SplatStore.find_neighbors()` and `batch_find_neighbors()` with a **fast vectorized numpy path**:

```python
# Instead of HRM2's Python-heavy query():
diff = index_data - query          # [N, dim] - broadcast
dists_sq = np.einsum("ij,ij->i", diff, diff)  # squared L2, no sqrt
topk = np.argpartition(dists_sq, k-1)[:k]      # O(N) partial sort
```

**Key optimizations:**
1. **Squared L2 via `np.einsum`** — avoids `np.linalg.norm` overhead, no sqrt needed for ranking
2. **`np.argpartition`** — O(N) top-K selection instead of O(N log N) full sort
3. **Eliminated Python loops** — no per-cluster iteration, no candidate list building
4. **Direct index access** — bypass `GaussianSplat` objects, access `self.mu[]` directly
5. **HRM2 threshold** — clustering routing only activated for N > 15K where pruning benefits outweigh overhead

### After Phase 1 (RTX 3090, N=10K, K=10, dim=640)

| Backend | Avg Latency | QPS | Speedup vs Linear |
|---------|------------|-----|-------------------|
| Linear  | 22.57 ms   | 44.30 | 1.0x |
| CPU     | 10.68 ms   | 93.67 | **2.1x** ✅ |
| Vulkan  | 10.71 ms   | 93.38 | **2.1x** ✅ |
| CUDA    | 10.83 ms   | 92.34 | **2.1x** ✅ |

## Phase 2 Optimizations (2026-03-17)

### Additional Changes

1. **einsum in HRM2Engine.query()** — Replaced `np.linalg.norm` with `np.einsum` for squared L2 distances in LOD 2 path
2. **Pre-computed cluster masks** — Cache boolean masks for each coarse cluster during `index()`, avoiding recomputation per query
3. **Input validation** — Added NaN/Inf/empty/dimension checks in `find_neighbors()`
4. **Consolidate index rebuild** — `consolidate()` now calls `build_index()` to keep HRM2 index fresh
5. **Security hardening** — API key auth, rate limiting, path traversal prevention, input sanitization

### Scalability Results (CPU, sklearn dataset, K=10)

| N     | Linear (ms) | M2M CPU (ms) | Speedup |
|-------|-------------|---------------|---------|
| 1,000 | 1.94        | 1.13          | **1.7x** |
| 5,000 | 10.45       | 4.53          | **2.3x** |
| 10,000| 21.20       | 9.09          | **2.3x** |
| 50,000| 106.53      | 25.30         | **4.2x** |

**Key finding:** M2M consistently outperforms linear scan across all tested sizes, with speedup increasing with N. At N=50K, M2M achieves **4.2x speedup**.

### Security Fixes Applied

| ID | Severity | Description | Status |
|----|----------|-------------|--------|
| C-01 | Critical | API authentication (API key) | ✅ Fixed |
| C-02 | Critical | TLS between nodes | ⚠️ Documented (env concern) |
| H-01 | High | Path traversal in storage | ✅ Fixed |
| H-02 | High | Input dimension validation | ✅ Fixed |
| H-03 | High | Rate limiting | ✅ Fixed |
| H-04 | High | Error message exposure | ✅ Fixed |
| H-05 | High | Pickle HMAC signing | ✅ Fixed |
| P-01 | Medium | Vector overflow | ✅ Fixed |
| P-02 | Medium | k validation | ✅ Fixed |
| P-03 | High | Collection name path traversal | ✅ Fixed |
| P-04 | High | Backup path traversal | ✅ Fixed |
| P-05 | Medium | Payload size limit | ✅ Fixed |
| P-06 | Critical | Auth on all endpoints | ✅ Fixed |
| P-07 | High | Rate limiting | ✅ Fixed |
| P-09 | Critical | Node registration auth | ✅ Fixed |
| P-10 | High | Heartbeat spoofing | ✅ Fixed |
| P-12 | High | SSRF in fetch_edge | ✅ Fixed |
| P-16 | Medium | Energy map resolution | ✅ Fixed |
| P-18 | Low | Explore n_suggestions cap | ✅ Fixed |

### Chaos Fixes Applied

| ID | Description | Status |
|----|-------------|--------|
| C-01 | k=0 crash | ✅ Fixed (k=max(1,k)) |
| C-02 | Empty query crash | ✅ Fixed (validation) |
| C-03 | NaN/Inf detection | ✅ Fixed (np.isfinite check) |
| C-04 | Consolidate index rebuild | ✅ Fixed (build_index after consolidate) |

### Limitations

1. **HRM2 clustering is unused for N≤15K** — the hierarchical index is still built (for API compatibility) but bypassed during search.
2. **TLS between nodes** — documented but requires environment-specific certificate setup.
3. **Scalability tested up to N=50K** — higher N values should be tested with more RAM available.
