# M2M Vector Search — Improvement Specs

**Baseline (DBpedia 10K, 640D, k=10):**
| Method | Time (ms) | QPS | Recall@10 |
|--------|-----------|-----|-----------|
| Linear scan | 9.85 | ~101 | 1.000 |
| M2M CPU (HRM2) | 22.83 | ~44 | ? |
| M2M Vulkan | 33.10 | ~30 | ? |

**Hardware target:** NVIDIA RTX 3090 24GB, PyTorch CUDA

**Core problem:** M2M is 2-3× slower than linear scan on 10K vectors. The hierarchical index overhead (coarse→fine clustering + candidate gathering) exceeds the brute-force cost at this scale.

---

## SPEC 1: CUDA Brute-Force Backend (Priority: CRITICAL)

**Paper:** None (infrastructure prerequisite for all other specs)  
**Component:** `engine.py`, `splats.py` (SplatStore), `gpu_vector_index.py`  
**What:** Replace Vulkan compute-shader path with PyTorch CUDA matmul-based kNN.

### Why
- RTX 3090 has 24GB VRAM — fits 100K × 640D float32 = ~256MB
- PyTorch CUDA matmul is heavily optimized (cuBLAS). Single fused kernel vs Vulkan dispatch overhead.
- Current Vulkan path: 33.10ms with shader compilation + buffer management overhead
- Expected: brute-force on GPU with [B,N,D] matmul should be <2ms for 10K vectors

### Changes
1. **New file:** `src/m2m/cuda_search.py`
   ```python
   class CUDASearch:
       def __init__(self, index: np.ndarray, device='cuda'):
           # Upload index to GPU once
           self.index = torch.from_numpy(index).pin_memory().to(device, non_blocking=True)
       
       def batch_search(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
           # [B,D] @ [D,N] → [B,N] squared distances via einsum
           # Or: ‖q - x‖² = ‖q‖² + ‖x‖² - 2q·x
           Q = torch.from_numpy(queries).to(self.index.device)
           # Pre-computed index_norms = ‖x‖² [N]
           dots = Q @ self.index.T  # [B, N]
           dists = q_norms - 2*dots + self.index_norms  # broadcast
           topk = torch.topk(dists, k, largest=False, dim=1)
           return topk.indices.cpu().numpy(), topk.values.cpu().numpy()
   ```

2. **Modify:** `splats.py` SplatStore.batch_find_neighbors() — add CUDA path before Vulkan fallback
3. **Modify:** `engine.py` M2MEngine — prefer CUDA over Vulkan when both available

### Baseline → Expected
| Metric | Before | After |
|--------|--------|-------|
| Query time (10K) | 33.10ms Vulkan | **<2ms** CUDA |
| QPS (10K) | ~30 | **>500** |
| Recall@10 | 1.0 (exact) | 1.0 (exact) |

### Verification
```python
# benchmark_cuda_search.py
queries = embeddings[:100]
%timeit cuda_search.batch_search(queries, k=10)
# Compare results against numpy brute-force for correctness
```

### Files
- **Create:** `src/m2m/cuda_search.py`
- **Modify:** `src/m2m/splats.py` (batch_find_neighbors)
- **Modify:** `src/m2m/engine.py` (M2MEngine.init)
- **Create:** `tests/test_cuda_search.py`

### Time: 2-3 hours

---

## SPEC 2: Adaptive Probe Count (Priority: HIGH)

**Paper:** HRM (2506.21734) — L-module (fast, coarse) + H-module (slow, precise). Adaptive allocation of computation.  
**Component:** `hrm2_engine.py` (query method), `splats.py` (find_neighbors)  
**What:** Replace fixed `n_probe=5` with adaptive probing based on query difficulty.

### Why
- Fixed n_probe=5 wastes time on easy queries (1-2 probes suffice) and loses recall on hard queries (boundary between clusters)
- HRM paper shows L-module (fast pass) can determine if H-module (slow pass) is needed
- For 10K dataset: most queries only need n_probe=1-2 to match linear recall

### Changes
1. **Modify:** `hrm2_engine.py` HRM2Engine.query()
   ```python
   def query(self, query_vector, k=10, lod=2):
       # Phase 1 (L-module): probe top-1 coarse cluster, compute distances
       coarse_dists = self.coarse_model.transform(q.reshape(1,-1))[0]
       top1 = np.argmin(coarse_dists)
       candidates = self._get_cluster_candidates(top1)
       dists = self._compute_distances(q, candidates)
       
       # Early termination: if top-k gap is large enough, skip more probes
       if self._is_easy_query(dists, k):
           return self._topk(candidates, dists, k)
       
       # Phase 2 (H-module): progressively add coarse clusters
       for probe in range(1, self.n_probe_max):
           next_cluster = np.argsort(coarse_dists)[probe]
           new_candidates, new_dists = self._add_cluster(next_cluster, candidates, dists)
           if self._recall_sufficient(new_dists, dists[:k]):
               break
           candidates, dists = new_candidates, new_dists
       
       return self._topk(candidates, dists, k)
   
   def _is_easy_query(self, dists, k):
       """If gap between k-th and (k+1)-th is large, query is easy."""
       if len(dists) < k + 1:
           return False
       sorted_d = np.sort(dists)
       return sorted_d[k] < 0.5 * sorted_d[k-1]  # k-th is far from (k-1)-th → boundary issue unlikely
   ```

2. **Modify:** `splats.py` — pass through adaptive n_probe

### Baseline → Expected
| Metric | Before | After |
|--------|--------|-------|
| Avg query time (CPU, 10K) | 22.83ms | **8-12ms** |
| Recall@10 | ~0.95 (est) | **>0.98** |

### Verification
- Run recall@10 against linear scan ground truth for 1000 random queries
- Measure avg query time with timeit

### Files
- **Modify:** `src/m2m/hrm2_engine.py`
- **Create:** `tests/test_adaptive_probe.py`

### Time: 3-4 hours

---

## SPEC 3: CUDA-Accelerated Coarse Distances (Priority: HIGH)

**Paper:** HRM (2506.21734) — L-module acceleration  
**Component:** `hrm2_engine.py`, `engine.py`  
**What:** Move coarse cluster distance computation to CUDA.

### Why
- `coarse_model.transform()` uses sklearn's CPU implementation for every query
- With 100 coarse centroids × 640D, this is a small matmul — trivial on GPU but adds Python/sklearn overhead on CPU
- For batch queries: [B,100] @ [100,640] → single GPU call

### Changes
1. **New method in** `cuda_search.py`:
   ```python
   def compute_coarse_distances(self, queries: np.ndarray, centroids: np.ndarray) -> np.ndarray:
       Q = torch.from_numpy(queries).to(self.device)
       C = torch.from_numpy(centroids).to(self.device)
       return (Q @ C.T).cpu().numpy()
   ```

2. **Modify:** `hrm2_engine.py` — use CUDA coarse distances when available

### Baseline → Expected
| Metric | Before | After |
|--------|--------|-------|
| Coarse dist (100 queries) | ~2ms CPU | **<0.1ms** CUDA |

### Files
- **Modify:** `src/m2m/cuda_search.py`
- **Modify:** `src/m2m/hrm2_engine.py`

### Time: 1-2 hours

---

## SPEC 4: Precomputed Index Norms + Fused Distance (Priority: HIGH)

**Paper:** None (standard ANN optimization)  
**Component:** `splats.py`, `cuda_search.py`  
**What:** Precompute ‖x‖² for all index vectors, use expanded distance formula.

### Why
- Current: `diff = x - q; dist_sq = einsum('ij,ij->i', diff, diff)` — creates [N,D] diff array
- Optimized: `dist_sq = ‖q‖² + ‖x‖² - 2q·x` — no diff array, just matmul
- For 10K × 640D: saves 10K×640 float32 allocation per query (25MB)

### Changes
1. **In** `cuda_search.py.__init__()`:
   ```python
   self.index_norms = (self.index ** 2).sum(dim=1)  # [N], computed once
   ```
2. **In** `splats.py` CPU path: precompute `index_norms` at `build_index()`

### Files
- **Modify:** `src/m2m/cuda_search.py`
- **Modify:** `src/m2m/splats.py`

### Time: 1 hour

---

## SPEC 5: Multi-Start Search with Energy-Guided Initialization (Priority: MEDIUM)

**Paper:** HRM Mechanistic Analysis (2601.10679) — multiple fixed points, input perturbation (+18.7% accuracy)  
**Component:** `splats.py`, `energy.py`, new `src/m2m/multistart_search.py`  
**What:** For queries near cluster boundaries, launch multiple search starting points with small perturbations and merge results.

### Why
- Hard queries (near cluster boundaries) lose recall because the query falls in one cluster but true neighbors are in adjacent clusters
- HRM analysis shows perturbation of inputs improves recall by 18.7%
- Multi-start: perturb query by small Gaussian noise → run kNN from each perturbed point → merge results

### Changes
1. **New file:** `src/m2m/multistart_search.py`
   ```python
   class MultiStartSearch:
       def __init__(self, n_starts=3, perturbation_scale=0.05):
           self.n_starts = n_starts
           self.sigma = perturbation_scale
       
       def search(self, base_query, search_fn, k):
           results = {}
           for i in range(self.n_starts):
               if i == 0:
                   q = base_query
               else:
                   q = base_query + np.random.randn(len(base_query)) * self.sigma
               indices, dists = search_fn(q, k)
               for idx, d in zip(indices, dists):
                   if idx not in results or d < results[idx]:
                       results[idx] = d
           sorted_results = sorted(results.items(), key=lambda x: x[1])
           return sorted_results[:k]
   ```

2. **Modify:** `splats.py` — optional multi-start wrapper

### Baseline → Expected
| Metric | Before | After |
|--------|--------|-------|
| Recall@10 (boundary queries) | ~0.85 (est) | **>0.95** |
| Query time | 1× | ~2-3× (multi-start overhead) |
| Net QPS | - | Still faster than linear for large N (>50K) |

### Verification
- Identify boundary queries (queries where top-k results span multiple coarse clusters)
- Measure recall improvement specifically for this subset

### Files
- **Create:** `src/m2m/multistart_search.py`
- **Modify:** `src/m2m/splats.py`
- **Create:** `tests/test_multistart.py`

### Time: 2-3 hours

---

## SPEC 6: Langevin Energy-Guided Search (Priority: MEDIUM)

**Paper:** Stochastic Attention via Langevin on Hopfield Energy (2603.06875)  
**Component:** `energy.py`, new `src/m2m/langevin_search.py`  
**What:** Instead of kNN distance, use energy landscape gradient descent to find nearest attractors (splats). Temperature controls exploration.

### Why
- Energy function E(x) = -log(Σ αᵢ exp(-κᵢ‖x - μᵢ‖²)) already exists in `energy.py`
- Langevin dynamics: x_{t+1} = x_t - ε∇E(x_t) + √(2εT)·ξ — naturally converges to nearest splat attractor
- Multiple starting points (multi-start) find multiple attractors → better recall near boundaries
- Temperature T controls how far the search explores before settling

### Changes
1. **New file:** `src/m2m/langevin_search.py`
   ```python
   class LangevinSearch:
       def __init__(self, energy_fn, splats, n_steps=20, dt=0.01, temperature=0.1):
           self.energy = energy_fn
           self.splats = splats
           self.n_steps = n_steps
           self.dt = dt
           self.T = temperature
       
       def search(self, query, k=10):
           x = torch.tensor(query, device='cuda', dtype=torch.float32, requires_grad=True)
           visited_splats = {}
           
           for step in range(self.n_steps):
               energy = self._compute_energy_cuda(x)
               grad = torch.autograd.grad(energy, x)[0]
               
               noise = torch.randn_like(x) * np.sqrt(2 * self.dt * self.T)
               x = x - self.dt * grad + noise
               
               # Track nearest splat at each step
               nearest = self._find_nearest_splat(x.detach())
               visited_splats[nearest['id']] = min(
                   visited_splats.get(nearest['id'], float('inf')),
                   nearest['dist']
               )
           
           return sorted(visited_splats.items(), key=lambda x: x[1])[:k]
   ```

2. **Modify:** `energy.py` — add CUDA-compatible energy computation
3. **Modify:** `splats.py` — add langevin search as alternative LOD level

### Expected behavior
- At low T (0.01): converges to nearest attractor — equivalent to gradient descent kNN
- At high T (0.5): explores multiple attractors — better recall, slower
- Sweet spot: T=0.1, n_steps=20

### Files
- **Create:** `src/m2m/langevin_search.py`
- **Modify:** `src/m2m/energy.py`
- **Modify:** `src/m2m/splats.py`
- **Create:** `tests/test_langevin_search.py`

### Time: 4-6 hours

---

## SPEC 7: Flow Matching Query Refinement (Priority: LOW)

**Paper:** Flow Matching (2210.02747) + Generalized FM for Transitions (2410.15128)  
**Component:** New `src/m2m/flow_search.py`  
**What:** Train a simple velocity field v(x,t) that maps queries to their nearest neighbors. At inference, integrate ODE from query toward nearest cluster center.

### Why
- For repeated/similar queries, a learned velocity field can shortcut the search
- FM between meta-stable states (query → nearest neighbor) is simpler than full nearest-neighbor search
- Especially useful for RAG workloads where queries tend to cluster in embedding space

### Changes
1. **New file:** `src/m2m/flow_search.py`
   ```python
   class FlowMatchingSearch:
       def __init__(self, dim=640, hidden_dim=256):
           # Small MLP velocity field: v(x, t) → Δx
           self.net = nn.Sequential(
               nn.Linear(dim + 1, hidden_dim),  # +1 for time t
               nn.GELU(),
               nn.Linear(hidden_dim, hidden_dim),
               nn.GELU(),
               nn.Linear(hidden_dim, dim),
           )
       
       def train(self, query_neighbor_pairs):
           # pairs: (query, nearest_neighbor) pairs from training data
           # Train: v(x_0, 0) should move x_0 toward x_1
           x0 = queries   # start
           x1 = neighbors  # target
           t = torch.rand(B, 1)
           xt = (1 - t) * x0 + t * x1
           v_target = x1 - x0
           v_pred = self.net(torch.cat([xt, t], dim=1))
           loss = F.mse_loss(v_pred, v_target)
       
       def search(self, query, n_steps=5, dt=0.2):
           x = query.clone()
           for step in range(n_steps):
               t = torch.tensor(step * dt)
               v = self.net(torch.cat([x, t.reshape(1,1)], dim=1))
               x = x + dt * v
           # x is now near the nearest neighbor region
           # Fall back to standard kNN from this refined point
           return standard_knn(x, k)
   ```

### Limitations
- Requires training data (query→neighbor pairs)
- Adds model overhead (inference through MLP)
- Only beneficial if query distribution is concentrated (not uniform random)

### Files
- **Create:** `src/m2m/flow_search.py`
- **Create:** `tests/test_flow_search.py`

### Time: 6-8 hours

---

## SPEC 8: HRM2 Threshold Strategy for Large-Scale (Priority: LOW)

**Paper:** HRM (2506.21734)  
**Component:** `hrm2_engine.py`  
**What:** For N > 50K, use HRM2 with higher n_probe. For N < 15K, bypass clustering entirely (use CUDA brute-force).

### Why
- Current threshold `use_hrm2 = n > 15000` in `splats.py` is correct but HRM2 itself is slow
- With CUDA brute-force (Spec 1), linear scan is faster for N < 50-100K
- HRM2 becomes valuable only at N > 100K where CUDA brute-force exceeds budget

### Changes
1. **Modify:** `splats.py`
   ```python
   use_hrm2 = n > 100_000 and self.engine.coarse_model is not None
   ```

### Time: 15 minutes

---

## Implementation Priority Order

| Order | Spec | Impact | Effort | Cumulative Expected Result |
|-------|------|--------|--------|---------------------------|
| 1 | Spec 1: CUDA Brute-Force | 🔴 Critical | 2-3h | 10K: <2ms, >500 QPS (10× faster than linear) |
| 2 | Spec 4: Precomputed Norms | 🔴 High | 1h | Further 20-30% speedup on CUDA |
| 3 | Spec 3: CUDA Coarse Dists | 🟡 High | 1-2h | Enables fast HRM2 at scale |
| 4 | Spec 8: HRM2 Threshold | 🟢 Low | 15min | Correct routing for all N |
| 5 | Spec 2: Adaptive Probe | 🟡 High | 3-4h | HRM2 competitive at N > 100K |
| 6 | Spec 5: Multi-Start | 🟡 Medium | 2-3h | +10-15% recall on boundary queries |
| 7 | Spec 6: Langevin Search | 🟢 Medium | 4-6h | Novel energy-based search |
| 8 | Spec 7: Flow Matching | 🟢 Low | 6-8h | Future: learned query refinement |

---

## Recommended First Sprint (Specs 1 + 4 + 3 + 8)

**Total time: ~4-6 hours**

After this sprint, expected results on DBpedia 10K:
| Method | Time | QPS | Recall@10 |
|--------|------|-----|-----------|
| Linear scan | 9.85ms | ~101 | 1.000 |
| M2M CUDA (new) | **<2ms** | **>500** | 1.000 |

This already achieves the primary goal: **M2M faster than linear scan**.

Second sprint (Specs 2 + 5) addresses scalability to 100K+ and recall on hard queries.
