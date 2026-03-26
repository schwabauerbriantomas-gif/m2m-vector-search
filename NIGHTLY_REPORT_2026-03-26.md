# Nightly Report 2026-03-26

**Executed:** 2026-03-26 03:00 ART (06:00 UTC)
**Duration:** ~6 minutes

---

## M2M Tests: 415/418 pass (99.3%)
- 415 passed, 3 skipped, 0 failed
- Runtime: 169.46s
- Warnings: 18 (HRM2 silhouette < 0.1 - expected for random embeddings, SwigPyPacked deprecation)
- Skipped tests likely due to optional dependencies

## M2M Benchmarks (dim=640, k=10, 10K splats)
| Backend       | Avg Latency (ms) | P95 (ms) | QPS    | Speedup vs Linear |
|---------------|-----------------|----------|--------|-------------------|
| Linear Scan   | 22.62           | --       | 44.20  | 1.0x              |
| CPU M2M       | 10.64           | 11.66    | 93.99  | 2.1x              |
| Vulkan GPU    | 10.72           | 11.80    | 93.30  | 2.1x              |
| CUDA          | 11.26           | 13.99    | 88.81  | 1.9x              |

- HRM2 silhouette: 0.0140 (expected for random data)
- Transformed benchmark: ERROR (TransformConfig missing enable_3_tier_memory attribute)
- Note: CPU and Vulkan perform nearly identically; CUDA slightly slower (likely kernel launch overhead at this scale)

## M2M RAG Benchmark (20 queries, 659 chunks, dim=384)
| Metric       | Value  |
|--------------|--------|
| Precision@5  | 0.830  |
| Recall@5     | 1.000  |
| Top Similarity | 0.649 |
| Linear Scan  | 8.44ms |

- M2M search not tested: dimension mismatch (expected 640, dataset is 384)
- 18/20 queries had P@5 >= 0.60

## M2M Energy Validation (100 splats, 1000 random points, dim=640)
| Component | Mean    | Std     | Min     | Max     |
|-----------|---------|---------|---------|---------|
| E_splats  | 15.087  | 0.089   | 14.753  | 15.378  |
| E_geom    | 0.000   | 0.000   | 0.000   | 0.000   |
| E_total   | 15.087  | 0.089   | 14.753  | 15.378  |

- NaN count: 0 | Inf count: 0
- Energy at splat center: ~0.0000 (correct - near-zero at attractor)
- Energy at random point: 15.14 (correct - high energy far from attractors)
- Near splats (first 50 splat positions) mean: ~0.0000
- Far random points mean: 5.49
- **Sign convention: PASS** (negative/zero near splats, positive far away)

## M2M HRM2 Clustering Quality (10K embeddings, 100 coarse clusters)
| Metric           | Value  |
|-----------------|--------|
| Silhouette       | 0.0189 |
| Calinski-Harabasz| 11.20  |
| Build Time       | 9.33s  |
| Diagnostics CH   | 11.2   |

- Low silhouette expected for uniform random embeddings
- 100 coarse clusters as configured

## EBM Tests: 41/41 pass (100%)
- Runtime: 18.68s
- All categories passing: ScoreNetwork, Energy, Langevin, ContextHierarchy, SOC, Config, Decoder

## EBM ScoreNetwork (dim=640, batch=100)
| Metric         | Value     |
|----------------|-----------|
| Output shape   | [100, 640]|
| Output mean    | -0.0000   |
| Output std     | 0.0070    |
| Gradient norm  | 21514.75  |
| NaN/Inf        | 0/0       |

## EBM Energy (500 points, 50 splats, dim=640)
- Could not collect standalone metrics (EnergyFunction requires SOC splat_store, API changed since last audit)
- Tests validate: energy near splat is lower, no NaN/Inf, gradient flows correctly

## EBM Langevin Dynamics
- Tests validate: energy decreases over 500 steps, no NaN, stays on unit sphere, step acceptance working
- Real metrics collected by test suite (41/41 pass confirms)

## EBM Decoder
- Tests validate: valid token range (0 <= token < 50257), no infinite repetition, unique token ratio > 0
- Real metrics collected by test suite

## EBM SOC Controller
- Tests validate: consolidation reduces splat count, search works after consolidation, no NaN

## EBM Config
- V2 params present: beta_global, beta_local, beta_medium, ema_decay, context windows - **PASS**
- vocab_size=50257 - **PASS**

## Security Audit
| Severity | Count | Details |
|----------|-------|---------|
| Critical | 0     | No hardcoded secrets, no shell=True, no os.system in project code |
| Medium   | 1     | `pickle.loads` in persistence.py:338 (deserialization of saved index data) |
| Low      | 5     | `eval()` calls: validate_project.py:151, evaluate_embeddings.py:104,224, train_embeddings.py:377,501 - all are `model.eval()` (PyTorch eval mode, not Python eval) |
| Info     | 0     | No hardcoded API keys, no .env files with secrets |
| OK       | 1     | `torch.load` always uses `weights_only=True` (3 occurrences) |
| OK       | 1     | `M2M_HMAC_SECRET` not hardcoded (tests verify it's required) |

- **M2M-specific findings:** 6 total, 0 critical
- **EBM-specific findings:** None (all eval/exec/pickle hits are in third-party packages)

## Performance Profiling
- Top bottleneck from benchmark: M2M search at ~10.6ms per query (dominates)
- CUDA backend init overhead: 2838ms (vs 83ms Vulkan, negligible CPU)
- Transform benchmark error: missing `enable_3_tier_memory` attribute on TransformConfig

## Code Quality
| Metric              | Count |
|---------------------|-------|
| TODO/FIXME/HACK     | 2     |
| Files > 500 lines   | 1     | (src/m2m/__init__.py: 1247 lines) |
| pycodestyle         | Not installed |
| torch.load safe     | 3/3 use weights_only=True |

## Fixes Applied
- None required (all tests pass)

## Regressions
- **TransformConfig missing `enable_3_tier_memory`**: benchmark error on transformed backend. Not a regression (feature gap) but prevents transformed benchmark from completing.
- **M2M RAG benchmark dim mismatch**: RAG dataset uses dim=384 (MiniLM-L6-v2) but M2M expects dim=640. Not a regression, architectural mismatch.

## Recommendations
1. **Split `src/m2m/__init__.py`** (1247 lines) into submodules for maintainability
2. **Add `enable_3_tier_memory`** to TransformConfig to fix transformed benchmark
3. **Investigate HRM2 low silhouette** on random data - consider using HNSW as fallback (already implemented)
4. **CUDA backend init time** (2.8s) - investigate if warm-start caching is possible
5. **pickle.loads in persistence.py** - consider alternative serialization or HMAC validation (HMAC is already used for index files, but pickle itself is risky)
