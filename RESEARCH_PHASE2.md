# Phase 2 Research Report — M2M Production with Z.AI Tools

**Date:** 2026-03-18
**Research Tool:** Z.AI Tools (GLM-5 web_search, github_analyze)
**Status:** Complete

---

## 1. Research Sources

### 1.1 Hybrid Search Fusion Methods
**Query:** "reciprocal rank fusion vs weighted score fusion vs other hybrid search ranking methods"

**Key Findings:**
- **RRF** is the industry standard (Elasticsearch, OpenSearch, Pinecone). Score-agnostic, works with default k=60. Best "set it and forget it" choice.
- **Weighted Score Fusion** needs min-max normalization (critical!). Better if tuned, but requires labeled data. Distribution drift is a risk.
- **Cross-Encoder Reranking** has highest accuracy but highest latency. Standard pattern: RRF → gather top 50 → reranker → top 10.
- **Learning to Rank (LTR)** requires user click logs. Not applicable for personal memory.

**Decision:** RRF as default. Added weighted as option. Cross-encoder noted as future enhancement.

### 1.2 Embedding Models
**Query:** "sentence-transformers best models 2025: all-MiniLM-L6-v2 vs bge-small vs gte-small"

**Key Findings:**
| Model | Speed | MTEB Quality | Notes |
|-------|-------|-------------|-------|
| all-MiniLM-L6-v2 | ⚡ Fastest | Good | Aging, older training data |
| bge-small-en-v1.5 | 🚀 Very Fast | Better | Best all-rounder (384D) |
| gte-small | 🚀 Very Fast | Best | Highest MTEB for this size |
| all-MiniLM-L12-v2 | 🐢 Slower | Good | **Avoid** — slower, not better |

**Decision:** Recommend bge-small-en-v1.5 for production. Updated documentation.

### 1.3 Vector Compression Techniques
**Query:** "vector database production best practices 2025: compression, scaling"

**Key Findings:**
- **Scalar Quantization (int8):** 4x memory reduction, <1% recall loss. Best for general production.
- **Binary Quantization:** 32x reduction, but needs oversampling (10-100x) + reranking.
- **Matryoshka Embeddings:** Store full vector, index truncated. Future consideration.
- **Production trend:** Storage/compute disaggregation, tenant-based sharding, time-based sharding for RAG.

**Decision:** Not implemented yet (AlfredMemoryDB targets 1-10K memories where compression isn't critical). Noted as v2.3 roadmap item.

### 1.4 Index Algorithms (HNSW vs IVF vs PQ)
**Query:** "HNSW vs IVF vs product quantization: accuracy speed memory tradeoffs"

**Key Findings:**
- **HNSW:** Best speed/accuracy, O(log N) search, but 30-50% memory overhead from graph edges.
- **IVF:** Good balance. Tunable via nprobe parameter. More memory-efficient than HNSW.
- **PQ:** Compression technique (not standalone index). Lossy. Best combined with IVF.
- **For 1-10K scale:** None of these are necessary. Linear search is competitive.

**Decision:** Current M2M HRM2 + linear is appropriate for Alfred's scale. HNSW/IVF considered for v3.0.

### 1.5 API Design Patterns
**Query:** Qdrant repo analysis via github_analyze

**Key Findings:**
- RESTful + gRPC dual protocol
- Filtering as first-class citizen (nested conditions)
- Idempotent operations (upsert)
- Statefulless design (no server-side sessions)
- Pagination via limit/offset

**Decision:** AlfredMemoryDB already follows similar patterns. Added `filter` parameter support.

---

## 2. Implementation Decisions

### What was implemented:
1. **Multiple fusion methods** (rrf, weighted, vector_only, bm25_only)
2. **Temporal decay** with configurable half-life (exponential decay)
3. **Auto-categorization** (10 categories, keyword-based)
4. **Auto-date** (automatic metadata enrichment)
5. **Score normalization** (min-max for weighted fusion)
6. **Input validation** (empty/whitespace text rejection)
7. **63 new chaos tests** covering edge cases

### What was NOT implemented (with rationale):
- **Scalar Quantization:** Not needed for 1-10K memories. Would add complexity. Roadmap v2.3.
- **HNSW/IVF:** Linear search is competitive at Alfred's scale. Roadmap v3.0.
- **Cross-encoder reranking:** Too slow for real-time personal assistant. Could be added as optional post-processing step.
- **Better embedding model switching:** Currently configurable via encoder parameter. Documented recommendations.

---

## 3. Test Results

- **Phase 1 tests:** 103/103 ✅ (unchanged)
- **Phase 2 new tests:** 63/63 ✅
- **Total:** 166/166 ✅
- **Time:** 42.19s

### New test categories:
- `TestAutoCategorize` (12 tests) — keyword matching, unicode, edge cases
- `TestTemporalDecay` (5 tests) — decay math, half-life, clamping
- `TestFusionMethods` (6 tests) — all 4 fusion methods + validation
- `TestScoreNormalization` (4 tests) — min-max edge cases
- `TestAutoCategorizeIntegration` (5 tests) — integration with store/batch_store
- `TestValidation` (4 tests) — input validation
- `TestChaosUnicode` (7 tests) — emoji, mixed scripts, long text, rare unicode
- `TestChaosBM25EdgeCases` (10 tests) — empty, single, duplicates, no matches
- `TestChaosMemoryDB` (6 tests) — large batches, clear, delete, stats
- `TestChaosConcurrent` (2 tests) — rapid cycles, interleaved ops
- `TestBM25Tokenizer` (2 tests) — Spanish chars, numbers

---

## 4. Files Created/Modified

| File | Action | Description |
|------|--------|-------------|
| `src/m2m/alfred_memory.py` | Modified | Added fusion methods, temporal decay, auto-categorize, validation |
| `tests/test_phase2_features.py` | Created | 63 new tests |
| `scripts/index_alfred_workspace.py` | Created | Workspace indexing script |
| `README.md` | Modified | Updated to v2.2 with Alfred Memory docs, research notes |
| `RESEARCH_PHASE2.md` | Created | This file |
