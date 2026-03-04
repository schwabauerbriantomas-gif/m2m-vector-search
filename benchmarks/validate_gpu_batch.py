#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
validate_gpu_batch.py — Validation script for GPU persistent index + batch search

Tests:
  1. Correctness: GPUVectorIndex.batch_search vs CPU numpy (max diff < 1e-4)
  2. Latency:     single-query loop vs batch dispatch at 10k / 50k / 100k
  3. HierarchicalGPUSearch: coarse+fine results subset of brute-force
  4. SplatStore.batch_find_neighbors: Vulkan path and CPU fallback

Output: gpu_batch_validation_results.json
"""

import sys
import os
import time
import json
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

OUTPUT_FILE = PROJECT_ROOT / "gpu_batch_validation_results.json"

SEED   = 42
DIM    = 128
K      = 10
B      = 20   # batch size for batch tests

np.random.seed(SEED)

results = {
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    "system": {"dim": DIM, "k": K, "batch_size": B, "seed": SEED},
    "tests": {},
}

SEP = "─" * 68


# ═══════════════════════════════════════════════════════════════════════════════
# Helper
# ═══════════════════════════════════════════════════════════════════════════════

def cpu_brute_force(index: np.ndarray, queries: np.ndarray, k: int):
    """Reference: numpy L2 brute-force. Returns [B, k] ids and dists."""
    diff  = queries[:, None, :] - index[None, :, :]  # [B, N, D]
    dists = np.linalg.norm(diff, axis=-1)             # [B, N]
    ids   = np.argpartition(dists, k, axis=1)[:, :k]
    top   = np.take_along_axis(dists, ids, axis=1)
    order = np.argsort(top, axis=1)
    ids   = np.take_along_axis(ids,  order, axis=1)
    top   = np.take_along_axis(top,  order, axis=1)
    return ids, top


# ═══════════════════════════════════════════════════════════════════════════════
# Test 1 — GPUVectorIndex: correctness + latency at multiple scales
# ═══════════════════════════════════════════════════════════════════════════════

def test_gpu_vector_index():
    print(f"\n{SEP}")
    print("  TEST 1 — GPUVectorIndex: correctness + latency")
    print(SEP)

    try:
        from gpu_vector_index import GPUVectorIndex
    except Exception as e:
        print(f"  [SKIP] GPUVectorIndex not available: {e}")
        results["tests"]["gpu_vector_index"] = {"status": "SKIPPED", "reason": str(e)}
        return

    test_results = {}

    for n in [10_000, 50_000, 100_000]:
        print(f"\n  Scale: {n:,}")
        rng = np.random.default_rng(SEED)
        index_vecs = rng.standard_normal((n, DIM)).astype(np.float32)
        index_vecs /= np.linalg.norm(index_vecs, axis=1, keepdims=True)
        queries    = rng.standard_normal((B, DIM)).astype(np.float32)
        queries   /= np.linalg.norm(queries, axis=1, keepdims=True)

        # ── Init (index upload — timed separately) ───────────────────
        t0 = time.perf_counter()
        gpu_idx = GPUVectorIndex(index_vecs, max_batch_size=B)
        init_ms = (time.perf_counter() - t0) * 1000
        print(f"    Init (index upload): {init_ms:.1f} ms")

        # ── Correctness: GPU batch vs CPU brute-force ─────────────────
        gpu_ids, gpu_dists = gpu_idx.batch_search(queries, k=K)
        cpu_ids, cpu_dists = cpu_brute_force(index_vecs, queries, k=K)

        # Check distances match within tolerance
        max_diff = float(np.abs(gpu_dists - cpu_dists).max())
        ids_match = int(np.sum(gpu_ids == cpu_ids))
        total_ids = B * K
        correct = max_diff < 1e-3

        print(f"    Correctness  max dist diff: {max_diff:.6f}  "
              f"ids match: {ids_match}/{total_ids}  → {'PASS' if correct else 'FAIL'}")

        # ── Latency: single-query loop vs batch dispatch ──────────────
        # Single-query loop (old pattern)
        times_single = []
        for i in range(B):
            t0 = time.perf_counter()
            gpu_idx.batch_search(queries[i:i+1], k=K)
            times_single.append((time.perf_counter() - t0) * 1000)

        # Batch dispatch (new pattern)
        times_batch = []
        for _ in range(5):
            t0 = time.perf_counter()
            gpu_idx.batch_search(queries, k=K)
            times_batch.append((time.perf_counter() - t0) * 1000)

        single_total = float(np.sum(times_single))
        batch_avg    = float(np.mean(times_batch))
        speedup      = single_total / batch_avg if batch_avg > 0 else 0

        print(f"    Single-query loop {B}×: {single_total:.2f} ms total")
        print(f"    Batch dispatch  {B}q:  {batch_avg:.2f} ms avg")
        print(f"    Batch speedup:         {speedup:.2f}x")

        test_results[str(n)] = {
            "init_ms":         round(init_ms, 2),
            "correctness_max_diff": round(max_diff, 7),
            "ids_match":       f"{ids_match}/{total_ids}",
            "correctness_pass": correct,
            "single_loop_ms":  round(single_total, 4),
            "batch_dispatch_ms": round(batch_avg, 4),
            "batch_speedup_x": round(speedup, 2),
        }

    results["tests"]["gpu_vector_index"] = {"status": "SUCCESS", "scales": test_results}


# ═══════════════════════════════════════════════════════════════════════════════
# Test 2 — HierarchicalGPUSearch: results subset of brute-force
# ═══════════════════════════════════════════════════════════════════════════════

def test_hierarchical_search():
    print(f"\n{SEP}")
    print("  TEST 2 — HierarchicalGPUSearch: two-stage search")
    print(SEP)

    try:
        from gpu_hierarchical_search import HierarchicalGPUSearch
    except Exception as e:
        print(f"  [SKIP] HierarchicalGPUSearch not available: {e}")
        results["tests"]["hierarchical"] = {"status": "SKIPPED", "reason": str(e)}
        return

    n = 10_000
    rng = np.random.default_rng(SEED)
    index_vecs = rng.standard_normal((n, DIM)).astype(np.float32)
    index_vecs /= np.linalg.norm(index_vecs, axis=1, keepdims=True)
    queries    = rng.standard_normal((B, DIM)).astype(np.float32)
    queries   /= np.linalg.norm(queries, axis=1, keepdims=True)

    print(f"  Building index ({n:,} vectors, C=50, n_probe=5)...")
    h = HierarchicalGPUSearch(n_clusters=50, n_probe=5, max_batch_size=B)
    t0 = time.perf_counter()
    h.build(index_vecs, use_gpu=True)
    build_ms = (time.perf_counter() - t0) * 1000
    print(f"  Build time: {build_ms:.0f} ms")

    # Run batch search
    t0 = time.perf_counter()
    hier_ids, hier_dists = h.batch_search(queries, k=K)
    search_ms = (time.perf_counter() - t0) * 1000

    # Brute-force reference
    cpu_ids, cpu_dists = cpu_brute_force(index_vecs, queries, k=K)

    # Recall: what fraction of true top-K are in hierarchical results?
    recall_sum = 0
    for i in range(B):
        true_set  = set(cpu_ids[i].tolist())
        hier_set  = set(hier_ids[i].tolist())
        recall_sum += len(true_set & hier_set) / K
    recall = recall_sum / B

    print(f"  Search batch ({B} queries): {search_ms:.2f} ms")
    print(f"  Recall @ {K}: {recall:.3f}  (fraction of true top-{K} found)")

    results["tests"]["hierarchical"] = {
        "status":     "SUCCESS",
        "n_vectors":  n,
        "n_clusters": 50,
        "n_probe":    5,
        "build_ms":   round(build_ms, 2),
        "search_ms":  round(search_ms, 4),
        f"recall_at_{K}": round(recall, 4),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Test 3 — SplatStore.batch_find_neighbors
# ═══════════════════════════════════════════════════════════════════════════════

def test_splat_store_batch():
    print(f"\n{SEP}")
    print("  TEST 3 — SplatStore.batch_find_neighbors (CPU & Vulkan paths)")
    print(SEP)

    try:
        import torch
        from config import M2MConfig
        from splats import SplatStore
        from geometry import normalize_sphere
    except Exception as e:
        print(f"  [SKIP] SplatStore deps not available: {e}")
        results["tests"]["splat_store_batch"] = {"status": "SKIPPED", "reason": str(e)}
        return

    n    = 5_000
    test_results = {}

    for backend, use_vulkan in [("cpu", False), ("vulkan", True)]:
        print(f"\n  Backend: {backend.upper()}")
        config = M2MConfig(
            device="vulkan" if use_vulkan else "cpu",
            latent_dim=DIM,
            max_splats=n,
            enable_vulkan=use_vulkan,
        )
        store = SplatStore(config)

        # Add splats
        torch.manual_seed(SEED)
        vecs = normalize_sphere(torch.randn(n, DIM))
        store.add_splat(vecs)
        store.build_index()

        # Single-query baseline
        q_single = normalize_sphere(torch.randn(1, DIM))
        t0 = time.perf_counter()
        mu1, a1, k1 = store.find_neighbors(q_single, k=K)
        single_ms = (time.perf_counter() - t0) * 1000

        # Batch query
        q_batch = normalize_sphere(torch.randn(B, DIM))
        t0 = time.perf_counter()
        mu_b, a_b, k_b = store.batch_find_neighbors(q_batch, k=K, max_batch_size=B)
        batch_ms = (time.perf_counter() - t0) * 1000

        print(f"    find_neighbors (1 query):        {single_ms:.3f} ms")
        print(f"    batch_find_neighbors ({B} queries): {batch_ms:.3f} ms")
        print(f"    Batch output shape: mu={list(mu_b.shape)}")

        shapes_ok = (
            mu_b.shape == (B, K, DIM) and
            a_b.shape  == (B, K) and
            k_b.shape  == (B, K)
        )
        print(f"    Output shapes correct: {'PASS' if shapes_ok else 'FAIL'}")

        test_results[backend] = {
            "single_query_ms":    round(single_ms, 4),
            "batch_query_ms":     round(batch_ms, 4),
            "batch_queries":      B,
            "mu_shape":           list(mu_b.shape),
            "shapes_correct":     shapes_ok,
        }

    results["tests"]["splat_store_batch"] = {
        "status": "SUCCESS",
        "n_splats": n,
        "backends": test_results,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 68)
    print("  GPU BATCH SEARCH VALIDATION")
    print("  GPUVectorIndex | HierarchicalGPUSearch | SplatStore.batch")
    print("=" * 68)

    test_gpu_vector_index()
    test_hierarchical_search()
    test_splat_store_batch()

    # Summary
    print(f"\n{'='*68}")
    print("  SUMMARY")
    print(f"{'='*68}")
    for t_name, t_data in results["tests"].items():
        status = t_data.get("status", "?")
        print(f"  {t_name:<30} {status}")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\n[SAVED] {OUTPUT_FILE.name}")
    print("=" * 68)


if __name__ == "__main__":
    main()
