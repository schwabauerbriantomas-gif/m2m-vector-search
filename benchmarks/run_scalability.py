#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
M2M Scalability Benchmark — 10k / 50k / 100k Vectors
=====================================================

Tests how M2M scales across dataset sizes with CPU-only and CPU+Vulkan backends.
Same methodology at every scale: real latency/throughput metrics, brute-force baseline,
deterministic seed, system specs recorded.

Output: scalability_results.json
"""

import sys
import os
import time
import json
import platform
import psutil
from pathlib import Path

import torch
import numpy as np

# ── Project imports ──────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from m2m import M2MConfig, M2MEngine, normalize_sphere

# ── Constants ────────────────────────────────────────────────────────────────
SCALES       = [10_000, 50_000, 100_000]
N_QUERIES    = 100
K            = 64
LATENT_DIM   = 128
SEED         = 42
WARMUP       = 3          # warmup queries (discarded)

RESULTS_DIR  = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)
OUTPUT_FILE  = RESULTS_DIR / "scalability_latest.json"


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def get_system_specs() -> dict:
    """Collect reproducible system specs."""
    specs = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_name": platform.processor() or "N/A",
        "cpu_cores_physical": psutil.cpu_count(logical=False),
        "cpu_cores_logical": psutil.cpu_count(logical=True),
        "ram_total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
    }

    # Try to get Vulkan GPU name
    try:
        import vulkan as vk
        instance = vk.vkCreateInstance(
            vk.VkInstanceCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
                pApplicationInfo=vk.VkApplicationInfo(
                    sType=vk.VK_STRUCTURE_TYPE_APPLICATION_INFO,
                    apiVersion=vk.VK_MAKE_VERSION(1, 0, 0),
                ),
            ),
            None,
        )
        devices = vk.vkEnumeratePhysicalDevices(instance)
        if devices:
            props = vk.vkGetPhysicalDeviceProperties(devices[0])
            specs["vulkan_gpu"] = props.deviceName
            specs["vulkan_api_version"] = f"{vk.VK_VERSION_MAJOR(props.apiVersion)}.{vk.VK_VERSION_MINOR(props.apiVersion)}.{vk.VK_VERSION_PATCH(props.apiVersion)}"
        vk.vkDestroyInstance(instance, None)
    except Exception:
        specs["vulkan_gpu"] = "N/A (vulkan not available)"

    return specs


def generate_data(n_vectors: int, dim: int, seed: int):
    """Generate deterministic, L2-normalised test vectors and queries."""
    torch.manual_seed(seed)
    data    = normalize_sphere(torch.randn(n_vectors, dim))
    queries = normalize_sphere(torch.randn(N_QUERIES, dim))
    return data, queries


def linear_baseline(data: torch.Tensor, queries: torch.Tensor, k: int) -> dict:
    """Brute-force linear search baseline via torch.cdist."""
    latencies = []
    for i in range(len(queries)):
        q = queries[i].unsqueeze(0)
        t0 = time.perf_counter()
        dists = torch.cdist(q, data, p=2)
        _ = torch.topk(dists.squeeze(0), k, largest=False)
        latencies.append((time.perf_counter() - t0) * 1000)

    lat = np.array(latencies)
    total_s = lat.sum() / 1000
    return {
        "avg_latency_ms":  round(float(lat.mean()), 4),
        "p50_latency_ms":  round(float(np.percentile(lat, 50)), 4),
        "p95_latency_ms":  round(float(np.percentile(lat, 95)), 4),
        "p99_latency_ms":  round(float(np.percentile(lat, 99)), 4),
        "min_latency_ms":  round(float(lat.min()), 4),
        "max_latency_ms":  round(float(lat.max()), 4),
        "throughput_qps":  round(len(queries) / total_s, 2),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Core benchmark runner
# ═══════════════════════════════════════════════════════════════════════════════

def run_engine_benchmark(device_name: str, data: torch.Tensor, queries: torch.Tensor,
                         k: int, scale: int) -> dict:
    """Benchmark M2MEngine on a given backend for a given dataset."""
    is_vulkan = (device_name == "vulkan")

    config = M2MConfig(
        device=device_name,
        latent_dim=LATENT_DIM,
        max_splats=scale + 1000,
        n_splats_init=scale,
        knn_k=k,
        enable_vulkan=is_vulkan,
    )

    # Init
    t0 = time.perf_counter()
    engine = M2MEngine(config)
    init_time = time.perf_counter() - t0

    # Add splats
    t0 = time.perf_counter()
    n_added = engine.add_splats(data)
    add_time = time.perf_counter() - t0
    add_throughput = n_added / add_time if add_time > 0 else 0

    # Warmup
    for i in range(min(WARMUP, len(queries))):
        engine.search(queries[i].unsqueeze(0), k=k)

    # Timed search
    latencies = []
    for i in range(len(queries)):
        q = queries[i].unsqueeze(0)
        t0 = time.perf_counter()
        engine.search(q, k=k)
        latencies.append((time.perf_counter() - t0) * 1000)

    lat = np.array(latencies)
    total_s = lat.sum() / 1000

    # Memory estimate (splat tensor only)
    mem_mb = (scale * LATENT_DIM * 4) / (1024**2)  # float32

    return {
        "device":                  device_name,
        "vulkan_enabled":          is_vulkan,
        "n_splats":                scale,
        "n_added":                 n_added,
        "init_time_s":             round(init_time, 4),
        "add_time_s":              round(add_time, 4),
        "add_throughput_splats_s": round(add_throughput, 1),
        "search_avg_latency_ms":   round(float(lat.mean()), 4),
        "search_p50_latency_ms":   round(float(np.percentile(lat, 50)), 4),
        "search_p95_latency_ms":   round(float(np.percentile(lat, 95)), 4),
        "search_p99_latency_ms":   round(float(np.percentile(lat, 99)), 4),
        "search_min_latency_ms":   round(float(lat.min()), 4),
        "search_max_latency_ms":   round(float(lat.max()), 4),
        "search_throughput_qps":   round(len(queries) / total_s, 2),
        "memory_estimate_mb":      round(mem_mb, 2),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 72)
    print("  M2M SCALABILITY BENCHMARK  —  10k / 50k / 100k")
    print("=" * 72)

    # ── System specs ─────────────────────────────────────────────────────
    print("\n[SYSTEM] Collecting specs...")
    specs = get_system_specs()
    for k_spec, v_spec in specs.items():
        print(f"  {k_spec}: {v_spec}")

    # ── Methodology ──────────────────────────────────────────────────────
    methodology = {
        "approach":       "Deterministic L2-normalised vectors, same seed across all scales",
        "seed":           SEED,
        "latent_dim":     LATENT_DIM,
        "n_queries":      N_QUERIES,
        "k":              K,
        "warmup_queries": WARMUP,
        "reproducible":   True,
    }

    dataset_info = {
        "type":          "Synthetic (torch.randn), deterministic seed",
        "source":        "torch.manual_seed(42)",
        "distribution":  "Standard Normal, L2-normalised to unit sphere",
        "dimensions":    LATENT_DIM,
        "dtype":         "float32",
        "normalization": "L2 (unit sphere S^{dim-1})",
        "scales_tested": SCALES,
    }

    # ── Run benchmarks ───────────────────────────────────────────────────
    all_results = {}

    for scale in SCALES:
        print(f"\n{'─'*72}")
        print(f"  SCALE: {scale:,} vectors  (dim={LATENT_DIM})")
        print(f"{'─'*72}")

        data, queries = generate_data(scale, LATENT_DIM, SEED)
        scale_key = str(scale)
        all_results[scale_key] = {}

        # 1 — Linear baseline
        print(f"\n  [BASELINE] Linear brute-force (torch.cdist)...")
        baseline = linear_baseline(data, queries, K)
        all_results[scale_key]["linear_baseline"] = baseline
        print(f"    Avg latency: {baseline['avg_latency_ms']:.4f} ms  |  QPS: {baseline['throughput_qps']:.2f}")

        # 2 — CPU-only
        print(f"\n  [CPU] Running M2MEngine (CPU-only)...")
        try:
            cpu_res = run_engine_benchmark("cpu", data, queries, K, scale)
            all_results[scale_key]["cpu"] = cpu_res
            cpu_speedup = baseline["avg_latency_ms"] / cpu_res["search_avg_latency_ms"] if cpu_res["search_avg_latency_ms"] > 0 else 0
            cpu_res["speedup_vs_linear"] = round(cpu_speedup, 2)
            print(f"    Avg latency: {cpu_res['search_avg_latency_ms']:.4f} ms  |  QPS: {cpu_res['search_throughput_qps']:.2f}  |  Speedup: {cpu_speedup:.1f}x")
        except Exception as e:
            all_results[scale_key]["cpu"] = {"error": str(e)}
            print(f"    [ERROR] {e}")

        # 3 — CPU + Vulkan
        print(f"\n  [VULKAN] Running M2MEngine (CPU+Vulkan GPU)...")
        try:
            vulkan_res = run_engine_benchmark("vulkan", data, queries, K, scale)
            all_results[scale_key]["cpu_vulkan"] = vulkan_res
            vk_speedup = baseline["avg_latency_ms"] / vulkan_res["search_avg_latency_ms"] if vulkan_res["search_avg_latency_ms"] > 0 else 0
            vulkan_res["speedup_vs_linear"] = round(vk_speedup, 2)
            vk_vs_cpu = cpu_res["search_avg_latency_ms"] / vulkan_res["search_avg_latency_ms"] if vulkan_res["search_avg_latency_ms"] > 0 else 0
            vulkan_res["speedup_vs_cpu"] = round(vk_vs_cpu, 2)
            print(f"    Avg latency: {vulkan_res['search_avg_latency_ms']:.4f} ms  |  QPS: {vulkan_res['search_throughput_qps']:.2f}  |  Speedup vs linear: {vk_speedup:.1f}x  |  vs CPU: {vk_vs_cpu:.1f}x")
        except Exception as e:
            all_results[scale_key]["cpu_vulkan"] = {"error": str(e)}
            print(f"    [ERROR] {e}")

    # ── Scaling analysis ────────────────────────────────────────────────
    print(f"\n{'='*72}")
    print("  SCALING ANALYSIS")
    print(f"{'='*72}\n")

    scaling = {"scales": SCALES, "cpu": {}, "cpu_vulkan": {}, "linear": {}}

    for backend in ["cpu", "cpu_vulkan", "linear_baseline"]:
        label = backend if backend != "linear_baseline" else "linear"
        qps_list = []
        lat_list = []
        for s in SCALES:
            entry = all_results.get(str(s), {}).get(backend, {})
            qps = entry.get("search_throughput_qps") or entry.get("throughput_qps", 0)
            lat = entry.get("search_avg_latency_ms") or entry.get("avg_latency_ms", 0)
            qps_list.append(qps)
            lat_list.append(lat)
        scaling[label]["throughput_qps"] = qps_list
        scaling[label]["avg_latency_ms"] = lat_list

    # Print comparison table
    header = f"{'Scale':>10} | {'Linear QPS':>12} | {'CPU QPS':>12} | {'Vulkan QPS':>12} | {'CPU Speedup':>12} | {'Vulkan Speedup':>14}"
    print(header)
    print("-" * len(header))
    for i, s in enumerate(SCALES):
        lin_qps = scaling["linear"]["throughput_qps"][i]
        cpu_qps = scaling["cpu"]["throughput_qps"][i]
        vk_qps  = scaling["cpu_vulkan"]["throughput_qps"][i]
        cpu_su  = cpu_qps / lin_qps if lin_qps > 0 else 0
        vk_su   = vk_qps / lin_qps if lin_qps > 0 else 0
        print(f"{s:>10,} | {lin_qps:>12.2f} | {cpu_qps:>12.2f} | {vk_qps:>12.2f} | {cpu_su:>11.1f}x | {vk_su:>13.1f}x")
    print()

    # ── Save ─────────────────────────────────────────────────────────────
    output = {
        "timestamp":      time.strftime("%Y-%m-%d %H:%M:%S"),
        "system_specs":   specs,
        "methodology":    methodology,
        "dataset_info":   dataset_info,
        "results":        all_results,
        "scaling_analysis": scaling,
    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    # Also save timestamped archive
    ts_file = RESULTS_DIR / f"scalability_{time.strftime('%Y%m%d_%H%M')}.json"
    with open(ts_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"[SAVED] {OUTPUT_FILE.name}")
    print(f"[SAVED] {ts_file.name}  (archive)")
    print("=" * 72)
    print("  BENCHMARK COMPLETE")
    print("=" * 72)


if __name__ == "__main__":
    main()
