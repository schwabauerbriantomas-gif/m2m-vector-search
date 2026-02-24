#!/usr/bin/env python3
"""
M2M CPU vs Vulkan GPU Benchmark Runner
Runs benchmarks on both backends and outputs comparison results.
"""
import sys
import time
import json
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import torch
import numpy as np

from m2m import M2MConfig, M2MEngine, normalize_sphere

def run_benchmark(device_name, n_splats=10000, n_queries=100, k=64, latent_dim=128):
    """Run a single benchmark on a given device."""
    print(f"\n{'='*60}")
    print(f"  BENCHMARK: {device_name.upper()}")
    print(f"  Splats={n_splats}, Queries={n_queries}, K={k}, Dim={latent_dim}")
    print(f"{'='*60}\n")
    
    # Config
    config = M2MConfig(
        device=device_name,
        latent_dim=latent_dim,
        max_splats=n_splats + 1000,
        n_splats_init=n_splats,
        knn_k=k,
        enable_vulkan=(device_name == 'vulkan')
    )
    
    print(f"[CONFIG] device={config.device}, torch_device={config.torch_device}, vulkan={config.enable_vulkan}")
    
    # Initialize engine
    t0 = time.perf_counter()
    engine = M2MEngine(config)
    init_time = time.perf_counter() - t0
    print(f"[INIT] Engine initialized in {init_time:.3f}s")
    
    # Generate data
    torch.manual_seed(42)
    data = torch.randn(n_splats, latent_dim)
    data = normalize_sphere(data)
    
    # Add splats
    t0 = time.perf_counter()
    n_added = engine.add_splats(data)
    add_time = time.perf_counter() - t0
    print(f"[ADD] Added {n_added} splats in {add_time:.3f}s ({n_added/add_time:.0f} splats/s)")
    
    # Generate queries
    queries = torch.randn(n_queries, latent_dim)
    queries = normalize_sphere(queries)
    
    # Warmup
    for i in range(min(3, n_queries)):
        q = queries[i].unsqueeze(0)
        engine.search(q, k=k)
    
    # Benchmark search
    latencies = []
    for i in range(n_queries):
        q = queries[i].unsqueeze(0)
        t0 = time.perf_counter()
        result = engine.search(q, k=k)
        latency = (time.perf_counter() - t0) * 1000  # ms
        latencies.append(latency)
    
    latencies_np = np.array(latencies)
    
    results = {
        'device': device_name,
        'n_splats': n_splats,
        'n_queries': n_queries,
        'k': k,
        'latent_dim': latent_dim,
        'init_time_s': round(init_time, 4),
        'add_time_s': round(add_time, 4),
        'add_throughput_splats_per_s': round(n_added / add_time, 1),
        'search_avg_latency_ms': round(float(latencies_np.mean()), 4),
        'search_p50_latency_ms': round(float(np.percentile(latencies_np, 50)), 4),
        'search_p95_latency_ms': round(float(np.percentile(latencies_np, 95)), 4),
        'search_p99_latency_ms': round(float(np.percentile(latencies_np, 99)), 4),
        'search_min_latency_ms': round(float(latencies_np.min()), 4),
        'search_max_latency_ms': round(float(latencies_np.max()), 4),
        'search_throughput_qps': round(n_queries / (latencies_np.sum() / 1000), 2),
        'vulkan_enabled': config.enable_vulkan,
    }
    
    print(f"\n[RESULTS] {device_name.upper()}")
    print(f"  Avg latency:  {results['search_avg_latency_ms']:.4f} ms")
    print(f"  P50 latency:  {results['search_p50_latency_ms']:.4f} ms")
    print(f"  P95 latency:  {results['search_p95_latency_ms']:.4f} ms")
    print(f"  P99 latency:  {results['search_p99_latency_ms']:.4f} ms")
    print(f"  Throughput:   {results['search_throughput_qps']:.2f} QPS")
    print(f"  Add rate:     {results['add_throughput_splats_per_s']:.0f} splats/s")
    
    return results


def linear_search_baseline(n_splats=10000, n_queries=100, k=1, latent_dim=128):
    """Run a linear brute-force search as baseline."""
    print(f"\n{'='*60}")
    print(f"  BASELINE: LINEAR SEARCH (Brute Force)")
    print(f"  Splats={n_splats}, Queries={n_queries}, Dim={latent_dim}")
    print(f"{'='*60}\n")
    
    torch.manual_seed(42)
    splats = torch.randn(n_splats, latent_dim)
    splats = splats / (torch.norm(splats, dim=-1, keepdim=True) + 1e-8)
    
    queries = torch.randn(n_queries, latent_dim)
    queries = queries / (torch.norm(queries, dim=-1, keepdim=True) + 1e-8)
    
    latencies = []
    for i in range(n_queries):
        t0 = time.perf_counter()
        dists = torch.cdist(queries[i:i+1], splats, p=2)
        topk = torch.topk(dists.squeeze(0), k, largest=False)
        latency = (time.perf_counter() - t0) * 1000
        latencies.append(latency)
    
    latencies_np = np.array(latencies)
    
    results = {
        'device': 'cpu_linear',
        'n_splats': n_splats,
        'n_queries': n_queries,
        'search_avg_latency_ms': round(float(latencies_np.mean()), 4),
        'search_p50_latency_ms': round(float(np.percentile(latencies_np, 50)), 4),
        'search_throughput_qps': round(n_queries / (latencies_np.sum() / 1000), 2),
    }
    
    print(f"[RESULTS] LINEAR BASELINE")
    print(f"  Avg latency:  {results['search_avg_latency_ms']:.4f} ms")
    print(f"  Throughput:   {results['search_throughput_qps']:.2f} QPS")
    
    return results


def main():
    N_SPLATS = 10000
    N_QUERIES = 100
    K = 64
    LATENT_DIM = 128
    
    print("="*60)
    print("  M2M CPU vs VULKAN GPU BENCHMARK")
    print("="*60)
    
    all_results = {}
    
    # 1. Linear baseline
    baseline = linear_search_baseline(N_SPLATS, N_QUERIES, k=1, latent_dim=LATENT_DIM)
    all_results['linear_baseline'] = baseline
    
    # 2. CPU benchmark
    cpu_results = run_benchmark('cpu', N_SPLATS, N_QUERIES, K, LATENT_DIM)
    all_results['cpu'] = cpu_results
    
    # 3. Vulkan GPU benchmark
    vulkan_results = run_benchmark('vulkan', N_SPLATS, N_QUERIES, K, LATENT_DIM)
    all_results['vulkan'] = vulkan_results
    
    # Comparison
    print(f"\n{'='*60}")
    print("  COMPARISON RESULTS")
    print(f"{'='*60}\n")
    
    speedup_vs_linear_cpu = baseline['search_avg_latency_ms'] / cpu_results['search_avg_latency_ms'] if cpu_results['search_avg_latency_ms'] > 0 else 0
    speedup_vs_linear_vulkan = baseline['search_avg_latency_ms'] / vulkan_results['search_avg_latency_ms'] if vulkan_results['search_avg_latency_ms'] > 0 else 0
    vulkan_vs_cpu = cpu_results['search_avg_latency_ms'] / vulkan_results['search_avg_latency_ms'] if vulkan_results['search_avg_latency_ms'] > 0 else 0
    
    print(f"{'Metric':<30} | {'Linear':<15} | {'CPU':<15} | {'Vulkan GPU':<15}")
    print("-"*80)
    print(f"{'Avg Latency (ms)':<30} | {baseline['search_avg_latency_ms']:<15.4f} | {cpu_results['search_avg_latency_ms']:<15.4f} | {vulkan_results['search_avg_latency_ms']:<15.4f}")
    print(f"{'P50 Latency (ms)':<30} | {baseline['search_p50_latency_ms']:<15.4f} | {cpu_results['search_p50_latency_ms']:<15.4f} | {vulkan_results['search_p50_latency_ms']:<15.4f}")
    print(f"{'Throughput (QPS)':<30} | {baseline['search_throughput_qps']:<15.2f} | {cpu_results['search_throughput_qps']:<15.2f} | {vulkan_results['search_throughput_qps']:<15.2f}")
    print(f"{'P95 Latency (ms)':<30} | {'N/A':<15} | {cpu_results['search_p95_latency_ms']:<15.4f} | {vulkan_results['search_p95_latency_ms']:<15.4f}")
    print(f"{'Speedup vs Linear':<30} | {'1.0x':<15} | {speedup_vs_linear_cpu:<15.1f}x | {speedup_vs_linear_vulkan:<15.1f}x")
    print(f"{'Vulkan GPU vs CPU':<30} | {'---':<15} | {'---':<15} | {vulkan_vs_cpu:<15.1f}x")
    print("-"*80)
    
    all_results['comparison'] = {
        'speedup_cpu_vs_linear': round(speedup_vs_linear_cpu, 2),
        'speedup_vulkan_vs_linear': round(speedup_vs_linear_vulkan, 2),
        'speedup_vulkan_vs_cpu': round(vulkan_vs_cpu, 2),
    }
    
    # Save
    output_path = project_root / 'benchmark_cpu_vs_vulkan.json'
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")
    print("="*60)


if __name__ == '__main__':
    main()
