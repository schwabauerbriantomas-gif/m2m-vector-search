#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
M2M Real-Data Benchmark
========================
Runs a full latency / throughput / training benchmark on the
sklearn Handwritten Digits dataset (the same source used in
validate_data_lake.py and README.md), so results are always
honest and reproducible.

Outputs
-------
  benchmarks/results/benchmark_latest.json   — latest run (overwritten)
  benchmarks/results/benchmark_YYYYMMDD_HHMM.json  — timestamped archive

Usage
-----
  python benchmarks/run_benchmark.py                  # CPU only
  python benchmarks/run_benchmark.py --device vulkan  # Vulkan GPU only
  python benchmarks/run_benchmark.py --device both    # CPU then Vulkan
  python benchmarks/run_benchmark.py --n-queries 500 --k 10
"""

import sys
import time
import json
import platform
import argparse
import shutil
from pathlib import Path
from typing import Dict, Any

import numpy as np
import torch

# ── Project root ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from m2m import M2MConfig, create_m2m, normalize_sphere

# ── Results directory (relative → portable) ──────────────────────────────────
RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


# ═════════════════════════════════════════════════════════════════════════════
# Data loading
# ═════════════════════════════════════════════════════════════════════════════

HF_CACHE = RESULTS_DIR / "hf_embeddings_cache.npy"

def load_hf_embeddings(n_target: int = 10_000, latent_dim: int = 640, seed: int = 42):
    """
    Load real semantic embeddings from local Qdrant/dbpedia-entities-openai3 parquet files at C:\\dbpedia_dataset\\data.
    We truncate the 3072D OpenAI embeddings down to `latent_dim` (default 640).

    Vectors are L2-normalised before returning.
    Results are cached to benchmarks/results/hf_embeddings_cache_{n_target}_{latent_dim}.npy.
    """
    cache = RESULTS_DIR / f"hf_embeddings_cache_{n_target}_{latent_dim}.npy"
    ds_name = "Local DBpedia OpenAI (3072D)"

    if cache.exists():
        print(f"  [CACHE] Loading from {cache.name}...")
        X = np.load(str(cache))
        data = normalize_sphere(torch.tensor(X))
        return data, None, {
            "source": ds_name,
            "url": "file:///C:/dbpedia_dataset",
            "latent_dim": X.shape[1],
            "n_samples": X.shape[0],
            "normalization": "L2 unit sphere",
            "cached": True,
        }

    import pandas as pd
    from pathlib import Path

    print(f"  [LOCAL] Loading {ds_name} from C:\\dbpedia_dataset\\data ...")
    data_dir = Path("C:/dbpedia_dataset/data")
    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {data_dir}. Please place the DBpedia parquet files there.")

    parquet_files = sorted(data_dir.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No .parquet files found in {data_dir}")

    print(f"  [LOAD] Extracting vectors and truncating 3072D → {latent_dim}D ...")
    rows = []
    for p_file in parquet_files:
        print(f"    -> Reading {p_file.name}")
        df = pd.read_parquet(p_file, columns=["text-embedding-3-large-3072-embedding"])
        for vec in df["text-embedding-3-large-3072-embedding"]:
            rows.append(vec[:latent_dim])
            if len(rows) >= n_target:
                break
        if len(rows) >= n_target:
            break

    if len(rows) < n_target:
        print(f"  [WARN] Request n={n_target} but only {len(rows)} vectors available.")

    X = np.array(rows, dtype=np.float32)           # [N, latent_dim]
    np.save(str(cache), X)
    print(f"  [CACHE] Saved {cache.name}")

    data = normalize_sphere(torch.tensor(X))
    return data, None, {
        "source": ds_name,
        "url": "file:///C:/dbpedia_dataset",
        "latent_dim": X.shape[1],
        "n_samples": X.shape[0],
        "normalization": "L2 unit sphere",
        "cached": False,
    }


def load_sklearn_data(n_target: int = 10_000, latent_dim: int = 640, seed: int = 42):
    """Fallback: sklearn Handwritten Digits (toy dataset, 64D projected to latent_dim)."""
    from sklearn.datasets import load_digits
    digits = load_digits()
    X_raw = digits.data.astype(np.float32)
    rng = np.random.default_rng(seed)
    repeats = (n_target // len(X_raw)) + 1
    X_up = np.tile(X_raw, (repeats, 1))[:n_target]
    X_up += rng.normal(0, 0.01, X_up.shape).astype(np.float32)
    rng2 = np.random.default_rng(seed + 1)
    proj = rng2.standard_normal((64, latent_dim)).astype(np.float32)
    proj /= np.linalg.norm(proj, axis=1, keepdims=True) + 1e-8
    X_proj = X_up @ proj
    data = normalize_sphere(torch.tensor(X_proj))
    return data, digits.target[:n_target], {
        "source": "sklearn.datasets.load_digits (toy fallback)",
        "latent_dim": latent_dim,
        "n_samples": n_target,
        "normalization": "L2 unit sphere",
    }


# ═════════════════════════════════════════════════════════════════════════════
# Benchmark helpers
# ═════════════════════════════════════════════════════════════════════════════

def get_system_specs() -> Dict[str, Any]:
    specs: Dict[str, Any] = {
        "platform":           platform.platform(),
        "processor":          platform.processor() or "N/A",
        "python_version":     platform.python_version(),
        "torch_version":      torch.__version__,
        "numpy_version":      np.__version__,
    }
    try:
        import psutil
        specs["cpu_cores_physical"] = psutil.cpu_count(logical=False)
        specs["cpu_cores_logical"]  = psutil.cpu_count(logical=True)
        specs["ram_total_gb"]       = round(psutil.virtual_memory().total / 1024**3, 2)
    except ImportError:
        pass
    try:
        import vulkan as vk
        inst = vk.vkCreateInstance(vk.VkInstanceCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
            pApplicationInfo=vk.VkApplicationInfo(
                sType=vk.VK_STRUCTURE_TYPE_APPLICATION_INFO,
                apiVersion=vk.VK_MAKE_VERSION(1, 0, 0),
            ),
        ), None)
        devs = vk.vkEnumeratePhysicalDevices(inst)
        if devs:
            p = vk.vkGetPhysicalDeviceProperties(devs[0])
            specs["vulkan_gpu"] = p.deviceName
            specs["vulkan_api"] = (f"{vk.VK_VERSION_MAJOR(p.apiVersion)}."
                                   f"{vk.VK_VERSION_MINOR(p.apiVersion)}."
                                   f"{vk.VK_VERSION_PATCH(p.apiVersion)}")
        vk.vkDestroyInstance(inst, None)
    except Exception:
        specs["vulkan_gpu"] = "N/A"
    return specs


def linear_baseline(data: torch.Tensor, queries: torch.Tensor, k: int) -> Dict:
    """Brute-force O(N) baseline via torch.cdist."""
    latencies = []
    for i in range(len(queries)):
        q  = queries[i].unsqueeze(0)
        t0 = time.perf_counter()
        d  = torch.cdist(q, data, p=2)
        torch.topk(d.squeeze(0), k, largest=False)
        latencies.append((time.perf_counter() - t0) * 1000)

    lat = np.array(latencies)
    return {
        "avg_latency_ms": round(float(lat.mean()), 4),
        "p50_latency_ms": round(float(np.percentile(lat, 50)), 4),
        "p95_latency_ms": round(float(np.percentile(lat, 95)), 4),
        "p99_latency_ms": round(float(np.percentile(lat, 99)), 4),
        "throughput_qps": round(len(queries) / (lat.sum() / 1000), 2),
    }


def run_backend(device_name: str, data: torch.Tensor, queries: torch.Tensor,
                k: int, n_warmup: int = 5) -> Dict:
    """
    Full benchmark for one backend (cpu or vulkan).
    Returns a dict with latency, throughput, ingest, and training metrics.
    """
    n_splats  = data.shape[0]
    latent_dim = data.shape[1]
    is_vulkan = (device_name == "vulkan")

    print(f"\n  ── {device_name.upper()} ─────────────────────────────────────")

    config = M2MConfig(
        device=device_name,
        latent_dim=latent_dim,
        max_splats=n_splats + 1000,
        n_splats_init=n_splats,
        knn_k=k,
        enable_vulkan=is_vulkan,
    )

    # Init — use create_m2m (M2MSystem) which exposes add_splats, export_to_dataloader, search
    # Transformed data bypasses standard ingestion
    if device_name == "transformed":
        print(f"  Transforming dataset to M2M Format (Hierarchy Levels: 4)...")
        from dataset_transformer import M2MDatasetTransformer
        import os
        
        # Transform offline
        data_np = data.detach().cpu().numpy()
        transformer = M2MDatasetTransformer(data_np, n_clusters_base=200, hierarchy_levels=4)
        transform_start = time.perf_counter()
        result = transformer.transform()
        transformer.save_for_m2m('benchmark_temp_transformed.bin')
        add_s = time.perf_counter() - transform_start
        
        # Load directly
        init_s = time.perf_counter() - t0
        engine = create_m2m(config)
        
        # Override the loading mechanism
        engine.load_optimized('benchmark_temp_transformed.bin')
        print(f"  Transformed Loading: {len(result['splats']):,} splats  |  {add_s:.2f} s")
        ingest_qps = 0
        os.remove('benchmark_temp_transformed.bin')
        
    else:
        # Standard Init
        init_s = time.perf_counter() - t0
        engine = create_m2m(config)
        print(f"  Init: {init_s*1000:.1f} ms")

        # Ingest
        t0 = time.perf_counter()
        n_added = engine.add_splats(data)
        add_s = time.perf_counter() - t0
        ingest_qps = n_added / add_s if add_s > 0 else 0
        print(f"  Ingest: {n_added:,} splats  |  {ingest_qps:,.0f} splats/s")

    # Standard training throughput (export_to_dataloader + 1 epoch)
    try:
        std_dl = engine.export_to_dataloader(batch_size=256,
                                              importance_sampling=True,
                                              generate_samples=False)
        t0 = time.perf_counter()
        loss_acc, n_batches = 0.0, 0
        import torch.nn as nn
        crit = nn.CrossEntropyLoss()
        dummy_head = torch.nn.Linear(latent_dim, 10)
        for batch_mu in std_dl:
            out = dummy_head(batch_mu)
            tgt = torch.randint(0, 10, (batch_mu.shape[0],))
            loss_acc += crit(out, tgt).item(); n_batches += 1
        std_s = time.perf_counter() - t0
        std_tp = n_splats / std_s if std_s > 0 else 0
        std_loss = loss_acc / n_batches if n_batches > 0 else 0.0
        print(f"  Std training: {std_tp:,.0f} splats/s  loss={std_loss:.4f}")

        # Generative training throughput
        gen_dl = engine.export_to_dataloader(batch_size=256, generate_samples=True)
        t0 = time.perf_counter()
        gen_loss_acc, gen_batches = 0.0, 0
        for batch_mu in gen_dl:
            out = dummy_head(batch_mu)
            tgt = torch.randint(0, 10, (batch_mu.shape[0],))
            gen_loss_acc += crit(out, tgt).item(); gen_batches += 1
        gen_s = time.perf_counter() - t0
        gen_tp = n_splats / gen_s if gen_s > 0 else 0
        gen_loss = gen_loss_acc / gen_batches if gen_batches > 0 else 0.0
        print(f"  Gen training:  {gen_tp:,.0f} splats/s  loss={gen_loss:.4f}")
    except Exception as e:
        print(f"  [WARN] Training skipping for device {device_name}. Reason: {str(e)}")
        std_tp, std_s, std_loss = (0, 0, 0)
        gen_tp, gen_s, gen_loss = (0, 0, 0)
    for i in range(min(n_warmup, len(queries))):
        engine.search(queries[i].unsqueeze(0), k=k)

    latencies: list[float] = []
    for i in range(len(queries)):
        q  = queries[i].unsqueeze(0)
        t0 = time.perf_counter()
        engine.search(q, k=k)
        latencies.append((time.perf_counter() - t0) * 1000)

    lat = np.array(latencies)
    total_s = lat.sum() / 1000
    qps = len(queries) / total_s
    print(f"  Retrieval QPS: {qps:.2f}  avg={lat.mean():.2f}ms  p95={np.percentile(lat,95):.2f}ms")

    return {
        "device":               device_name,
        "vulkan_enabled":       is_vulkan,
        "init_time_ms":         round(init_s * 1000, 2),
        "n_splats":             n_splats,
        "ingest_throughput_qps": round(ingest_qps, 2),
        "standard_training": {
            "throughput_splats_per_sec": round(std_tp, 2),
            "epoch_time_s":              round(std_s, 4),
            "avg_loss":                  round(std_loss, 6),
        },
        "generative_training": {
            "throughput_splats_per_sec": round(gen_tp, 2),
            "epoch_time_s":              round(gen_s, 4),
            "avg_loss":                  round(gen_loss, 6),
        },
        "retrieval": {
            "n_queries":        len(queries),
            "k":                k,
            "avg_latency_ms":   round(float(lat.mean()), 4),
            "p50_latency_ms":   round(float(np.percentile(lat, 50)), 4),
            "p95_latency_ms":   round(float(np.percentile(lat, 95)), 4),
            "p99_latency_ms":   round(float(np.percentile(lat, 99)), 4),
            "min_latency_ms":   round(float(lat.min()), 4),
            "max_latency_ms":   round(float(lat.max()), 4),
            "throughput_qps":   round(qps, 2),
        },
    }


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="M2M Real-Data Benchmark — OpenAI Embeddings via Qdrant DBpedia (HuggingFace)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--device",   default="both", choices=["cpu", "vulkan", "transformed", "both", "all"])
    parser.add_argument("--dataset",  default="hf", choices=["hf", "sklearn"],
                        help="Dataset source (default: hf = HuggingFace Qdrant/DBpedia embeddings)")
    parser.add_argument("--n-splats", type=int, default=10_000)
    parser.add_argument("--n-queries",type=int, default=1_000)
    parser.add_argument("--k",        type=int, default=10)
    parser.add_argument("--dim",      type=int, default=640,
                        help="Latent dimension to use/truncate to (default 640)")
    parser.add_argument("--seed",     type=int, default=42)
    args = parser.parse_args()

    ts = time.strftime("%Y%m%d_%H%M")
    ds_label = f"HuggingFace Qdrant dbpedia ({args.dim}D)" if args.dataset == "hf" else f"sklearn digits ({args.dim}D)"
    print("=" * 70)
    print("  M2M Real-Data Benchmark")
    print(f"  Dataset: {ds_label}")
    print(f"  N={args.n_splats:,}  queries={args.n_queries}  K={args.k}")
    print("=" * 70)

    # System specs
    print("\n[SYSTEM] Collecting specs...")
    specs = get_system_specs()
    for k_spec, v in specs.items():
        print(f"  {k_spec}: {v}")

    # Load data
    print(f"\n[DATA] Loading {ds_label}  →  {args.n_splats:,} vectors...")
    if args.dataset == "hf":
        data, labels, data_meta = load_hf_embeddings(args.n_splats, args.dim, args.seed)
    else:
        data, labels, data_meta = load_sklearn_data(args.n_splats, args.dim, args.seed)
    rng = np.random.default_rng(args.seed + 99)
    q_idx = rng.choice(len(data), size=min(args.n_queries, len(data)), replace=False)
    queries = data[q_idx]
    print(f"  Data:    {data.shape}  (L2-normalised)")
    print(f"  Queries: {queries.shape}")

    # Linear baseline
    print(f"\n[BASELINE] Linear brute-force (torch.cdist)  K={args.k}...")
    baseline = linear_baseline(data, queries, args.k)
    print(f"  Avg: {baseline['avg_latency_ms']:.2f} ms  |  QPS: {baseline['throughput_qps']:.2f}")

    # Backend benchmarks
    device_map = {
        "all": ["cpu", "vulkan", "transformed"],
        "both": ["cpu", "vulkan", "transformed"],
        "cpu": ["cpu"],
        "vulkan": ["vulkan"],
        "transformed": ["transformed"]
    }
    devices = device_map[args.device]
    backend_results = {}
    for dev in devices:
        try:
            backend_results[dev] = run_backend(dev, data, queries, args.k)
        except Exception as e:
            print(f"  [ERROR] {dev}: {e}")
            backend_results[dev] = {"error": str(e)}

    # Speedup summary
    print("\n" + "=" * 70)
    print("  RESULTS SUMMARY")
    print("=" * 70)
    hdr = f"{'Metric':<28} | {'Linear':>12} | {'CPU':>12} | {'Vulkan GPU':>12}"
    print(hdr)
    print("-" * len(hdr))
    def _v(d, key):
        if d is None or "error" in d: return "--"
        return f"{d['retrieval'][key]:.2f}"
    cpu_r = backend_results.get("cpu")
    vk_r  = backend_results.get("vulkan")
    print(f"{'Avg Latency (ms)':<28} | {baseline['avg_latency_ms']:>12.2f} | {_v(cpu_r,'avg_latency_ms'):>12} | {_v(vk_r,'avg_latency_ms'):>12}")
    print(f"{'P95 Latency (ms)':<28} | {'—':>12} | {_v(cpu_r,'p95_latency_ms'):>12} | {_v(vk_r,'p95_latency_ms'):>12}")
    print(f"{'Throughput (QPS)':<28} | {baseline['throughput_qps']:>12.2f} | {_v(cpu_r,'throughput_qps'):>12} | {_v(vk_r,'throughput_qps'):>12}")
    if cpu_r and "error" not in cpu_r:
        sp = baseline['avg_latency_ms'] / cpu_r['retrieval']['avg_latency_ms']
        print(f"  CPU speedup vs linear: {sp:.1f}x")
    if vk_r and "error" not in vk_r:
        sp = baseline['avg_latency_ms'] / vk_r['retrieval']['avg_latency_ms']
        print(f"  Vulkan speedup vs linear: {sp:.1f}x")

    # Save
    output = {
        "timestamp":      time.strftime("%Y-%m-%d %H:%M:%S"),
        "system_specs":   specs,
        "data_meta":      data_meta,
        "config": {
            "n_splats":    args.n_splats,
            "n_queries":   args.n_queries,
            "k":           args.k,
            "latent_dim":  args.dim,
            "seed":        args.seed,
        },
        "linear_baseline": baseline,
        "backends":        backend_results,
    }

    latest = RESULTS_DIR / "benchmark_latest.json"
    archive = RESULTS_DIR / f"benchmark_{ts}.json"
    for p in [latest, archive]:
        with open(p, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2)

    print(f"\n[SAVED] {latest.name}")
    print(f"[SAVED] {archive.name}  (archive)")
    print("=" * 70)
    return output


if __name__ == "__main__":
    main()
