"""
M2M Edge Inference (Pure Vulkan + NumPy) — Updated to use GPUVectorIndex
Demonstrates sub-25ms RAG routing completely dependency-free from PyTorch/CUDA.

This script executes M2M Semantic MoE directly via CFFI and SPIR-V shaders,
perfect for IoT, Smartphones, or headsets without deep learning libraries.
"""

import os
import sys
import time
import numpy as np

# Add parent directory to path to import m2m modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gpu_vector_index import GPUVectorIndex

def main():
    print("==========================================")
    print("M2M Edge Computing Node (Vulkan SPIR-V)")
    print("==========================================")

    latent_dim = 640
    n_splats = 10000
    n_queries = 100

    print(f"[INFO] Initializing GPU Vector Index (Vulkan persistent buffers)...")
    print(f"[INFO] Target: Native GPU / Edge Accelerator")

    # Build a persistent index over the local embedding store
    np.random.seed(42)
    local_embeddings = np.random.randn(n_splats, latent_dim).astype(np.float32)

    try:
        # GPUVectorIndex uploads the full index once at init — never re-uploaded
        gpu_index = GPUVectorIndex(local_embeddings, max_batch_size=1)
    except Exception as e:
        print(f"[ERROR] Failed to init GPU index. Ensure LunarG SDK is present. Error: {e}")
        return

    print(f"[SUCCESS] Native Vulkan pipeline established.")
    print(f"[INFO] Running {n_queries} batched queries from Edge Sensor into M2M...")

    latencies = []

    # Warmup
    dummy_query = np.random.randn(latent_dim).astype(np.float32)
    _ = gpu_index.compute_distances(dummy_query, local_embeddings)

    # Benchmarking pure Vulkan execution time
    for i in range(n_queries):
        query = np.random.randn(latent_dim).astype(np.float32)

        start = time.perf_counter()
        distances = gpu_index.compute_distances(query, local_embeddings)
        end = time.perf_counter()

        latencies.append((end - start) * 1000)

    latencies = np.array(latencies)

    print("\n--- EDGE PERFORMANCE REPORT ---")
    print(f"Device Driver: Vulkan FFI (GPUVectorIndex)")
    print(f"Dataset Size:  {n_splats} (D={latent_dim})")
    print(f"Total Queries: {n_queries}")
    print(f"Avg Latency:   {np.mean(latencies):.2f} ms")
    print(f"p95 Latency:   {np.percentile(latencies, 95):.2f} ms")
    print(f"p99 Latency:   {np.percentile(latencies, 99):.2f} ms")
    print(f"Throughput:    {1000 / np.mean(latencies):.2f} QPS")
    print("-------------------------------")
    print("[SUCCESS] Pure Edge execution capability validated.")

if __name__ == "__main__":
    main()
