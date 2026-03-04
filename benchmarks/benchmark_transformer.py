"""Benchmark comparando datos originales vs transformados."""

import numpy as np
import time
import argparse
from dataset_transformer import M2MDatasetTransformer
from m2m import M2MEngine, M2MConfig, normalize_sphere


def benchmark_transformed(vectors: np.ndarray, queries: np.ndarray, k: int = 10):
    """
    Compara rendimiento antes y después de transformar.
    
    Args:
        vectors: Embeddings originales [N, D]
        queries: Queries de prueba [Q, D]
        k: Top-k resultados
    """
    print("=" * 60)
    print("BENCHMARK: DATOS ORIGINALES vs TRANSFORMADOS")
    print("=" * 60)
    
    # 1. Linear Scan (baseline)
    print("\n[1] Linear Scan...")
    start = time.perf_counter()
    for q in queries:
        distances = np.linalg.norm(vectors - q, axis=1)
        _ = np.argsort(distances)[:k]
    linear_time = (time.perf_counter() - start) / len(queries) * 1000
    linear_qps = len(queries) / (len(queries) * linear_time / 1000)
    
    # Normalize inputs for M2M which works on sphere S^d-1
    # Actually the transformer already clusters them. Let's provide them to the transformer and index.
    import torch
    
    # 2. M2M con datos originales
    print("[2] M2M Original...")
    config_orig = M2MConfig(device='cpu', latent_dim=vectors.shape[1], max_splats=len(vectors)+1000)
    index_orig = M2MEngine(config=config_orig)
    index_orig.add_splats(torch.tensor(vectors))
    
    start = time.perf_counter()
    for q in queries:
        _ = index_orig.search(torch.tensor(q).unsqueeze(0), k=k)
    m2m_orig_time = (time.perf_counter() - start) / len(queries) * 1000
    m2m_orig_qps = len(queries) / (len(queries) * m2m_orig_time / 1000)
    
    # 3. Transformar datos
    print("[3] Transformando dataset...")
    # Normalize before transforming to match what M2M assumes
    vectors_norm = normalize_sphere(torch.tensor(vectors)).numpy()
    transformer = M2MDatasetTransformer(vectors_norm, n_clusters_base=200, hierarchy_levels=4)
    result = transformer.transform()
    transformer.save_for_m2m('m2m_transformed.bin')
    
    # 4. M2M con datos transformados
    print("[4] M2M Transformado...")
    config_trans = M2MConfig(device='cpu', latent_dim=vectors.shape[1], max_splats=len(vectors)+1000)
    index_trans = M2MEngine(config=config_trans)
    index_trans.load_optimized('m2m_transformed.bin')
    
    start = time.perf_counter()
    for q in queries:
        _ = index_trans.search(torch.tensor(q).unsqueeze(0), k=k)
    m2m_trans_time = (time.perf_counter() - start) / len(queries) * 1000
    m2m_trans_qps = len(queries) / (len(queries) * m2m_trans_time / 1000)
    
    # Resultados
    print("\n" + "=" * 60)
    print("RESULTADOS")
    print("=" * 60)
    print(f"{'Método':<25} {'Latencia':>12} {'QPS':>12}")
    print("-" * 60)
    print(f"{'Linear Scan':<25} {linear_time:>10.2f}ms {linear_qps:>12.1f}")
    print(f"{'M2M Original':<25} {m2m_orig_time:>10.2f}ms {m2m_orig_qps:>12.1f}")
    print(f"{'M2M Transformado':<25} {m2m_trans_time:>10.2f}ms {m2m_trans_qps:>12.1f}")
    print("-" * 60)
    
    speedup = m2m_trans_qps / m2m_orig_qps if m2m_orig_qps > 0 else 0
    if m2m_trans_qps > m2m_orig_qps:
        print(f"✅ M2M Transformado es {speedup:.1f}x más rápido")
    else:
        print(f"⚠️  M2M Transformado es {1/speedup:.1f}x más lento")
    
    return {
        'linear_qps': linear_qps,
        'm2m_original_qps': m2m_orig_qps,
        'm2m_transformed_qps': m2m_trans_qps,
        'compression_ratio': result['stats']['compression_ratio'],
        'memory_savings': result['stats']['memory_savings_pct']
    }

def main():
    # Attempt to load data like in run_benchmark.py
    import sys
    from pathlib import Path
    
    # Add project root to sys path
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(PROJECT_ROOT))
    
    from benchmarks.run_benchmark import load_hf_embeddings

    data, _, _ = load_hf_embeddings(n_target=10000, latent_dim=640)
    data = data.numpy()
    
    # Generate random queries from data
    rng = np.random.default_rng(42)
    q_idx = rng.choice(len(data), size=1000, replace=False)
    queries = data[q_idx]

    benchmark_transformed(data, queries, k=10)

if __name__ == "__main__":
    main()
