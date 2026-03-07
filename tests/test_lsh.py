import numpy as np
import time
import pytest
from src.m2m.lsh_index import CrossPolytopeLSH, LSHConfig
from src.m2m.__init__ import SimpleVectorDB

def test_lsh_recall_uniform_data():
    """Verifica que LSH logra >95% recall en datos uniformes."""
    # Generar datos uniformes en S^639
    np.random.seed(42)
    n_vectors = 5000
    dim = 640

    vectors = np.random.randn(n_vectors, dim).astype(np.float32)
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

    # Generar queries (perturbaciones de vectores existentes para asegurar vecindad)
    n_queries = 50
    base_indices = np.random.choice(n_vectors, n_queries, replace=False)
    noise = np.random.randn(n_queries, dim).astype(np.float32) * 0.001
    queries = vectors[base_indices] + noise
    queries = queries / np.linalg.norm(queries, axis=1, keepdims=True)

    # Ground truth (linear scan)
    k = 1
    ground_truth = np.zeros((n_queries, k), dtype=int)
    for i, q in enumerate(queries):
        distances = np.linalg.norm(vectors - q, axis=1)
        ground_truth[i] = np.argsort(distances)[:k]

    # Test LSH
    config = LSHConfig(
        dim=dim,
        n_tables=20,
        n_bits=18,
        n_probes=60,
        n_candidates=600,
        seed=42
    )

    index = CrossPolytopeLSH(config)
    index.index(vectors)

    recall = index.get_recall(queries, ground_truth, k)

    print(f"Recall@{k}: {recall:.4f}")
    assert recall >= 0.95, f"Recall {recall} < 0.95"

def test_lsh_speedup():
    """Verifica que LSH es más rápido que linear scan."""
    np.random.seed(42)
    n_vectors = 20000
    dim = 640
    k = 10

    vectors = np.random.randn(n_vectors, dim).astype(np.float32)
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

    n_queries = 50
    base_indices = np.random.choice(n_vectors, n_queries, replace=False)
    noise = np.random.randn(n_queries, dim).astype(np.float32) * 0.05
    queries = vectors[base_indices] + noise
    queries = queries / np.linalg.norm(queries, axis=1, keepdims=True)

    config = LSHConfig(dim=dim, n_tables=15, n_bits=18, n_probes=50, n_candidates=500)
    index = CrossPolytopeLSH(config)
    index.index(vectors)

    # Medir linear scan
    start = time.time()
    for q in queries:
        distances = np.linalg.norm(vectors - q, axis=1)
        _ = np.argsort(distances)[:k]
    linear_time = (time.time() - start) / n_queries * 1000

    # Medir LSH
    start = time.time()
    for q in queries:
        index.query(q, k=k)
    lsh_time = (time.time() - start) / n_queries * 1000

    speedup = linear_time / max(lsh_time, 1e-10) # avoid division by zero
    print(f"Linear Scan: {linear_time:.2f}ms")
    print(f"LSH: {lsh_time:.2f}ms")
    print(f"Speedup: {speedup:.2f}x")

    assert speedup >= 1.5, f"Speedup {speedup} < 1.5x"

def test_m2m_integration_with_lsh():
    """Verifica integración con M2M SimpleVectorDB."""
    db = SimpleVectorDB(device='cpu', latent_dim=640, enable_lsh_fallback=True)
    
    np.random.seed(42)
    n_vectors = 2000
    dim = 640
    vectors = np.random.randn(n_vectors, dim).astype(np.float32)
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

    db.add(vectors)

    # Should use LSH because data is random/uniform
    assert db._use_lsh == True
    assert db.lsh is not None

    query = np.random.randn(dim).astype(np.float32)
    query = query / np.linalg.norm(query)

    k = 10
    result_vectors, _, _ = db.search(query, k=k)
    assert len(result_vectors) == k
    assert result_vectors.shape[1] == dim
