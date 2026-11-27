import numpy as np

from m2m import SimpleVectorDB
from m2m.lsh_index import CrossPolytopeLSH, LSHConfig


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
    config = LSHConfig(dim=dim, n_tables=20, n_bits=18, n_probes=60, n_candidates=600, seed=42)

    index = CrossPolytopeLSH(config)
    index.index(vectors)

    recall = index.get_recall(queries, ground_truth, k)

    print(f"Recall@{k}: {recall:.4f}")
    assert recall >= 0.95, f"Recall {recall} < 0.95"


def test_lsh_matches_exact_scan():
    """Paridad LSH vs scan exacto: test de CORRECCIÓN determinista, sin medir tiempos.

    Reemplaza a ``test_lsh_speedup``, que asertaba ``speedup >= 0.3`` con ``time.time``
    y fallaba bajo carga del host (dio -1.35x con el host ocupado). La medición de
    velocidad vive en ``benchmarks/bench_lsh.py``; acá se verifica que el camino LSH
    devuelve lo mismo que el camino exacto:

    * con queries idénticas a vectores indexados, el top-1 de LSH ES el vecino exacto
      (distancia 0) para el 100% de las queries;
    * las distancias que devuelve LSH son las distancias L2 exactas de los índices
      devueltos (el re-rankeo de candidatos usa la misma métrica que el scan exacto);
    * el resultado tiene ``k`` elementos ordenados por distancia no decreciente.
    """
    rng = np.random.default_rng(42)
    n_vectors, dim, k, n_queries = 2000, 640, 5, 50

    vectors = rng.standard_normal((n_vectors, dim)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)

    config = LSHConfig(dim=dim, n_tables=15, n_bits=18, n_probes=50, n_candidates=500, seed=42)
    index = CrossPolytopeLSH(config)
    index.index(vectors)

    query_idx = rng.choice(n_vectors, size=n_queries, replace=False)
    queries = vectors[query_idx].copy()

    for query, self_idx in zip(queries, query_idx):
        # Camino exacto: scan lineal con la misma métrica (L2) que usa el re-rankeo.
        exact_dist = np.linalg.norm(vectors - query, axis=1)
        exact_top1 = int(np.argmin(exact_dist))
        assert exact_top1 == self_idx  # sanidad: la copia exacta es el vecino único a distancia 0

        lsh_idx, lsh_dist = index.query(query, k=k)

        assert lsh_idx.shape == (k,) and lsh_dist.shape == (k,)
        assert int(lsh_idx[0]) == exact_top1, "top-1 de LSH != top-1 del scan exacto"
        np.testing.assert_allclose(lsh_dist, exact_dist[lsh_idx], rtol=1e-5, atol=1e-6)
        assert np.all(np.diff(lsh_dist) >= 0), "LSH no devolvió los resultados ordenados"


def test_m2m_integration_with_lsh():
    """Verifica integración con M2M SimpleVectorDB."""
    db = SimpleVectorDB(device="cpu", latent_dim=640, enable_lsh_fallback=True)

    np.random.seed(42)
    n_vectors = 2000
    dim = 640
    vectors = np.random.randn(n_vectors, dim).astype(np.float32)
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

    db.add(vectors)

    # Should use LSH because data is random/uniform
    assert db._use_lsh
    assert db.lsh is not None

    query = np.random.randn(dim).astype(np.float32)
    query = query / np.linalg.norm(query)

    k = 10
    # Retorno legacy SIEMPRE 4-tupla (mu, alpha, kappa, idx): decision del orquestador
    # para el audit r1-18 (antes 3 en LSH/vacio y 4 en el camino HRM2).
    result_vectors, _, _, _ = db.search(query, k=k)
    assert len(result_vectors) == k
    assert result_vectors.shape[1] == dim
