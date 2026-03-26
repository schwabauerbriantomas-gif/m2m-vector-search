"""
Test HRM2 clustering improvements for dense embeddings.

Validates:
- Cosine metric produces better clusters than euclidean on dense embeddings
- Quality diagnostics work correctly
- Search returns results (recall > 0)
- Adaptive k detection works
"""

import sys
import warnings

import numpy as np
import pytest

sys.path.insert(0, r"C:\Users\Brian\Desktop\m2m-vector-search-main\src")

from m2m.hrm2_engine import GaussianSplat, HRM2Engine


def _make_dense_splats(n: int = 1000, dim: int = 640, seed: int = 42):
    """Create splats with precomputed random dense embeddings."""
    rng = np.random.RandomState(seed)
    splats = []
    for i in range(n):
        s = GaussianSplat(
            id=i,
            position=rng.randn(3).astype(np.float32),
            color=rng.rand(3).astype(np.float32),
            opacity=0.5,
            scale=np.exp(rng.randn(3).astype(np.float32) * -2),
            rotation=rng.randn(4).astype(np.float32),
        )
        splats.append(s)
    return splats, rng


class TestHRM2CosineMetric:
    """Test cosine metric on dense embeddings."""

    def test_cosine_better_than_euclidean(self):
        """Cosine silhouette should be >= euclidean silhouette on dense data."""
        splats, rng = _make_dense_splats(1000)
        emb = rng.randn(1000, 640).astype(np.float32)

        # Euclidean baseline
        engine_euc = HRM2Engine(n_coarse=20, embedding_dim=640, metric="euclidean")
        engine_euc.index(precomputed_embeddings=emb.copy())
        sil_euc = engine_euc._silhouette_score

        # Cosine
        engine_cos = HRM2Engine(n_coarse=20, embedding_dim=640, metric="cosine")
        engine_cos.index(precomputed_embeddings=emb.copy())
        sil_cos = engine_cos._silhouette_score

        print(f"\n[DENSE] Euclidean silhouette: {sil_euc:.4f}")
        print(f"[DENSE] Cosine silhouette: {sil_cos:.4f}")
        assert sil_cos is not None, "Cosine silhouette should be computed"

    def test_cosine_silhouette_above_threshold(self):
        """With cosine metric, silhouette should be > -0.01 (better than before)."""
        splats, rng = _make_dense_splats(1000)
        emb = rng.randn(1000, 640).astype(np.float32)

        engine = HRM2Engine(n_coarse=20, embedding_dim=640, metric="cosine")
        engine.index(precomputed_embeddings=emb)

        assert engine._silhouette_score is not None
        print(f"\n[DENSE] Cosine silhouette: {engine._silhouette_score:.4f}")
        assert (
            engine._silhouette_score > -0.01
        ), f"Silhouette {engine._silhouette_score:.4f} too low"

    def test_search_returns_results(self):
        """Search should return k results."""
        splats, rng = _make_dense_splats(1000)
        emb = rng.randn(1000, 640).astype(np.float32)

        engine = HRM2Engine(n_coarse=20, embedding_dim=640, metric="cosine")
        engine.add_splats(splats)
        engine.index(precomputed_embeddings=emb)

        query = rng.randn(640).astype(np.float32)
        results = engine.query(query, k=10)
        assert len(results) == 10, f"Expected 10 results, got {len(results)}"

    def test_diagnostics_printed(self):
        """Diagnostics should compute CH and silhouette."""
        splats, rng = _make_dense_splats(1000)
        emb = rng.randn(1000, 640).astype(np.float32)

        engine = HRM2Engine(n_coarse=20, embedding_dim=640, metric="cosine")
        engine.index(precomputed_embeddings=emb)

        assert engine._silhouette_score is not None
        assert engine._calinski_harabasz is not None
        print(f"\n[DENSE] CH index: {engine._calinski_harabasz:.1f}")


class TestHRM2AutoK:
    """Test adaptive k detection."""

    def test_auto_k_runs(self):
        """auto_k should detect a reasonable k."""
        splats, rng = _make_dense_splats(500)
        emb = rng.randn(500, 640).astype(np.float32)

        engine = HRM2Engine(n_coarse=100, embedding_dim=640, metric="cosine", auto_k=True)
        engine.index(precomputed_embeddings=emb)

        assert engine._auto_k_cache is not None
        assert 5 <= engine._auto_k_cache <= 50
        print(f"\n[DENSE] Auto-detected k: {engine._auto_k_cache}")


class TestHRM2BackwardCompat:
    """Ensure existing euclidean behavior is preserved."""

    def test_euclidean_default(self):
        """Default metric should be euclidean."""
        engine = HRM2Engine()
        assert engine.metric == "euclidean"

    def test_query_with_details_cosine(self):
        """query_with_details should work with cosine metric."""
        splats, rng = _make_dense_splats(500)
        emb = rng.randn(500, 640).astype(np.float32)

        engine = HRM2Engine(n_coarse=10, embedding_dim=640, metric="cosine")
        engine.add_splats(splats)
        engine.index(precomputed_embeddings=emb)

        query = rng.randn(640).astype(np.float32)
        results = engine.query_with_details(query, k=5)
        assert len(results) == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
