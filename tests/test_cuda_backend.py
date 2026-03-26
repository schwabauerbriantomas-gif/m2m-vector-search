"""CUDA backend tests for M2M.

Skips all tests if no NVIDIA GPU / CUDA is available.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pytest

# Check CUDA availability once
try:
    import torch

    CUDA_AVAILABLE = torch.cuda.is_available()
except ImportError:
    CUDA_AVAILABLE = False

CUDA_SKIP_REASON = "CUDA not available (no NVIDIA GPU or PyTorch not installed)"

requires_cuda = pytest.mark.skipif(not CUDA_AVAILABLE, reason=CUDA_SKIP_REASON)


def _make_embeddings(n: int, dim: int = 64) -> np.ndarray:
    """Generate random normalized embeddings."""
    rng = np.random.default_rng(42)
    emb = rng.standard_normal((n, dim)).astype(np.float32)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-8, None)
    return emb / norms


def _cpu_search(embeddings: np.ndarray, query: np.ndarray, k: int, metric: str = "cosine"):
    """Reference CPU brute-force search."""
    if query.ndim == 1:
        query = query[np.newaxis, :]
    if metric == "cosine":
        q_norms = np.linalg.norm(query, axis=1, keepdims=True)
        x_norms = np.linalg.norm(embeddings, axis=1, keepdims=True).T
        dots = query @ embeddings.T
        scores = dots / (q_norms * x_norms + 1e-8)
        dists = 1.0 - scores
    else:
        diff = query[:, np.newaxis, :] - embeddings[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
    idx = np.argsort(dists, axis=1)[:, :k]
    top_dists = np.take_along_axis(dists, idx, axis=1)
    return idx, top_dists


@requires_cuda
class TestCUDASearcher:
    """Tests for CUDASearcher when CUDA is available."""

    def test_basic_search_matches_cpu(self):
        from m2m.cuda_search import CUDASearcher

        emb = _make_embeddings(1000, 64)
        query = emb[0]

        searcher = CUDASearcher(emb, metric="cosine")
        gpu_idx, gpu_dists = searcher.search(query, k=10)
        cpu_idx, cpu_dists = _cpu_search(emb, query, 10, "cosine")

        # Indices should match exactly for brute-force cosine
        np.testing.assert_array_equal(gpu_idx, cpu_idx[0])
        np.testing.assert_allclose(gpu_dists, cpu_dists[0], atol=1e-5)

    def test_10k_embeddings(self):
        from m2m.cuda_search import CUDASearcher

        emb = _make_embeddings(10000, 64)
        query = emb[0]

        searcher = CUDASearcher(emb, metric="cosine")
        gpu_idx, gpu_dists = searcher.search(query, k=10)
        cpu_idx, cpu_dists = _cpu_search(emb, query, 10, "cosine")

        np.testing.assert_array_equal(gpu_idx, cpu_idx[0])

    def test_empty_input_raises(self):
        from m2m.cuda_search import CUDASearcher

        with pytest.raises(ValueError):
            CUDASearcher(np.array([], dtype=np.float32).reshape(0, 64))

    def test_single_query(self):
        from m2m.cuda_search import CUDASearcher

        emb = _make_embeddings(100, 64)
        query = emb[5]

        searcher = CUDASearcher(emb, metric="cosine")
        gpu_idx, gpu_dists = searcher.search(query, k=1)

        assert gpu_idx.shape == (1,)
        assert gpu_idx[0] == 5  # self should be nearest for cosine
        assert gpu_dists[0] < 1e-5

    def test_batch_queries(self):
        from m2m.cuda_search import CUDASearcher

        emb = _make_embeddings(500, 64)
        queries = emb[:5]

        searcher = CUDASearcher(emb, metric="cosine")
        gpu_idx, gpu_dists = searcher.search_batch(queries, k=5)

        assert gpu_idx.shape == (5, 5)
        assert gpu_dists.shape == (5, 5)

        # First query should match single search
        single_idx, single_dists = searcher.search(queries[0], k=5)
        np.testing.assert_array_equal(gpu_idx[0], single_idx)
        np.testing.assert_allclose(gpu_dists[0], single_dists, atol=1e-5)

    def test_no_nan_inf(self):
        from m2m.cuda_search import CUDASearcher

        emb = _make_embeddings(1000, 64)
        rng = np.random.default_rng(123)
        query = rng.standard_normal(64).astype(np.float32)

        searcher = CUDASearcher(emb, metric="cosine")
        _, dists = searcher.search(query, k=100)

        assert not np.any(np.isnan(dists)), "NaN found in distances"
        assert not np.any(np.isinf(dists)), "Inf found in distances"

    @pytest.mark.parametrize("k", [1, 10, 100])
    def test_various_k(self, k):
        from m2m.cuda_search import CUDASearcher

        emb = _make_embeddings(500, 64)
        query = emb[0]

        searcher = CUDASearcher(emb, metric="cosine")
        gpu_idx, gpu_dists = searcher.search(query, k=k)

        assert gpu_idx.shape == (k,)
        assert gpu_dists.shape == (k,)
        # Distances should be sorted ascending
        assert np.all(np.diff(gpu_dists) >= -1e-6)

    def test_l2_metric(self):
        from m2m.cuda_search import CUDASearcher

        emb = _make_embeddings(1000, 64)
        query = emb[0]

        searcher = CUDASearcher(emb, metric="l2")
        gpu_idx, gpu_dists = searcher.search(query, k=10)
        cpu_idx, cpu_dists = _cpu_search(emb, query, 10, "l2")

        np.testing.assert_array_equal(gpu_idx, cpu_idx[0])
        np.testing.assert_allclose(gpu_dists, cpu_dists[0], atol=1e-3)

    def test_rebuild(self):
        from m2m.cuda_search import CUDASearcher

        emb1 = _make_embeddings(100, 64)
        searcher = CUDASearcher(emb1, metric="cosine")
        assert searcher.n_vectors == 100

        emb2 = _make_embeddings(200, 64)
        searcher.rebuild(emb2)
        assert searcher.n_vectors == 200

        query = emb2[0]
        idx, dists = searcher.search(query, k=1)
        assert idx[0] == 0

    def test_memory_no_leak(self):
        """Verify VRAM usage doesn't grow unbounded after repeated searches."""
        import torch

        from m2m.cuda_search import CUDASearcher

        emb = _make_embeddings(5000, 64)
        searcher = CUDASearcher(emb, metric="cosine")

        query = emb[0]

        torch.cuda.reset_peak_memory_stats()
        baseline_mem = torch.cuda.memory_allocated()

        # Run 100 searches
        for _ in range(100):
            searcher.search(query, k=10)

        after_mem = torch.cuda.memory_allocated()
        peak_mem = torch.cuda.max_memory_allocated()

        # Memory should not grow more than 1MB beyond baseline
        growth = after_mem - baseline_mem
        assert growth < 1024 * 1024, f"VRAM grew by {growth / 1024:.1f} KB after 100 searches"


@pytest.mark.gpu
def test_cuda_searcher_raises_runtime_when_no_cuda(monkeypatch):
    """CUDASearcher.__init__ raises RuntimeError when CUDA is unavailable."""
    import m2m.cuda_search as cs

    # Monkeypatch _has_cuda to always return False
    monkeypatch.setattr(cs, "_has_cuda", lambda: False)

    emb = np.random.randn(100, 64).astype(np.float32)
    with pytest.raises(RuntimeError, match="CUDA not available"):
        cs.CUDASearcher(emb)


def test_create_gpu_index_returns_none_when_no_backends(monkeypatch):
    """create_gpu_index returns None (CPU fallback) when CUDA and Vulkan are unavailable."""
    import m2m.gpu_vector_index as gi

    monkeypatch.setattr(gi, "_has_cuda", lambda: False)
    monkeypatch.setattr(gi, "_has_vulkan", lambda: False)

    emb = np.random.randn(100, 64).astype(np.float32)
    result = gi.create_gpu_index(emb)
    assert result is None, "Expected None (CPU fallback) when no GPU backends available"


@requires_cuda
class TestMultiStartSearcher:
    def test_basic(self):
        from m2m.cuda_search import MultiStartSearcher

        emb = _make_embeddings(1000, 64)
        query = emb[0]

        searcher = MultiStartSearcher(emb, n_starts=3, metric="cosine")
        idx, dists = searcher.search(query, k=10)

        assert idx.shape == (10,)
        # Self should still be in top results
        assert 0 in idx
