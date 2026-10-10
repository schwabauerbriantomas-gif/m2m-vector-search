"""Tests de SQ8DiskStore (H10 integrado: store comprimido con rescoring mmap)."""
import numpy as np
import pytest

from m2m.sq8_disk import SQ8DiskStore


def _clustered(n, dim, n_clusters, seed=42):
    rng = np.random.RandomState(seed)
    centers = rng.randn(n_clusters, dim).astype(np.float32)
    data = centers[rng.randint(0, n_clusters, size=n)] + 0.35 * rng.randn(n, dim).astype(np.float32)
    data /= np.linalg.norm(data, axis=1, keepdims=True)
    return data


def test_roundtrip_and_recall(tmp_path):
    dim, n, K = 64, 5000, 10
    data = _clustered(n, dim, 20)
    st = SQ8DiskStore(dim=dim, dump_dir=str(tmp_path))
    st.add(data)
    assert len(st) == n
    st.flush()

    rng = np.random.RandomState(7)
    queries = data[rng.randint(0, n, size=50)]
    gt = np.argsort(2 - 2 * (queries @ data.T), axis=1)[:, :K]
    got = st.search_batch(queries, k=K)
    rec = np.mean([len(set(got[i].tolist()) & set(gt[i].tolist())) / K for i in range(50)])
    assert rec >= 0.95, f"recall@10 {rec:.3f} < 0.95"


def test_load_from_disk(tmp_path):
    dim, n = 32, 1000
    data = _clustered(n, dim, 10)
    st = SQ8DiskStore(dim=dim, dump_dir=str(tmp_path))
    st.add(data)
    st.flush()

    st2 = SQ8DiskStore.load(str(tmp_path))
    assert len(st2) == n
    q = data[0]
    got = st2.search_batch(q, k=5)[0]
    gt = np.argsort(2 - 2 * (data @ q))[:5]
    assert len(set(got.tolist()) & set(gt.tolist())) >= 4


def test_incremental_add():
    dim = 16
    st = SQ8DiskStore(dim=dim)
    st.add(np.eye(dim, dtype=np.float32))
    st.add(np.eye(dim, dtype=np.float32) * -1.0)
    assert len(st) == 2 * dim


def test_dim_mismatch():
    st = SQ8DiskStore(dim=8)
    with pytest.raises(ValueError):
        st.add(np.zeros((3, 9), dtype=np.float32))
    with pytest.raises(ValueError):
        st.search_batch(np.zeros((2, 9), dtype=np.float32))


def test_ram_footprint_is_quarter():
    dim, n = 64, 2000
    data = _clustered(n, dim, 10)
    st = SQ8DiskStore(dim=dim)
    st.add(data)
    expected = n * dim + n * 8  # uint8 + scale/zero float32
    assert st.ram_bytes() <= expected * 1.05
    assert st.ram_bytes() < data.nbytes / 3.0  # al menos ~4x menos


def test_search_requires_flush(tmp_path):
    st = SQ8DiskStore(dim=8)
    st.add(np.random.RandomState(0).randn(100, 8).astype(np.float32))
    with pytest.raises(AssertionError):
        st.search_batch(np.random.randn(2, 8).astype(np.float32))
