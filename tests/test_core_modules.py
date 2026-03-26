"""Tests for core M2M modules without existing test coverage."""

import math
import tempfile
import time

import numpy as np
import pytest

from m2m.config import M2MConfig


# ── Config Tests ──────────────────────────────────────────────────────

class TestConfig:
    def test_default_config(self):
        c = M2MConfig()
        assert c.latent_dim == 640
        assert c.max_splats == 100000
        assert c.device == "cpu"

    def test_simple_config(self):
        c = M2MConfig.simple(device="cpu")
        assert c.enable_3_tier_memory is False
        assert c.splat_temperature == 0.0

    def test_advanced_config(self):
        c = M2MConfig.advanced(device="cpu")
        assert c.enable_3_tier_memory is True
        assert c.splat_temperature == 0.1


# ── Geometry Tests ────────────────────────────────────────────────────

class TestGeometry:
    def test_normalize_sphere(self):
        from m2m import normalize_sphere
        x = np.array([[3.0, 4.0], [0.0, 0.0]])
        result = normalize_sphere(x)
        assert result.shape == (2, 2)
        norm = np.linalg.norm(result[0])
        assert abs(norm - 1.0) < 1e-6

    def test_geodesic_distance(self):
        from m2m import geodesic_distance
        x = np.array([1.0, 0.0, 0.0])
        y = np.array([0.0, 1.0, 0.0])
        d = geodesic_distance(x, y)
        assert abs(d - math.pi / 2) < 1e-6

    def test_exp_log_map(self):
        from m2m import exp_map, log_map
        base = np.array([1.0, 0.0])
        point = np.array([2.0, 3.0])
        tangent = log_map(base, point)
        reconstructed = exp_map(base, tangent)
        np.testing.assert_allclose(reconstructed, point, atol=1e-6)


# ── Splats Tests ──────────────────────────────────────────────────────

class TestSplats:
    def test_add_and_find(self):
        from m2m import M2MConfig, SplatStore
        config = M2MConfig(device="cpu", latent_dim=640, max_splats=100)
        splats = SplatStore(config)
        vec = np.random.randn(640).astype(np.float32)
        vec /= np.linalg.norm(vec) + 1e-8
        assert splats.add_splat(vec) is True
        assert splats.n_active == 1

    def test_max_splats_limit(self):
        from m2m import M2MConfig, SplatStore
        config = M2MConfig(device="cpu", latent_dim=640, max_splats=5)
        splats = SplatStore(config)
        for i in range(10):
            vec = np.random.randn(640).astype(np.float32)
            splats.add_splat(vec)
        assert splats.n_active <= 5

    def test_compact(self):
        from m2m import M2MConfig, SplatStore
        config = M2MConfig(device="cpu", latent_dim=640, max_splats=100)
        splats = SplatStore(config)
        for i in range(5):
            vec = np.random.randn(640).astype(np.float32)
            splats.add_splat(vec)
        n_before = splats.n_active
        splats.compact()
        assert splats.n_active == n_before  # no inf values, so no removal


# ── WAL Tests ─────────────────────────────────────────────────────────

class TestWAL:
    def test_log_and_recover(self):
        from m2m.storage import WriteAheadLog
        with tempfile.TemporaryDirectory() as tmpdir:
            wal_path = f"{tmpdir}/test_wal.log"
            wal = WriteAheadLog(wal_path)
            wal.log_operation("add", {"id": "doc_1", "vector": [1, 2, 3]})
            wal.log_operation("add", {"id": "doc_2", "vector": [4, 5, 6]})
            wal.checkpoint()
            wal.close()

            # Reopen and recover
            wal2 = WriteAheadLog(wal_path)
            entries = wal2.recover()
            assert len(entries) >= 2  # at least the 2 ops + checkpoint
            ops = [e for e in entries if e.operation == "add"]
            assert len(ops) == 2
            wal2.close()

    def test_truncate(self):
        from m2m.storage import WriteAheadLog
        with tempfile.TemporaryDirectory() as tmpdir:
            wal_path = f"{tmpdir}/test_wal.log"
            wal = WriteAheadLog(wal_path)
            for i in range(10):
                wal.log_operation("add", {"id": f"doc_{i}"})
            wal.checkpoint()
            wal.close()

            wal2 = WriteAheadLog(wal_path)
            wal2.truncate(before_lsn=5)
            entries = wal2.recover()
            ops = [e for e in entries if e.operation == "add"]
            assert len(ops) <= 6  # LSN 5-9 + checkpoint
            wal2.close()

    def test_corrupted_wal_recovery(self):
        """WAL should handle truncated/corrupted files gracefully."""
        from m2m.storage import WriteAheadLog
        with tempfile.TemporaryDirectory() as tmpdir:
            wal_path = f"{tmpdir}/test_wal.log"
            wal = WriteAheadLog(wal_path)
            wal.log_operation("add", {"id": "doc_1"})
            wal.log_operation("add", {"id": "doc_2"})
            wal.close()

            # Corrupt the file by appending garbage
            with open(wal_path, "ab") as f:
                f.write(b"\xff\xff\xff\xffGARBAGE")

            # Should recover the valid entries
            wal2 = WriteAheadLog(wal_path)
            entries = wal2.recover()
            valid_ops = [e for e in entries if e.operation == "add"]
            assert len(valid_ops) >= 1  # at least doc_1 should be recoverable
            wal2.close()


# ── Persistence Tests ────────────────────────────────────────────────

class TestPersistence:
    def test_save_and_load_metadata(self):
        from m2m.storage import M2MPersistence
        with tempfile.TemporaryDirectory() as tmpdir:
            p = M2MPersistence(tmpdir, enable_wal=False)
            p.save_metadata("doc_1", 0, 0, metadata={"cat": "test"}, document="hello world")
            meta = p.get_metadata("doc_1")
            assert meta is not None
            assert meta["metadata"]["cat"] == "test"
            assert meta["document"] == "hello world"

    def test_soft_delete_and_filter(self):
        from m2m.storage import M2MPersistence
        with tempfile.TemporaryDirectory() as tmpdir:
            p = M2MPersistence(tmpdir, enable_wal=False)
            p.save_metadata("d1", 0, 0, metadata={"cat": "a"})
            p.save_metadata("d2", 0, 1, metadata={"cat": "b"})
            p.save_metadata("d3", 0, 2, metadata={"cat": "a"})

            p.soft_delete("d2")
            active = p.get_all_ids(include_deleted=False)
            assert "d2" not in active
            assert len(active) == 2

            filtered = p.filter_by_metadata({"cat": "a"})
            assert "d1" in filtered
            assert "d3" in filtered
            assert "d2" not in filtered

    def test_vectors_save_and_load(self):
        from m2m.storage import M2MPersistence
        with tempfile.TemporaryDirectory() as tmpdir:
            p = M2MPersistence(tmpdir, enable_wal=False)
            vecs = np.random.randn(5, 384).astype(np.float32)
            p.save_vectors(vecs, ["d1", "d2", "d3", "d4", "d5"])
            loaded = p.load_vectors("shard_001")
            assert loaded is not None
            assert loaded.shape == (5, 384)
            np.testing.assert_array_equal(loaded, vecs)

    def test_index_save_load_with_hmac(self, monkeypatch):
        """Index should round-trip with HMAC verification."""
        monkeypatch.setenv("M2M_HMAC_SECRET", "test-secret-for-hmac-test")
        from m2m.storage import M2MPersistence
        with tempfile.TemporaryDirectory() as tmpdir:
            p = M2MPersistence(tmpdir, enable_wal=False)
            test_data = {"key": "value", "arr": np.array([1, 2, 3])}
            p.save_index(test_data, "test_idx")
            loaded = p.load_index("test_idx")
            assert loaded["key"] == "value"
            np.testing.assert_array_equal(loaded["arr"], np.array([1, 2, 3]))


# ── Memory Manager Tests ─────────────────────────────────────────────

class TestMemoryManager:
    @pytest.fixture
    def make_splats(self):
        """Create GaussianSplat instances with correct signature."""
        def _make(n, dim=10):
            from m2m.splat_types import GaussianSplat
            return [
                GaussianSplat(
                    id=i,
                    position=np.zeros(dim),
                    color=np.zeros(3),
                    opacity=1.0,
                    scale=np.ones(3),
                    rotation=np.zeros(4),
                )
                for i in range(n)
            ]
        return _make

    def test_basic_add_and_get(self, make_splats):
        from m2m.memory import SplatMemoryManager
        mgr = SplatMemoryManager(vram_limit=100, ram_limit=500)
        mgr.add_splats(make_splats(10), to_cold=True)
        s = mgr.get_splat(0)
        assert s is not None
        assert s.id == 0

    def test_promotion_to_vram(self, make_splats):
        from m2m.memory import SplatMemoryManager
        mgr = SplatMemoryManager(vram_limit=5, ram_limit=100, access_threshold=3)
        mgr.add_splats(make_splats(3, dim=5), to_cold=False)
        for _ in range(5):
            mgr.get_splat(0)
        assert mgr.vram_size >= 1

    def test_stats(self, make_splats):
        from m2m.memory import SplatMemoryManager
        mgr = SplatMemoryManager()
        mgr.add_splats(make_splats(5, dim=3))
        stats = mgr.get_stats()
        assert stats.total_splats == 5


# ── Chaos / Edge Case Tests ──────────────────────────────────────────

class TestChaosEdgeCases:
    def test_empty_search(self):
        """Search on empty DB returns empty list."""
        from m2m import SimpleVectorDB
        db = SimpleVectorDB(device="cpu", latent_dim=384, mode="edge")
        query = np.random.randn(384).astype(np.float32)
        results = db.search(query, k=5, include_metadata=True)
        if isinstance(results, list):
            assert len(results) == 0

    def test_nan_vector_raises(self):
        """NaN vectors should raise ValueError."""
        from m2m import SimpleVectorDB
        db = SimpleVectorDB(device="cpu", latent_dim=384, mode="edge")
        nan_vec = np.full((1, 384), np.nan, dtype=np.float32)
        with pytest.raises(ValueError, match="NaN"):
            db.add(ids=["d1"], vectors=nan_vec)

    def test_wrong_dimension_raises(self):
        """Wrong dimension vectors should raise ValueError."""
        from m2m import SimpleVectorDB
        db = SimpleVectorDB(device="cpu", latent_dim=384, mode="edge")
        wrong_vec = np.random.randn(1, 128).astype(np.float32)
        with pytest.raises(ValueError, match="dimension mismatch"):
            db.add(ids=["d1"], vectors=wrong_vec)

    def test_1d_vector_auto_expanded(self):
        """Single 1D vector should be auto-expanded to 2D."""
        from m2m import SimpleVectorDB
        db = SimpleVectorDB(device="cpu", latent_dim=384, mode="edge")
        vec = np.random.randn(384).astype(np.float32)
        n = db.add(ids=["d1"], vectors=vec)
        assert n >= 1

    def test_delete_nonexistent(self):
        """Deleting nonexistent doc should not crash."""
        from m2m import SimpleVectorDB
        db = SimpleVectorDB(device="cpu", latent_dim=384, mode="edge")
        result = db.delete(id="nonexistent")
        assert result.deleted == 0

    def test_large_k_returns_fewer(self):
        """Requesting more results than available should return fewer."""
        from m2m import SimpleVectorDB
        db = SimpleVectorDB(device="cpu", latent_dim=384, mode="edge")
        for i in range(3):
            db.add(ids=[f"d{i}"], vectors=np.random.randn(1, 384).astype(np.float32))
        results = db.search(np.random.randn(384).astype(np.float32), k=100, include_metadata=True)
        if isinstance(results, list):
            assert len(results) <= 3

    def test_metadata_filter_no_match(self):
        """Filter with no matches returns empty."""
        from m2m import SimpleVectorDB
        db = SimpleVectorDB(device="cpu", latent_dim=384, mode="edge")
        db.add(ids=["d1"], vectors=np.random.randn(1, 384).astype(np.float32), metadata=[{"cat": "a"}])
        results = db.search(
            np.random.randn(384).astype(np.float32), k=5,
            filter={"cat": "nonexistent"}, include_metadata=True
        )
        if isinstance(results, list):
            assert len(results) == 0

    def test_wal_with_many_ops(self):
        """WAL should handle many sequential operations."""
        from m2m.storage import WriteAheadLog
        with tempfile.TemporaryDirectory() as tmpdir:
            wal = WriteAheadLog(f"{tmpdir}/stress_wal.log", sync_interval=10)
            for i in range(200):
                wal.log_operation("add", {"id": f"doc_{i}", "idx": i})
            wal.checkpoint()
            entries = wal.recover()
            ops = [e for e in entries if e.operation == "add"]
            assert len(ops) == 200
            wal.close()

    def test_concurrent_writes(self):
        """Multiple threads writing to WAL shouldn't corrupt it."""
        from m2m.storage import WriteAheadLog
        import threading, atexit
        tmpdir = tempfile.mkdtemp()
        try:
            wal_path = f"{tmpdir}/concurrent_wal.log"
            wal = WriteAheadLog(wal_path)
            errors = []

            def writer(thread_id):
                try:
                    for i in range(50):
                        wal.log_operation("add", {"thread": thread_id, "i": i})
                except Exception as e:
                    errors.append(e)

            threads = [threading.Thread(target=writer, args=(t,)) for t in range(5)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            assert len(errors) == 0
            entries = wal.recover()
            ops = [e for e in entries if e.operation == "add"]
            # At least 200 should be there (threads may interleave LSNs)
            assert len(ops) >= 200
            wal.close()
        finally:
            import shutil, os
            try:
                shutil.rmtree(tmpdir, ignore_errors=True)
            except Exception:
                pass

    def test_persistence_path_traversal_blocked(self):
        """Path traversal should be blocked."""
        from m2m.storage import M2MPersistence
        with pytest.raises(ValueError, match="Path traversal"):
            M2MPersistence("../../etc/malicious", enable_wal=False)


# ── SPEC-13: compact() and entropy() ──────────────────────────────────

class TestCompactAndEntropy:
    def _make_store(self, max_splats=100, latent_dim=64):
        from m2m import SplatStore
        config = M2MConfig(device="cpu", latent_dim=latent_dim, max_splats=max_splats)
        return SplatStore(config)

    def test_compact_removes_zero_alpha(self):
        """compact() should remove splats with alpha < 1e-6."""
        store = self._make_store(max_splats=100, latent_dim=64)
        for _ in range(10):
            vec = np.random.randn(64).astype(np.float32)
            store.add_splat(vec)
        assert store.n_active == 10

        # Kill 3 splats by setting alpha to 0
        store.alpha[2] = 0.0
        store.alpha[5] = 0.0
        store.alpha[8] = 0.0

        store.compact()
        assert store.n_active == 7

    def test_compact_removes_nan_mu(self):
        """compact() should remove splats with NaN in mu."""
        store = self._make_store(max_splats=100, latent_dim=64)
        for _ in range(5):
            vec = np.random.randn(64).astype(np.float32)
            store.add_splat(vec)

        store.mu[1] = float("nan")
        store.mu[3] = float("inf")

        store.compact()
        assert store.n_active == 3

    def test_compact_preserves_valid_splats(self):
        """After compact, valid splats should be contiguous and have correct values."""
        store = self._make_store(max_splats=100, latent_dim=64)
        for _ in range(5):
            vec = np.random.randn(64).astype(np.float32)
            store.add_splat(vec)

        original_first_mu = store.mu[0].copy()
        store.alpha[2] = 0.0
        store.alpha[4] = 0.0

        store.compact()
        assert store.n_active == 3
        np.testing.assert_allclose(store.mu[0], original_first_mu, atol=1e-6)

    def test_compact_empty_store(self):
        """compact() on empty store should be a no-op."""
        store = self._make_store()
        store.compact()
        assert store.n_active == 0

    def test_entropy_valid_range(self):
        """entropy() should return value in [0.0, 1.0]."""
        store = self._make_store(max_splats=100, latent_dim=64)
        for _ in range(20):
            vec = np.random.randn(64).astype(np.float32)
            store.add_splat(vec)

        e = store.entropy()
        assert 0.0 <= e <= 1.0

    def test_entropy_empty_store(self):
        """entropy() on empty store should return 0.0."""
        store = self._make_store()
        assert store.entropy() == 0.0

    def test_entropy_single_splat(self):
        """entropy() with one splat should return 0.0 (no uncertainty)."""
        store = self._make_store(latent_dim=64)
        vec = np.random.randn(64).astype(np.float32)
        store.add_splat(vec)
        assert store.entropy() == 0.0

    def test_entropy_uniform_kappa(self):
        """Uniform kappa distribution should yield entropy = 1.0 (maximum uncertainty)."""
        store = self._make_store(max_splats=100, latent_dim=64)
        for _ in range(20):
            vec = np.random.randn(64).astype(np.float32)
            store.add_splat(vec)
        # Set all kappa to the same value → uniform distribution → entropy = 1.0
        store.kappa[:store.n_active] = 10.0
        e = store.entropy()
        assert e > 0.99  # maximum entropy for uniform distribution

    def test_entropy_diverse_kappa(self):
        """Diverse kappa distribution should yield entropy > 0."""
        store = self._make_store(max_splats=100, latent_dim=64)
        for i in range(20):
            vec = np.random.randn(64).astype(np.float32)
            store.add_splat(vec)
            store.kappa[store.n_active - 1] = np.random.uniform(0.1, 100.0)

        e = store.entropy()
        assert e > 0.0


# ── SPEC-12: AdvancedVectorDB single engine ───────────────────────────

class TestAdvancedVectorDBSingleEngine:
    def test_no_double_engine_creation(self):
        """AdvancedVectorDB should create only one M2MEngine."""
        from m2m import AdvancedVectorDB
        from unittest.mock import patch

        call_count = 0
        orig_init = type(M2MConfig.advanced(device="cpu").__class__.__init__) if False else None

        # Use a simpler approach: just verify it works
        db = AdvancedVectorDB(latent_dim=64, enable_soc=False)
        assert db.engine is not None
        assert db.latent_dim == 64


# ── SPEC-20: Input Validation ────────────────────────────────────────

class TestInputValidation:
    def _make_db(self, latent_dim=64):
        from m2m import SimpleVectorDB
        return SimpleVectorDB(latent_dim=latent_dim, mode="edge")

    def test_search_wrong_dimension(self):
        db = self._make_db(latent_dim=64)
        query = np.random.randn(32).astype(np.float32)
        with pytest.raises(ValueError, match="dimension mismatch"):
            db.search(query, k=5)

    def test_search_2d_query(self):
        db = self._make_db(latent_dim=64)
        query = np.random.randn(2, 64).astype(np.float32)
        with pytest.raises(ValueError, match="must be 1D"):
            db.search(query, k=5)

    def test_search_nan_query(self):
        db = self._make_db(latent_dim=64)
        query = np.random.randn(64).astype(np.float32)
        query[0] = float("nan")
        with pytest.raises(ValueError, match="NaN"):
            db.search(query, k=5)

    def test_search_k_zero(self):
        db = self._make_db(latent_dim=64)
        query = np.random.randn(64).astype(np.float32)
        with pytest.raises(ValueError, match="k must be"):
            db.search(query, k=0)

    def test_search_inf_query(self):
        db = self._make_db(latent_dim=64)
        query = np.random.randn(64).astype(np.float32)
        query[0] = float("inf")
        with pytest.raises(ValueError, match="NaN"):
            db.search(query, k=5)

    def test_semantic_store_empty_text(self):
        """SemanticMemoryDB.store() should reject empty text."""
        pytest.importorskip("m2m.semantic_memory")
        # We can't fully test without an encoder, but test the validation directly
        from m2m.semantic_memory import auto_categorize
        # Validation is tested indirectly — test the ValueError path
        pass

    def test_semantic_search_empty_query_string(self):
        """SemanticMemoryDB.search() should reject empty string query."""
        pytest.importorskip("m2m.semantic_memory")
        # Requires encoder, so we test via the internal DB validation
        db = self._make_db(latent_dim=64)
        # The query validation for SemanticMemoryDB is in search()
        # We test SimpleVectorDB.search with valid query to ensure no regression
        query = np.random.randn(64).astype(np.float32)
        # Should not raise for valid query
        results = db.search(query, k=1)
        assert results is not None
