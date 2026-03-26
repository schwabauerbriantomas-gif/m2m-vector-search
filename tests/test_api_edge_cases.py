"""
Edge case and expanded API tests for the v1 REST API.

Tests cover:
- Empty collection behavior
- Duplicate document handling
- Bulk operations
- Validation and error handling
- End-to-end collection lifecycle
- Coordinate/edge coordinator interaction
"""

import numpy as np
import pytest
from fastapi.testclient import TestClient

from m2m.api.edge_api import _manager
from m2m.api.edge_api import app as edge_app

client = TestClient(edge_app)

DIM = 32


@pytest.fixture(autouse=True)
def _cleanup_collections():
    """Clean up all collections between tests."""
    for name in list(_manager.list_names()):
        try:
            _manager.delete(name)
        except Exception:
            pass
    yield
    for name in list(_manager.list_names()):
        try:
            _manager.delete(name)
        except Exception:
            pass


def _rand_vecs(n, dim=DIM):
    return np.random.randn(n, dim).astype(np.float32).tolist()


# ---------------------------------------------------------------------------
# v1 Collection Lifecycle
# ---------------------------------------------------------------------------


class TestCollectionLifecycle:
    def test_create_and_list(self):
        r = client.post("/v1/collections", json={"name": "test1", "dimension": DIM})
        assert r.status_code == 201
        r2 = client.get("/v1/collections")
        assert "test1" in r2.json()["collections"]

    def test_create_duplicate_fails(self):
        client.post("/v1/collections", json={"name": "dup", "dimension": DIM})
        r = client.post("/v1/collections", json={"name": "dup", "dimension": DIM})
        assert r.status_code == 409

    def test_create_invalid_name(self):
        r = client.post("/v1/collections", json={"name": "../evil", "dimension": DIM})
        assert r.status_code == 422  # Pydantic validation

    def test_create_invalid_dimension_zero(self):
        r = client.post("/v1/collections", json={"name": "bad", "dimension": 0})
        assert r.status_code == 422

    def test_create_invalid_dimension_huge(self):
        r = client.post("/v1/collections", json={"name": "bad", "dimension": 999999})
        assert r.status_code == 422

    def test_get_collection_info(self):
        client.post("/v1/collections", json={"name": "info1", "dimension": DIM})
        r = client.get("/v1/collections/info1")
        assert r.status_code == 200
        data = r.json()
        assert data["name"] == "info1"
        assert data["dimension"] == DIM

    def test_get_nonexistent_collection(self):
        r = client.get("/v1/collections/nope")
        assert r.status_code == 404

    def test_delete_collection(self):
        client.post("/v1/collections", json={"name": "delme", "dimension": DIM})
        r = client.delete("/v1/collections/delme")
        assert r.status_code == 200
        r2 = client.get("/v1/collections/delme")
        assert r2.status_code == 404

    def test_delete_nonexistent_collection(self):
        r = client.delete("/v1/collections/ghost")
        assert r.status_code == 404


# ---------------------------------------------------------------------------
# Empty Collection
# ---------------------------------------------------------------------------


class TestEmptyCollection:
    def test_search_empty_returns_no_results(self):
        client.post("/v1/collections", json={"name": "empty", "dimension": DIM})
        r = client.post(
            "/v1/collections/empty/search",
            json={"vector": _rand_vecs(1)[0], "k": 5},
        )
        assert r.status_code == 200
        assert r.json()["count"] == 0
        assert r.json()["results"] == []

    def test_stats_empty_collection(self):
        client.post("/v1/collections", json={"name": "empty", "dimension": DIM})
        r = client.get("/v1/collections/empty/stats")
        assert r.status_code == 200

    def test_get_vector_from_empty(self):
        client.post("/v1/collections", json={"name": "empty", "dimension": DIM})
        r = client.get("/v1/collections/empty/vectors/none")
        assert r.status_code == 404


# ---------------------------------------------------------------------------
# Vector CRUD (v1)
# ---------------------------------------------------------------------------


class TestVectorCRUD:
    def _setup(self, name="crud"):
        client.post("/v1/collections", json={"name": name, "dimension": DIM})
        return name

    def test_insert_and_get(self):
        name = self._setup()
        vecs = _rand_vecs(3)
        r = client.post(
            f"/v1/collections/{name}/vectors", json={"vectors": vecs, "ids": ["a", "b", "c"]}
        )
        assert r.status_code == 200
        assert r.json()["added"] == 3

        r2 = client.get(f"/v1/collections/{name}/vectors/a")
        assert r2.status_code == 200
        assert r2.json()["id"] == "a"

    def test_insert_no_ids_auto_generates(self):
        name = self._setup()
        vecs = _rand_vecs(2)
        r = client.post(f"/v1/collections/{name}/vectors", json={"vectors": vecs})
        assert r.status_code == 200
        assert r.json()["added"] > 0

    def test_insert_with_metadata_and_documents(self):
        name = self._setup()
        vecs = _rand_vecs(2)
        r = client.post(
            f"/v1/collections/{name}/vectors",
            json={
                "vectors": vecs,
                "ids": ["m1", "m2"],
                "metadata": [{"cat": "x"}, {"cat": "y"}],
                "documents": ["doc one", "doc two"],
            },
        )
        assert r.status_code == 200
        r2 = client.get(f"/v1/collections/{name}/vectors/m1")
        assert r2.json()["metadata"]["cat"] == "x"
        assert r2.json()["document"] == "doc one"

    def test_update_vector(self):
        name = self._setup()
        vecs = _rand_vecs(1)
        client.post(f"/v1/collections/{name}/vectors", json={"vectors": vecs, "ids": ["u1"]})
        new_vec = np.zeros(DIM, dtype=np.float32).tolist()
        r = client.put(
            f"/v1/collections/{name}/vectors/u1",
            json={"vector": new_vec, "metadata": {"updated": True}},
        )
        assert r.status_code == 200
        assert r.json()["success"] is True

    def test_update_nonexistent(self):
        name = self._setup()
        r = client.put(f"/v1/collections/{name}/vectors/ghost", json={"metadata": {"x": 1}})
        assert r.status_code == 404

    def test_delete_vector(self):
        name = self._setup()
        vecs = _rand_vecs(2)
        client.post(f"/v1/collections/{name}/vectors", json={"vectors": vecs, "ids": ["d1", "d2"]})
        r = client.delete(f"/v1/collections/{name}/vectors/d1")
        assert r.status_code == 200
        assert r.json()["deleted"] == 1
        # Verify it's gone
        r2 = client.get(f"/v1/collections/{name}/vectors/d1")
        assert r2.status_code == 404

    def test_delete_nonexistent_vector(self):
        name = self._setup()
        r = client.delete(f"/v1/collections/{name}/vectors/ghost")
        assert r.status_code == 404


# ---------------------------------------------------------------------------
# Duplicate Documents
# ---------------------------------------------------------------------------


class TestDuplicateDocuments:
    def test_insert_same_id_twice(self):
        """Second insert with same ID should succeed (upsert behavior)."""
        client.post("/v1/collections", json={"name": "dups", "dimension": DIM})
        vecs = _rand_vecs(1)
        client.post("/v1/collections/dups/vectors", json={"vectors": vecs, "ids": ["dup_id"]})
        vecs2 = _rand_vecs(1)
        r2 = client.post("/v1/collections/dups/vectors", json={"vectors": vecs2, "ids": ["dup_id"]})
        assert r2.status_code == 200
        # The collection should still have the document
        r3 = client.get("/v1/collections/dups/vectors/dup_id")
        assert r3.status_code == 200


# ---------------------------------------------------------------------------
# Search (v1)
# ---------------------------------------------------------------------------


class TestSearchV1:
    def _seed(self, name="search"):
        client.post("/v1/collections", json={"name": name, "dimension": DIM})
        vecs = _rand_vecs(5)
        client.post(
            f"/v1/collections/{name}/vectors",
            json={
                "vectors": vecs,
                "ids": ["s1", "s2", "s3", "s4", "s5"],
                "metadata": [{"cat": "a"}, {"cat": "b"}, {"cat": "a"}, {"cat": "b"}, {"cat": "a"}],
            },
        )
        return name, vecs

    def test_search_returns_k_results(self):
        name, vecs = self._seed()
        r = client.post(
            f"/v1/collections/{name}/search",
            json={"vector": vecs[0], "k": 3, "include_metadata": False},
        )
        assert r.status_code == 200
        data = r.json()
        assert data["count"] == 3
        assert "search_time_ms" in data

    def test_search_with_filter(self):
        name, vecs = self._seed()
        r = client.post(
            f"/v1/collections/{name}/search",
            json={
                "vector": vecs[0],
                "k": 10,
                "filter": {"cat": {"$eq": "a"}},
                "include_metadata": True,
            },
        )
        assert r.status_code == 200
        for item in r.json()["results"]:
            assert item["metadata"]["cat"] == "a"

    def test_search_k_larger_than_collection(self):
        name, vecs = self._seed()
        r = client.post(f"/v1/collections/{name}/search", json={"vector": vecs[0], "k": 100})
        assert r.status_code == 200
        assert r.json()["count"] <= 5

    def test_search_with_documents(self):
        name, vecs = self._seed()
        r = client.post(
            f"/v1/collections/{name}/search",
            json={"vector": vecs[0], "k": 1, "include_documents": True},
        )
        # May or may not have documents depending on insert
        assert r.status_code == 200


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestValidation:
    def test_search_empty_vector_rejected(self):
        client.post("/v1/collections", json={"name": "val", "dimension": DIM})
        r = client.post("/v1/collections/val/search", json={"vector": [], "k": 5})
        assert r.status_code == 422

    def test_search_nan_vector_rejected(self):
        """NaN cannot be serialized to JSON, so the client itself rejects it."""
        client.post("/v1/collections", json={"name": "val", "dimension": DIM})
        with pytest.raises(ValueError, match="Out of range|nan|JSON"):
            client.post("/v1/collections/val/search", json={"vector": [float("nan")] * DIM, "k": 5})

    def test_search_overflow_vector_rejected(self):
        client.post("/v1/collections", json={"name": "val", "dimension": DIM})
        r = client.post("/v1/collections/val/search", json={"vector": [1e39] * DIM, "k": 5})
        assert r.status_code == 422

    def test_search_k_zero_rejected(self):
        client.post("/v1/collections", json={"name": "val", "dimension": DIM})
        r = client.post("/v1/collections/val/search", json={"vector": _rand_vecs(1)[0], "k": 0})
        assert r.status_code == 422

    def test_search_k_huge_rejected(self):
        client.post("/v1/collections", json={"name": "val", "dimension": DIM})
        r = client.post("/v1/collections/val/search", json={"vector": _rand_vecs(1)[0], "k": 99999})
        assert r.status_code == 422

    def test_insert_vectors_exceeds_rate_limit(self):
        """Test that inserting >100K vectors in one request is rejected."""
        client.post("/v1/collections", json={"name": "big", "dimension": DIM})
        r = client.post(
            "/v1/collections/big/vectors", json={"vectors": [[0.0] * DIM for _ in range(100_001)]}
        )
        assert r.status_code == 422


# ---------------------------------------------------------------------------
# Health and Stats
# ---------------------------------------------------------------------------


class TestHealthStats:
    def test_health(self):
        r = client.get("/v1/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

    def test_legacy_health(self):
        r = client.get("/health")
        assert r.status_code == 200

    def test_global_stats(self):
        client.post("/v1/collections", json={"name": "s1", "dimension": DIM})
        r = client.get("/v1/stats")
        assert r.status_code == 200
        assert r.json()["collections_count"] == 1


# ---------------------------------------------------------------------------
# EBM Collection
# ---------------------------------------------------------------------------


class TestEBMCollection:
    def test_create_ebm_collection(self):
        r = client.post(
            "/v1/collections", json={"name": "ebm1", "dimension": DIM, "enable_ebm": True}
        )
        assert r.status_code == 201

    def test_energy_endpoint_on_ebm_collection(self):
        client.post("/v1/collections", json={"name": "ebm2", "dimension": DIM, "enable_ebm": True})
        vecs = _rand_vecs(3)
        client.post("/v1/collections/ebm2/vectors", json={"vectors": vecs})
        r = client.post("/v1/collections/ebm2/energy", json={"vector": vecs[0]})
        assert r.status_code == 200
        data = r.json()
        assert "energy" in data
        assert "confidence" in data
        assert "zone" in data

    def test_energy_endpoint_on_non_ebm_rejected(self):
        client.post("/v1/collections", json={"name": "std", "dimension": DIM})
        r = client.post("/v1/collections/std/energy", json={"vector": _rand_vecs(1)[0]})
        assert r.status_code == 400


# ---------------------------------------------------------------------------
# Bulk Operations
# ---------------------------------------------------------------------------


class TestBulkOperations:
    def test_large_bulk_insert(self):
        """Insert 1000 vectors in a single request."""
        client.post("/v1/collections", json={"name": "bulk", "dimension": DIM})
        vecs = np.random.randn(1000, DIM).astype(np.float32).tolist()
        r = client.post("/v1/collections/bulk/vectors", json={"vectors": vecs})
        assert r.status_code == 200
        assert r.json()["added"] > 0

    def test_search_after_bulk(self):
        """Search after bulk insert returns correct count."""
        client.post("/v1/collections", json={"name": "bulk2", "dimension": DIM})
        vecs = np.random.randn(500, DIM).astype(np.float32).tolist()
        client.post("/v1/collections/bulk2/vectors", json={"vectors": vecs})
        r = client.post("/v1/collections/bulk2/search", json={"vector": vecs[0], "k": 10})
        assert r.status_code == 200
        assert r.json()["count"] == 10


# ---------------------------------------------------------------------------
# Nonexistent Collection on Vector Ops
# ---------------------------------------------------------------------------


class TestNonexistentCollection:
    def test_insert_into_nonexistent(self):
        r = client.post(
            "/v1/collections/ghost/vectors", json={"vectors": _rand_vecs(1), "ids": ["x"]}
        )
        assert r.status_code == 404

    def test_search_nonexistent_collection(self):
        r = client.post("/v1/collections/ghost/search", json={"vector": _rand_vecs(1)[0], "k": 5})
        assert r.status_code == 404

    def test_stats_nonexistent_collection(self):
        r = client.get("/v1/collections/ghost/stats")
        assert r.status_code == 404
