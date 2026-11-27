"""
HTTP 400 para payloads inválidos en los endpoints de escritura (pendiente de P1,
audit r1-02/r1-19/r1-23): el ValueError de validación de SimpleVectorDB.add/update
(NaN/Inf, norma ~0, ids inválidos) debe salir como 400 con el mensaje, no como 500.
"""

import json

import pytest
from fastapi.testclient import TestClient

from m2m.api.edge_api import _manager
from m2m.api.edge_api import app as edge_app

client = TestClient(edge_app)

DIM = 32
NAME = "p6_validation"
_JSON = {"Content-Type": "application/json"}


def _cleanup():
    for name in list(_manager.list_names()):
        try:
            _manager.delete(name)
        except Exception:  # noqa: BLE001
            pass


@pytest.fixture(autouse=True)
def _isolated_collections():
    _cleanup()
    yield
    _cleanup()


@pytest.fixture
def collection():
    r = client.post("/v1/collections", json={"name": NAME, "dimension": DIM})
    assert r.status_code == 201
    return NAME


def _vec(fill=1.0):
    return [fill] * DIM


def _post(url, **payload):
    # Body JSON crudo: NaN/Infinity no son JSON estándar y json= del TestClient
    # los rechaza del lado cliente; el servidor sí los recibe (json.loads).
    return client.post(url, content=json.dumps(payload), headers=_JSON)


def _put(url, **payload):
    return client.put(url, content=json.dumps(payload), headers=_JSON)


def _assert_400(r, *needles):
    assert r.status_code == 400, r.text
    detail = r.json()["detail"]
    assert all(n.lower() in detail.lower() for n in needles), detail


def test_insert_nan_vector_is_400_and_state_intact(collection):
    r = _post(
        f"/v1/collections/{collection}/vectors",
        ids=["a", "b"],
        vectors=[_vec(), [float("nan")] * DIM],
    )
    _assert_400(r, "NaN")
    assert client.get(f"/v1/collections/{collection}/vectors/a").status_code == 404


def test_insert_inf_vector_is_400(collection):
    r = _post(f"/v1/collections/{collection}/vectors", ids=["a"], vectors=[[float("inf")] * DIM])
    _assert_400(r, "NaN")


def test_insert_zero_vector_is_400(collection):
    r = _post(f"/v1/collections/{collection}/vectors", ids=["z"], vectors=[_vec(0.0)])
    _assert_400(r, "zero")


def test_insert_len_mismatch_is_400(collection):
    r = _post(f"/v1/collections/{collection}/vectors", ids=["a"], vectors=[_vec(), _vec(2.0)])
    _assert_400(r, "len(ids)")


def test_insert_valid_still_works(collection):
    r = client.post(
        f"/v1/collections/{collection}/vectors",
        json={"ids": ["a", "b"], "vectors": [_vec(), _vec(2.0)]},
    )
    assert r.status_code == 200 and r.json()["added"] == 2


def test_update_nan_vector_is_400(collection):
    client.post(f"/v1/collections/{collection}/vectors", json={"ids": ["a"], "vectors": [_vec()]})
    r = _put(f"/v1/collections/{collection}/vectors/a", vector=[float("nan")] * DIM)
    _assert_400(r, "NaN")
    # el doc conserva su vector original
    got = client.get(f"/v1/collections/{collection}/vectors/a").json()
    assert got["vector"] == _vec()


def test_update_upsert_without_valid_vector_is_400(collection):
    r = _put(f"/v1/collections/{collection}/vectors/new", vector=_vec(0.0), upsert=True)
    _assert_400(r, "zero")


def test_update_missing_without_upsert_is_404(collection):
    r = client.put(f"/v1/collections/{collection}/vectors/ghost", json={"vector": _vec()})
    assert r.status_code == 404


def test_delete_missing_is_404(collection):
    assert client.delete(f"/v1/collections/{collection}/vectors/ghost").status_code == 404


def test_legacy_ingest_inf_is_400():
    r = _post("/ingest", doc_ids=["x"], vectors=[[float("-inf")] * 128])
    _assert_400(r, "NaN")
    # y el válido sigue entrando
    r = client.post("/ingest", json={"doc_ids": ["y"], "vectors": [[1.0] * 128]})
    assert r.status_code == 200 and r.json()["added"] == 1
