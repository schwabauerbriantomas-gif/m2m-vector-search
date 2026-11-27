"""Defectos de la HTTP API que encontró el validador independiente de la documentación.

[validator D02] ``POST /search`` con un filtro de metadatos inválido respondía ``500``
(``ValueError`` sin capturar) en vez de ``400``.
[validator D04] ``POST /query`` respondía SIEMPRE ``500`` en colecciones sin EBM (llamaba
al endpoint ``search`` sin el argumento ``request``); el campo del cuerpo es ``vector``.
Con ``include_energy=True`` el filtro no se aplicaba (se ignoraba en silencio): ahora
falla cerrado con ``400``.
[validator D15] ``M2M_CORS_ORIGINS="https://a, https://b"`` (con un espacio) ignoraba el
segundo origen.
[validator D19] una clave no ASCII en el coordinator respondía ``500``
(``hmac.compare_digest`` sobre ``str``); la edge API ya codificaba a bytes.
"""

import numpy as np
import pytest
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.testclient import TestClient

from m2m import SimpleVectorDB
from m2m.api import coordinator_api, edge_api
from m2m.api.edge_api import _manager
from m2m.api.edge_api import app as edge_app

client = TestClient(edge_app)
client_no_raise = TestClient(edge_app, raise_server_exceptions=False)

DIM = 32
V0 = [1.0] + [0.0] * (DIM - 1)
V1 = [0.0, 1.0] + [0.0] * (DIM - 2)

INVALID_FILTERS = [
    {"g": {"$between": [1, 2]}},  # operador desconocido
    {},  # filtro vacío
    {"g": {"$in": "x"}},  # operando que no es una lista
]


@pytest.fixture(autouse=True)
def _clean_api(monkeypatch):
    monkeypatch.setattr(edge_api, "_API_KEY", None)
    monkeypatch.delenv("M2M_API_KEY", raising=False)
    monkeypatch.delenv("M2M_RATE_LIMIT_PER_MIN", raising=False)
    edge_api._rate_limit_store.clear()
    _manager._collections.clear()
    _manager._dbs.clear()
    yield
    _manager._collections.clear()
    _manager._dbs.clear()
    edge_api._rate_limit_store.clear()


def _collection(name="p19", enable_ebm=False):
    r = client.post(
        "/v1/collections", json={"name": name, "dimension": DIM, "enable_ebm": enable_ebm}
    )
    assert r.status_code == 201, r.text
    r = client.post(
        f"/v1/collections/{name}/vectors",
        json={"ids": ["v0", "v1"], "vectors": [V0, V1], "metadata": [{"g": 1}, {"g": 2}]},
    )
    assert r.status_code == 200, r.text
    return name


# --------------------------------------------------------------------------- D02


@pytest.mark.parametrize("bad", INVALID_FILTERS)
def test_search_with_invalid_filter_answers_400_not_500(bad):
    name = _collection()
    r = client_no_raise.post(
        f"/v1/collections/{name}/search", json={"vector": V0, "k": 5, "filter": bad}
    )
    assert r.status_code == 400, (r.status_code, r.text)


def test_search_with_valid_filter_still_works():
    name = _collection()
    r = client.post(
        f"/v1/collections/{name}/search",
        json={"vector": V0, "k": 5, "filter": {"g": 2}, "include_metadata": True},
    )
    assert r.status_code == 200, r.text
    ids = [x["id"] for x in r.json()["results"]]
    assert ids == ["v1"]


# --------------------------------------------------------------------------- D04


def test_query_endpoint_works_on_a_collection_without_ebm():
    name = _collection()
    r = client_no_raise.post(f"/v1/collections/{name}/query", json={"vector": V0, "k": 5})
    assert r.status_code == 200, (r.status_code, r.text)
    body = r.json()
    assert [x["id"] for x in body["results"]] == ["v0", "v1"]
    assert body["count"] == 2
    # mismo orden y mismos scores que /search
    r2 = client.post(f"/v1/collections/{name}/search", json={"vector": V0, "k": 5})
    assert r2.status_code == 200 and r2.json()["results"] == body["results"]


def test_query_body_field_is_vector_not_query():
    name = _collection()
    r = client.post(f"/v1/collections/{name}/query", json={"query": V0, "k": 5})
    assert r.status_code == 422, r.text
    assert "vector" in r.text


def test_query_on_ebm_collection_returns_energy():
    name = _collection("p19ebm", enable_ebm=True)
    r = client_no_raise.post(
        f"/v1/collections/{name}/query", json={"vector": V0, "k": 2, "include_energy": True}
    )
    assert r.status_code == 200, (r.status_code, r.text)
    body = r.json()
    assert "query_energy" in body and len(body["results"]) == 2
    assert all("energy" in x for x in body["results"])


def test_query_with_filter_and_energy_fails_closed():
    """Antes el filtro se ignoraba en silencio en el camino con energía."""
    name = _collection("p19ebm2", enable_ebm=True)
    r = client_no_raise.post(
        f"/v1/collections/{name}/query",
        json={"vector": V0, "k": 2, "include_energy": True, "filter": {"g": 2}},
    )
    assert r.status_code == 400, (r.status_code, r.text)
    assert "/search" in r.json()["detail"]


def test_query_without_energy_applies_the_filter():
    name = _collection()
    r = client_no_raise.post(
        f"/v1/collections/{name}/query", json={"vector": V0, "k": 5, "filter": {"g": 2}}
    )
    assert r.status_code == 200, r.text
    assert [x["id"] for x in r.json()["results"]] == ["v1"]


# --------------------------------------------------------------------------- D06


def test_search_without_include_metadata_returns_empty_metadata():
    """`metadata`/`document` sólo se cargan con include_metadata=True; la docs lo
    prometía también para `filter=` sola."""
    db = SimpleVectorDB(latent_dim=DIM)
    X = np.array([V0, V1], dtype=np.float32)
    db.add(ids=["a", "b"], vectors=X, metadata=[{"g": 1}, {"g": 2}], documents=["doc a", "doc b"])
    res = db.search(X[0], k=2, filter={"g": 1})
    assert [r.id for r in res] == ["a"]
    assert res[0].metadata == {} and res[0].document is None
    full = db.search(X[0], k=2, filter={"g": 1}, include_metadata=True)
    assert full[0].metadata == {"g": 1} and full[0].document == "doc a"


# --------------------------------------------------------------------------- D15


def test_cors_origins_list_tolerates_spaces():
    """Se prueba el parser y la middleware sin recargar el módulo (un reload deja objetos
    ``app``/``_manager`` viejos en los demás módulos de test)."""
    origins = edge_api._parse_cors_origins("https://a.example, https://b.example")
    assert origins == ["https://a.example", "https://b.example"]
    assert edge_api._parse_cors_origins("  ") == []
    assert edge_api._parse_cors_origins("*") == ["*"]
    probe = FastAPI()
    probe.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @probe.get("/ping")
    def _ping():
        return {"ok": True}

    c = TestClient(probe)
    preflight = {"Access-Control-Request-Method": "GET"}
    for origin in origins:
        r = c.options("/ping", headers={"Origin": origin, **preflight})
        assert r.headers.get("access-control-allow-origin") == origin, (origin, r.headers)
    r = c.options("/ping", headers={"Origin": "https://evil.example", **preflight})
    assert "access-control-allow-origin" not in r.headers


# --------------------------------------------------------------------------- D19


class _Request:
    def __init__(self, headers):
        self.headers = headers


@pytest.mark.parametrize(
    "headers",
    [
        {"X-API-Key": "clave-ñ"},
        {"Authorization": "Bearer clave-ñ"},
        {"X-API-Key": "coord-secret"},
    ],
)
def test_coordinator_key_compare_accepts_non_ascii(monkeypatch, headers):
    monkeypatch.setattr(coordinator_api, "_COORDINATOR_API_KEY", "coord-secret")
    try:
        coordinator_api._validate_coordinator_key(_Request(headers))
    except HTTPException as e:
        assert e.status_code == 401
    else:
        assert headers in ({"X-API-Key": "coord-secret"},)
