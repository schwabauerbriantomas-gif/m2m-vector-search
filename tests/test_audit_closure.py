"""
P15 — cierre de pendientes transversales de código (BACKLOG P15, ítems 1-13).

Tests de regresión de los packs anteriores (P3, P4, P7b, P7c, P8a, P12a, P14).
Los marcados ROJO fallaban antes del fix (``reports/p15_red.log``):

  1. PUT/DELETE/GET de vectores validan la API key cuando ``M2M_API_KEY`` está
     configurada (antes: sin clave se leía, sobreescribía y borraba).
  2. Rate limit REAL y opt-in: ``M2M_RATE_LIMIT_PER_MIN`` (ausente o 0 = apagado,
     que es el default). Antes ``_check_rate_limit`` nunca registraba el pedido,
     así que el tope nominal de 100/min jamás se disparaba.
  3. ``M2MPersistence.restore()`` valida la fuente ANTES de borrar el storage
     (``..``/ausente/dentro del storage/symlink). Nota: no existe
     ``AdvancedVectorDB.restore()``; el único ``restore()`` es el del storage.
  4. ``docker-compose.yml`` raíz no publica 8080 en todas las interfaces.
  5. Coordinator: reenvía ``M2M_EDGE_API_KEY`` a los edges y expone ``GET /health``.
  6. Guardia: md5/sha1 de ``sharding.py``/``gate_modulus.py`` son de bucketing y
     su valor NO cambia al anotarlos ``usedforsecurity=False``.
  7. ``get_stats()['backend']``; ``M2MConfig`` nombra lo SOLICITADO y expone la
     disponibilidad real de CUDA; el fallo del camino CUDA de ``SplatStore`` se
     avisa UNA vez y cae al camino numpy.
  8. ``_doc_results_from_lsh`` evalúa ``filter`` contra la metadata real también
     con ``include_metadata=False``; docstring veraz de ``filter_by_metadata``.
  9. ``SQ8DiskStore.search_batch``: queries NaN/Inf → ``ValueError``.
  10. Docstring de la integración LangChain (``VectorStore``); ``python -m m2m``
      sugiere el extra ``[api]``.
  12. ``FeedbackResult`` en ``__all__``; el estado aprendido por ``feedback()``
      sobrevive al reopen también en ``SimpleVectorDB``; ``update(vector=)`` y el
      hard-delete actualizan el estado persistido (sin resurrecciones).
  13. No queda el wrapper de lock sobre ``engine.search``: el único caller es
      ``SimpleVectorDB.search()``, que ya toma ``_rw_lock`` en todo su cuerpo.
"""

import hashlib
import json
import logging
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

import m2m
from m2m import AdvancedVectorDB, M2MConfig, M2MPersistence, SimpleVectorDB, SplatStore
from m2m.api import coordinator_api, edge_api
from m2m.api.coordinator_api import app as coord_app
from m2m.api.edge_api import _manager
from m2m.api.edge_api import app as edge_app
from m2m.sq8_disk import SQ8DiskStore

ROOT = Path(__file__).resolve().parent.parent

client = TestClient(edge_app)
coord_client = TestClient(coord_app)

DIM = 32
KEY = "p15-secret-key"
AUTH = {"Authorization": f"Bearer {KEY}"}
WRONG = {"X-API-Key": "not-the-key"}
V0 = [1.0] + [0.0] * (DIM - 1)
V1 = [0.0, 1.0] + [0.0] * (DIM - 2)


def _unit(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def _reset_collections():
    _manager._collections.clear()
    _manager._dbs.clear()


@pytest.fixture(autouse=True)
def _clean_api(monkeypatch):
    """Estado de auth y rate limit determinístico por test."""
    monkeypatch.setattr(edge_api, "_API_KEY", None)
    monkeypatch.delenv("M2M_API_KEY", raising=False)
    monkeypatch.delenv("M2M_RATE_LIMIT_PER_MIN", raising=False)
    edge_api._rate_limit_store.clear()
    _reset_collections()
    yield
    _reset_collections()
    edge_api._rate_limit_store.clear()


@pytest.fixture
def api_key(monkeypatch):
    monkeypatch.setattr(edge_api, "_API_KEY", KEY)
    return KEY


def _seed(name="c15", vid="v0"):
    r = client.post("/v1/collections", json={"name": name, "dimension": DIM}, headers=AUTH)
    assert r.status_code == 201, r.text
    r = client.post(
        f"/v1/collections/{name}/vectors", json={"ids": [vid], "vectors": [V0]}, headers=AUTH
    )
    assert r.status_code == 200, r.text
    return f"/v1/collections/{name}/vectors/{vid}"


# ---------------------------------------------------------------------------
# 1. PUT/DELETE/GET de vectores validan la API key
# ---------------------------------------------------------------------------


class TestVectorEndpointsValidateKey:
    def test_get_without_key_is_401(self, api_key):
        url = _seed()
        assert client.get(url).status_code == 401
        assert client.get(url, headers=WRONG).status_code == 401
        r = client.get(url, headers=AUTH)
        assert r.status_code == 200, r.text
        assert r.json()["id"] == "v0"

    def test_put_without_key_is_401_and_does_not_write(self, api_key):
        url = _seed()
        body = {"vector": V1}
        assert client.put(url, json=body).status_code == 401
        assert client.put(url, json=body, headers=WRONG).status_code == 401
        # el pedido rechazado no escribió nada
        np.testing.assert_allclose(client.get(url, headers=AUTH).json()["vector"], V0, atol=1e-7)
        r = client.put(url, json=body, headers=AUTH)
        assert r.status_code == 200, r.text
        np.testing.assert_allclose(
            client.get(url, headers=AUTH).json()["vector"], V1, atol=1e-6
        )

    def test_delete_without_key_is_401_and_keeps_vector(self, api_key):
        url = _seed()
        assert client.delete(url).status_code == 401
        assert client.delete(url, headers=WRONG).status_code == 401
        assert client.get(url, headers=AUTH).status_code == 200
        r = client.delete(url, headers=AUTH)
        assert r.status_code == 200, r.text
        assert r.json()["deleted"] == 1
        assert client.get(url, headers=AUTH).status_code == 404

    def test_auth_precedes_lookup(self, api_key):
        """Sin clave no se filtra si la colección existe: 401 antes que 404."""
        url = "/v1/collections/nope/vectors/x"
        assert client.get(url).status_code == 401
        assert client.put(url, json={"vector": V1}).status_code == 401
        assert client.delete(url).status_code == 401

    def test_open_without_configured_key(self):
        """Guardia: sin M2M_API_KEY la API sigue abierta (decisión documentada)."""
        url = _seed()
        assert client.get(url).status_code == 200
        assert client.put(url, json={"vector": V1}).status_code == 200
        assert client.delete(url).status_code == 200


# ---------------------------------------------------------------------------
# 2. Rate limit REAL y opt-in
# ---------------------------------------------------------------------------


class TestRateLimitOptIn:
    def test_off_by_default_never_counts_nor_rejects(self, monkeypatch):
        monkeypatch.delenv("M2M_RATE_LIMIT_PER_MIN", raising=False)
        for i in range(500):
            edge_api._check_rate_limit("ip-off", now=100.0 + i)
        assert edge_api._rate_limit_store["ip-off"] == []

    def test_counts_and_rejects_above_limit(self, monkeypatch):
        monkeypatch.setenv("M2M_RATE_LIMIT_PER_MIN", "3")
        for t in (100.0, 101.0, 102.0):
            edge_api._check_rate_limit("ip-a", now=t)
        assert edge_api._rate_limit_store["ip-a"] == [100.0, 101.0, 102.0]
        with pytest.raises(HTTPException) as ei:
            edge_api._check_rate_limit("ip-a", now=103.0)
        assert ei.value.status_code == 429
        # los pedidos rechazados no cuentan; ventana corrediza de 60 s
        with pytest.raises(HTTPException):
            edge_api._check_rate_limit("ip-a", now=159.0)
        edge_api._check_rate_limit("ip-a", now=160.5)
        assert edge_api._rate_limit_store["ip-a"] == [101.0, 102.0, 160.5]

    def test_limit_is_per_ip(self, monkeypatch):
        monkeypatch.setenv("M2M_RATE_LIMIT_PER_MIN", "2")
        edge_api._check_rate_limit("a", now=10.0)
        edge_api._check_rate_limit("a", now=10.5)
        with pytest.raises(HTTPException):
            edge_api._check_rate_limit("a", now=11.0)
        edge_api._check_rate_limit("b", now=11.0)  # otra IP: no afectada
        assert edge_api._rate_limit_store["b"] == [11.0]

    @pytest.mark.parametrize("raw", ["0", "-5", "abc", "  "])
    def test_zero_negative_or_invalid_means_off(self, monkeypatch, raw):
        monkeypatch.setenv("M2M_RATE_LIMIT_PER_MIN", raw)
        for i in range(50):
            edge_api._check_rate_limit(f"ip-{raw!r}", now=200.0 + i)
        assert edge_api._rate_limit_store[f"ip-{raw!r}"] == []

    def test_search_endpoint_returns_429(self, monkeypatch, api_key):
        _seed()
        # el límite se activa DESPUÉS de sembrar: sólo cuentan los pedidos de abajo
        monkeypatch.setenv("M2M_RATE_LIMIT_PER_MIN", "3")
        edge_api._rate_limit_store.clear()
        body = {"vector": V0, "k": 1}
        for _ in range(3):
            assert client.post("/v1/collections/c15/search", json=body, headers=AUTH).status_code == 200
        r = client.post("/v1/collections/c15/search", json=body, headers=AUTH)
        assert r.status_code == 429, r.text

    def test_put_and_delete_count_toward_the_limit(self, monkeypatch, api_key):
        url = _seed()
        monkeypatch.setenv("M2M_RATE_LIMIT_PER_MIN", "1")
        edge_api._rate_limit_store.clear()
        assert client.put(url, json={"vector": V1}, headers=AUTH).status_code == 200
        assert client.put(url, json={"vector": V1}, headers=AUTH).status_code == 429
        assert client.delete(url, headers=AUTH).status_code == 429

    def test_off_by_default_at_endpoint_level(self, monkeypatch, api_key):
        monkeypatch.delenv("M2M_RATE_LIMIT_PER_MIN", raising=False)
        url = _seed()
        body = {"vector": V0, "k": 1}
        for _ in range(150):  # más que el viejo tope nominal de 100/min
            assert client.post("/v1/collections/c15/search", json=body, headers=AUTH).status_code == 200
        assert client.put(url, json={"vector": V1}, headers=AUTH).status_code == 200


# ---------------------------------------------------------------------------
# 3. restore() valida la fuente antes de tocar el storage
# ---------------------------------------------------------------------------


def _tree(root):
    root = Path(root)
    skip = ("-journal", "-wal", "-shm")
    return sorted(
        str(q.relative_to(root)) for q in root.rglob("*") if not q.name.endswith(skip)
    )


def _db_with_docs(tmp_path, n=5):
    db = SimpleVectorDB(latent_dim=8, storage_path=str(tmp_path / "storage"))
    rng = np.random.default_rng(5)
    db.add(ids=[f"d{i}" for i in range(n)], vectors=_unit(rng.standard_normal((n, 8))))
    return db


class TestRestoreValidatesSource:
    def test_traversal_rejected_and_storage_untouched(self, tmp_path):
        db = _db_with_docs(tmp_path)
        p = db.storage
        before = _tree(p.storage_path)
        bk = Path(p.backup(str(tmp_path / "bk")))
        sneaky = bk.parent / ".." / bk.parent.name / bk.name
        with pytest.raises(ValueError):
            p.restore(str(sneaky))
        assert _tree(p.storage_path) == before
        assert p.get_stats()["total_documents"] == 5

    def test_missing_source_raises_before_destroying_storage(self, tmp_path):
        db = _db_with_docs(tmp_path)
        p = db.storage
        before = _tree(p.storage_path)
        with pytest.raises(FileNotFoundError):
            p.restore(str(tmp_path / "does-not-exist"))
        assert _tree(p.storage_path) == before
        assert p.get_stats()["total_documents"] == 5
        a_file = tmp_path / "afile"
        a_file.write_text("x")
        with pytest.raises(NotADirectoryError):
            p.restore(str(a_file))
        assert _tree(p.storage_path) == before

    def test_source_inside_storage_rejected(self, tmp_path):
        db = _db_with_docs(tmp_path)
        p = db.storage
        bk = Path(p.backup(str(tmp_path / "bk")))
        inside = Path(p.storage_path) / "old"
        shutil.move(str(bk), str(inside))
        before = _tree(p.storage_path)
        with pytest.raises(ValueError):
            p.restore(str(inside))
        assert _tree(p.storage_path) == before
        # un symlink que resuelve adentro del storage: mismo rechazo
        link = tmp_path / "link"
        link.symlink_to(str(p.storage_path), target_is_directory=True)
        with pytest.raises(ValueError):
            p.restore(str(link))
        assert _tree(p.storage_path) == before
        assert p.get_stats()["total_documents"] == 5

    def test_backup_of_storage_parent_rejected(self, tmp_path):
        db = _db_with_docs(tmp_path)
        p = db.storage
        before = _tree(p.storage_path)
        with pytest.raises(ValueError):
            p.restore(str(tmp_path))  # contiene al storage
        assert _tree(p.storage_path) == before

    def test_symlink_to_external_backup_is_allowed(self, tmp_path):
        db = _db_with_docs(tmp_path)
        p = db.storage
        bk = Path(p.backup(str(tmp_path / "bk")))
        link = tmp_path / "link"
        link.symlink_to(str(bk), target_is_directory=True)
        db.delete("d0", hard=True)
        assert p.get_stats()["total_documents"] == 4
        p.restore(str(link))
        assert p.get_stats()["total_documents"] == 5


# ---------------------------------------------------------------------------
# 4. docker-compose.yml raíz
# ---------------------------------------------------------------------------


def test_root_compose_does_not_publish_8080_on_all_interfaces():
    text = (ROOT / "docker-compose.yml").read_text()
    assert not re.search(r'^\s*-\s*"?8080:8080"?\s*$', text, re.M), (
        "8080 publicado en 0.0.0.0"
    )
    loopback_only = "127.0.0.1:8080:8080" in text
    key_required = re.search(r"\$\{M2M_API_KEY:\?", text) is not None
    assert loopback_only or key_required, "ni bind a loopback ni M2M_API_KEY exigida"


# ---------------------------------------------------------------------------
# 5. Coordinator: reenvía la clave a los edges y expone /health
# ---------------------------------------------------------------------------

QUERY = {"query": [0.1] * 8, "k": 5, "strategy": "broadcast"}


class _FakeResponse:
    status_code = 200

    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


class _FakeEdgeClient:
    """Edge de mentira: registra cada POST y devuelve un resultado fijo (sin red)."""

    def __init__(self):
        self.calls = []

    async def post(self, url, json=None, timeout=None, headers=None):
        self.calls.append(
            {"url": url, "json": json, "timeout": timeout, "headers": dict(headers or {})}
        )
        return _FakeResponse({"results": [{"doc_id": 7, "distance": 0.25}]})


@pytest.fixture
def stub_edge(monkeypatch):
    fake = _FakeEdgeClient()
    monkeypatch.setattr(coordinator_api, "http_client", fake)
    router = coordinator_api.router
    saved = dict(router.edge_nodes)
    router.edge_nodes.clear()
    try:
        router.register_edge("edge-p15", "http://127.0.0.1:9")
        yield fake
    finally:
        router.edge_nodes.clear()
        router.edge_nodes.update(saved)


class TestCoordinator:
    def test_health_without_key(self, monkeypatch):
        monkeypatch.setattr(coordinator_api, "_COORDINATOR_API_KEY", "coord-secret")
        r = coord_client.get("/health")
        assert r.status_code == 200, r.text
        assert r.json()["status"] == "ok"

    def test_search_forwards_edge_key(self, monkeypatch, stub_edge):
        monkeypatch.setenv("M2M_EDGE_API_KEY", "edge-secret")
        r = coord_client.post("/search", json=QUERY)
        assert r.status_code == 200, r.text
        assert [x["doc_id"] for x in r.json()["results"]] == [7]
        call = stub_edge.calls[0]
        assert call["url"] == "http://127.0.0.1:9/search"
        assert call["headers"].get("X-API-Key") == "edge-secret"

    def test_without_edge_key_no_credentials_are_sent(self, monkeypatch, stub_edge):
        monkeypatch.delenv("M2M_EDGE_API_KEY", raising=False)
        monkeypatch.setattr(coordinator_api, "_COORDINATOR_API_KEY", "coord-secret")
        r = coord_client.post("/search", json=QUERY, headers={"Authorization": "Bearer coord-secret"})
        assert r.status_code == 200, r.text
        sent = stub_edge.calls[0]["headers"]
        assert "X-API-Key" not in sent and "Authorization" not in sent
        assert "coord-secret" not in json.dumps(stub_edge.calls)

    def test_empty_edge_key_means_no_header(self, monkeypatch, stub_edge):
        monkeypatch.setenv("M2M_EDGE_API_KEY", "")
        assert coord_client.post("/search", json=QUERY).status_code == 200
        assert stub_edge.calls[0]["headers"] == {}


# ---------------------------------------------------------------------------
# 6. Guardia: hashes de bucketing/etiquetas sin cambio de valor
# ---------------------------------------------------------------------------


def test_bucketing_hashes_keep_their_digest():
    from m2m.cluster.sharding import shard_by_hash
    from m2m.gate_modulus import _data_fingerprint

    for doc_id in ["a", "doc-1", "ünïcode-_clave"]:
        expected = f"edge-{int(hashlib.md5(doc_id.encode('utf-8')).hexdigest(), 16) % 4}"
        assert shard_by_hash(doc_id, 4) == expected
    X = np.ascontiguousarray(np.arange(12, dtype=np.float32).reshape(3, 4))
    h = hashlib.sha1()
    h.update(b"3x4 float32;")
    h.update(X.tobytes())
    assert _data_fingerprint(X) == h.hexdigest()


# ---------------------------------------------------------------------------
# 7. backend en get_stats, dispositivos en M2MConfig, fallo CUDA avisado una vez
# ---------------------------------------------------------------------------


def test_get_stats_exposes_backend():
    db = SimpleVectorDB(latent_dim=8)
    rng = np.random.default_rng(1)
    db.add(ids=[f"d{i}" for i in range(6)], vectors=_unit(rng.standard_normal((6, 8))))
    st = db.get_stats()
    info = db.engine.get_backend_info()
    assert st["backend"] == db.engine.backend == info["effective"]
    assert st["requested_device"] == info["requested"]
    assert st["backend"] in ("cpu", "cuda", "vulkan")


def test_config_names_requested_device_and_reports_cuda_availability(monkeypatch):
    import m2m.engine as engine_mod

    cfg = M2MConfig(device="cuda")
    # Nombres públicos conservados: son lo SOLICITADO (flags), no lo efectivo.
    assert cfg.compute_device == "cuda"
    assert cfg.effective_device == "cuda"
    assert cfg.requested_device == "cuda"
    assert "SOLICITADO" in (M2MConfig.compute_device.__doc__ + M2MConfig.effective_device.__doc__)
    monkeypatch.setattr(engine_mod, "_cuda_unavailable_reason", lambda: None)
    assert cfg.cuda_unavailable_reason() is None
    monkeypatch.setattr(engine_mod, "_cuda_unavailable_reason", lambda: "sin GPU")
    assert cfg.cuda_unavailable_reason() == "sin GPU"
    assert M2MConfig().requested_device == "cpu"


def test_cuda_failure_is_warned_once_and_falls_back(monkeypatch, caplog):
    import m2m.cuda_search as cuda_mod

    class _Boom:
        def __init__(self, *a, **k):
            raise RuntimeError("cuda boom")

    monkeypatch.setattr(cuda_mod, "CUDASearcher", _Boom)
    rng = np.random.default_rng(2)
    x = _unit(rng.standard_normal((12, 8)))

    def build(device):
        store = SplatStore(M2MConfig(latent_dim=8, max_splats=64, device=device))
        store.add_splat(x)
        return store

    ref, bad = build("cpu"), build("cuda")
    q = _unit(x[:3] + 0.05 * rng.standard_normal((3, 8)))
    with caplog.at_level(logging.WARNING, logger="m2m.splats"):
        for _ in range(4):
            got = bad.batch_find_neighbors(q, k=3)
        want = ref.batch_find_neighbors(q, k=3)
    for g, w in zip(got, want):
        np.testing.assert_allclose(g, w, atol=1e-5)
    cuda_warnings = [
        r
        for r in caplog.records
        if r.name == "m2m.splats" and r.levelno == logging.WARNING and "CUDA" in r.getMessage()
    ]
    assert len(cuda_warnings) == 1, [r.getMessage() for r in cuda_warnings]
    assert "cuda boom" in cuda_warnings[0].getMessage()


# ---------------------------------------------------------------------------
# 8. _doc_results_from_lsh + docstring de filter_by_metadata
# ---------------------------------------------------------------------------


def _lsh_db(n=80, d=16):
    rng = np.random.default_rng(3)
    x = _unit(rng.standard_normal((n, d)))  # uniforme: silhouette < umbral -> modo LSH
    db = SimpleVectorDB(latent_dim=d, enable_lsh_fallback=True)
    ids = [f"d{i}" for i in range(n)]
    meta = [{"cat": "a" if i % 2 == 0 else "b"} for i in range(n)]
    db.add(ids=ids, vectors=x, metadata=meta)
    assert db._use_lsh, "el test requiere modo LSH"
    return db, ids, x


class TestLshFilter:
    def test_filter_is_evaluated_without_metadata(self):
        db, ids, x = _lsh_db()
        n = len(ids)
        res = db._doc_results_from_lsh(
            np.arange(n), np.zeros(n), k=n, include_metadata=False, filter={"cat": "a"}, query=x[0]
        )
        assert {r.id for r in res} == {ids[i] for i in range(0, n, 2)}
        assert all(r.metadata == {} for r in res)  # include_metadata=False se respeta

    def test_filter_matches_with_and_without_metadata(self):
        db, ids, x = _lsh_db(n=40)
        n = len(ids)
        kw = dict(indices=np.arange(n), distances=np.zeros(n), k=n, query=x[1])
        with_meta = db._doc_results_from_lsh(filter={"cat": "b"}, include_metadata=True, **kw)
        without = db._doc_results_from_lsh(filter={"cat": "b"}, include_metadata=False, **kw)
        assert [r.id for r in with_meta] == [r.id for r in without] == ids[1::2]

    def test_no_filter_keeps_everything(self):
        db, ids, x = _lsh_db(n=40)
        res = db._doc_results_from_lsh(
            np.arange(40), np.zeros(40), k=40, include_metadata=False, query=x[0]
        )
        assert {r.id for r in res} == set(ids)


def test_filter_by_metadata_docstring_is_truthful(tmp_path):
    doc = M2MPersistence.filter_by_metadata.__doc__
    for token in ("$eq", "$ne", "$in", "$nin", "$gt", "$gte", "$lt", "$lte", "callable", "ValueError"):
        assert token in doc, token


# ---------------------------------------------------------------------------
# 9. SQ8DiskStore.search_batch: queries finitas
# ---------------------------------------------------------------------------


def _sq8(tmp_path, n=40, d=8):
    st = SQ8DiskStore(dim=d, dump_dir=str(tmp_path))
    rng = np.random.default_rng(4)
    st.add(rng.standard_normal((n, d)).astype(np.float32))
    st.flush(str(tmp_path))
    return st, rng


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_sq8_search_batch_rejects_non_finite_queries(tmp_path, bad):
    st, rng = _sq8(tmp_path)
    q = rng.standard_normal((3, 8)).astype(np.float32)
    q[1, 4] = bad
    with pytest.raises(ValueError, match="NaN|Inf"):
        st.search_batch(q, k=3)
    with pytest.raises(ValueError):
        st.search_batch(np.full(8, bad, dtype=np.float32), k=3)
    ok = rng.standard_normal((3, 8)).astype(np.float32)
    out = st.search_batch(ok, k=3)
    assert out.shape == (3, 3) and (out >= 0).all()


# ---------------------------------------------------------------------------
# 10. Docstrings / mensajes veraces
# ---------------------------------------------------------------------------


def test_main_suggests_the_api_extra(monkeypatch):
    import m2m.__main__ as main_mod

    monkeypatch.setitem(sys.modules, "uvicorn", None)  # import uvicorn -> ImportError
    with pytest.raises(SystemExit) as ei:
        main_mod.main()
    msg = str(ei.value)
    assert "m2m-vector-search[api]" in msg and "[all]" not in msg, msg


def test_langchain_docstring_says_vectorstore():
    pytest.importorskip("langchain_core")
    from langchain_core.vectorstores import VectorStore

    from m2m.integrations.langchain import M2MVectorStore

    assert issubclass(M2MVectorStore, VectorStore)
    doc = M2MVectorStore.__doc__
    assert "VectorStore" in doc and "BaseRetriever" not in doc


# ---------------------------------------------------------------------------
# 12. Estado aprendido: export, reopen en SimpleVectorDB, writers
# ---------------------------------------------------------------------------


def test_feedback_result_is_exported():
    assert "FeedbackResult" in m2m.__all__
    assert m2m.FeedbackResult.__name__ == "FeedbackResult"


def _learned_db(path, n=60, d=16, iters=120, seed=11):
    rng = np.random.default_rng(seed)
    x = _unit(rng.standard_normal((n, d)))
    ids = [f"d{i}" for i in range(n)]
    db = SimpleVectorDB(latent_dim=d, storage_path=path)
    db.add(ids=ids, vectors=x)
    q = _unit(rng.standard_normal(d))
    target = ids[int(np.argsort(x @ q)[-4])]  # rank 3: tiene margen para mejorar
    for _ in range(iters):
        db.feedback(q, relevant_ids=[target])
    return db, ids, q, target, d


def _learned_alpha(db, doc_id):
    st = db.engine.m2m.splats
    return float(st.alpha[db._splat_id_order.index(doc_id)])


class TestLearnedStateSurvivesReopen:
    def test_simple_db_restores_learned_state(self, tmp_path):
        path = str(tmp_path / "s")
        db, ids, q, target, d = _learned_db(path)
        assert _learned_alpha(db, target) > 1.0  # algo se aprendió
        rank_learned = [r.id for r in db.search(q, k=10, include_metadata=True)].index(target)
        assert rank_learned < 3
        p0 = db._splat_id_order.index(target)
        mu0 = db.engine.m2m.splats.mu[p0].copy()
        k0 = float(db.engine.m2m.splats.kappa[p0])
        db.storage.close()

        db2 = SimpleVectorDB(latent_dim=d, storage_path=path)
        rank2 = [r.id for r in db2.search(q, k=10, include_metadata=True)].index(target)
        assert rank2 == rank_learned
        p2 = db2._splat_id_order.index(target)
        st2 = db2.engine.m2m.splats
        assert st2.alpha[p2] == pytest.approx(_learned_alpha(db, target), rel=1e-5)
        assert st2.kappa[p2] == pytest.approx(k0, rel=1e-5)
        np.testing.assert_allclose(st2.mu[p2], mu0, atol=1e-5)

    def test_advanced_db_still_restores(self, tmp_path):
        """Guardia: AdvancedVectorDB ya lo hacía; no se rompe al unificar."""
        rng = np.random.default_rng(12)
        x = _unit(rng.standard_normal((40, 16)))
        ids = [f"d{i}" for i in range(40)]
        path = str(tmp_path / "adv")
        db = AdvancedVectorDB(latent_dim=16, storage_path=path)
        db.add(ids=ids, vectors=x)
        q = _unit(rng.standard_normal(16))
        for _ in range(120):
            db.feedback(q, relevant_ids=["d5"])
        a = _learned_alpha(db, "d5")
        assert a > 1.0
        db.storage.close()
        db2 = AdvancedVectorDB(latent_dim=16, storage_path=path)
        assert _learned_alpha(db2, "d5") == pytest.approx(a, rel=1e-5)

    def test_hard_delete_then_readd_does_not_resurrect_learned_state(self, tmp_path):
        path = str(tmp_path / "s")
        db, ids, q, target, d = _learned_db(path, iters=20)
        x = np.array(db._vectors[target], dtype=np.float32, copy=True)
        db.delete(id=target, hard=True)
        db.add(ids=[target], vectors=x[None])
        assert _learned_alpha(db, target) == pytest.approx(1.0)  # en memoria: default
        db.storage.close()
        db2 = SimpleVectorDB(latent_dim=d, storage_path=path)
        assert _learned_alpha(db2, target) == pytest.approx(1.0)  # ni tras reopen
        p2 = db2._splat_id_order.index(target)
        np.testing.assert_allclose(db2.engine.m2m.splats.mu[p2], _unit(x), atol=1e-5)

    def test_update_vector_then_revert_does_not_resurrect_learned_state(self, tmp_path):
        path = str(tmp_path / "s")
        db, ids, q, target, d = _learned_db(path, iters=20)
        x0 = np.array(db._vectors[target], dtype=np.float32, copy=True)
        rng = np.random.default_rng(13)
        db.update(id=target, vector=_unit(rng.standard_normal(d)))
        db.update(id=target, vector=x0)
        assert _learned_alpha(db, target) == pytest.approx(1.0)
        db.storage.close()
        db2 = SimpleVectorDB(latent_dim=d, storage_path=path)
        assert _learned_alpha(db2, target) == pytest.approx(1.0)

    def test_update_keeps_learned_state_of_other_docs(self, tmp_path):
        path = str(tmp_path / "s")
        db, ids, q, target, d = _learned_db(path, iters=20)
        other = ids[1]
        db.update(id=other, metadata={"tag": "x"})  # sólo metadata: no toca splats
        a = _learned_alpha(db, target)
        db.storage.close()
        db2 = SimpleVectorDB(latent_dim=d, storage_path=path)
        assert _learned_alpha(db2, target) == pytest.approx(a, rel=1e-5)


# ---------------------------------------------------------------------------
# 13. Sin wrapper de lock sobre engine.search
# ---------------------------------------------------------------------------


def test_engine_search_is_not_monkey_patched():
    """El único caller de ``engine.search`` es ``SimpleVectorDB.search()``, que
    toma ``_rw_lock`` en todo el cuerpo (P3/P6); el wrapper era redundante. La
    concurrencia la cubren tests/test_audit_write_lifecycle.py y
    tests/test_audit_search_filters.py."""
    db = SimpleVectorDB(latent_dim=8)
    rng = np.random.default_rng(6)
    db.add(ids=[f"d{i}" for i in range(4)], vectors=_unit(rng.standard_normal((4, 8))))
    assert "search" not in vars(db.engine)
    res = db.search(_unit(rng.standard_normal(8)), k=2)
    assert len(res) == 2
