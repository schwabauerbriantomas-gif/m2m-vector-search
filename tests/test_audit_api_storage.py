"""
P14 — API y storage: seguridad básica + escritura por lotes (audit r3a-17 resto).

Tests de regresión; todos los marcados RED fallaban antes del fix
(``reports/p14_red.log``):

  1. API key: ``hmac.compare_digest`` + header ``Authorization: Bearer`` normalizado;
     ``/v1/admin/*`` fail-closed sin key configurada; ``GET /suggest`` valida la key
     sin romper (antes: 500 siempre que hubiera key configurada).
  2. Body > ``M2M_MAX_BODY_BYTES`` → 413 sin leerlo; conteo de vectores ANTES de
     normalizar/validar (100.001 vectores → error en < 2 s).
  3. Rate limit: memoria acotada (purga de entradas viejas + tope de IPs).
  4. ``/v1/admin/checkpoint`` y ``/v1/admin/backup`` no se tragan excepciones.
  5. ``save_metadata``: UNA transacción por ``add()`` + durabilidad al volver.
  6. ``SplatStore.reset()``: el store queda como recién construido.
  7. Confinamiento de rutas (``..``, absolutas, symlinks) en índice y shards.

Decisión documentada (ítem 1): sin ``M2M_API_KEY`` la API sigue ABIERTA para los
endpoints normales (desarrollo local / Quick Start); sólo ``/v1/admin/*`` exige key.
"""

import hashlib
import hmac
import json
import os
import pickle
import sqlite3
import subprocess
import sys
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

import m2m
from m2m import M2MConfig, SplatStore
from m2m.api import edge_api
from m2m.api.edge_api import _manager
from m2m.api.edge_api import app as edge_app
from m2m.storage.persistence import M2MPersistence

client = TestClient(edge_app)

DIM = 32
MAX_BODY = 32 * 1024 * 1024
MAX_VECTORS = 100_000
KEY = "p14-secret-key"
AUTH = {"Authorization": f"Bearer {KEY}"}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _cleanup():
    _manager._collections.clear()
    _manager._dbs.clear()


@pytest.fixture(autouse=True)
def _isolated(monkeypatch):
    _cleanup()
    # Estado de auth determinístico por test: sin key salvo que el test la configure.
    monkeypatch.setattr(edge_api, "_API_KEY", None)
    monkeypatch.delenv("M2M_API_KEY", raising=False)
    monkeypatch.delenv("M2M_MAX_BODY_BYTES", raising=False)
    edge_api._rate_limit_store.clear()
    yield
    _cleanup()
    edge_api._rate_limit_store.clear()


@pytest.fixture
def api_key(monkeypatch):
    monkeypatch.setattr(edge_api, "_API_KEY", KEY)
    return KEY


def _create(name="c", **extra):
    # AUTH es inofensivo sin key configurada y necesario en los tests que la configuran.
    r = client.post("/v1/collections", json={"name": name, "dimension": DIM, **extra}, headers=AUTH)
    assert r.status_code == 201, r.text
    return name


class _BoomStorage:
    def __init__(self, exc):
        self._exc = exc

    def backup(self, path):
        raise self._exc


class _BoomDB:
    """DB de mentira: tiene storage pero toda escritura falla."""

    def __init__(self, exc):
        self._exc = exc
        self.storage = _BoomStorage(exc)

    def save(self, *args, **kwargs):
        raise self._exc

    def get_stats(self):
        raise self._exc


def _install_boom(name, exc):
    _create(name)
    _manager._dbs[name] = _BoomDB(exc)


# ---------------------------------------------------------------------------
# 1. API key
# ---------------------------------------------------------------------------


class TestApiKey:
    def test_comparison_goes_through_compare_digest(self, monkeypatch, api_key):
        calls = []
        real = hmac.compare_digest

        def spy(a, b):
            calls.append((a, b))
            return real(a, b)

        monkeypatch.setattr(hmac, "compare_digest", spy)
        r = client.get("/v1/stats", headers=AUTH)
        assert r.status_code == 200
        assert len(calls) == 1, "la key se comparó sin hmac.compare_digest"
        a, b = calls[0]
        norm = lambda v: v.decode() if isinstance(v, bytes) else v  # noqa: E731
        assert (norm(a), norm(b)) == (KEY, KEY)

    def test_wrong_key_is_401(self, api_key):
        assert client.get("/v1/stats", headers={"X-API-Key": "nope"}).status_code == 401
        assert client.get("/v1/stats", headers={"Authorization": "Bearer nope"}).status_code == 401
        assert client.get("/v1/stats").status_code == 401

    def test_bearer_scheme_is_normalized(self, api_key):
        """Scheme case-insensitive y espacios tolerados (RFC 7235)."""
        assert client.get("/v1/stats", headers={"Authorization": f"bearer {KEY}"}).status_code == 200
        assert client.get("/v1/stats", headers={"Authorization": f"Bearer   {KEY} "}).status_code == 200

    def test_x_api_key_header_still_accepted(self, api_key):
        assert client.get("/v1/stats", headers={"X-API-Key": KEY}).status_code == 200

    def test_other_authorization_scheme_is_not_a_key(self, api_key):
        assert client.get("/v1/stats", headers={"Authorization": "Basic abc"}).status_code == 401

    def test_without_key_normal_endpoints_stay_open(self):
        """Decisión: sin M2M_API_KEY la API es abierta (desarrollo local)."""
        assert client.get("/v1/stats").status_code == 200
        assert client.post("/v1/collections", json={"name": "open", "dimension": DIM}).status_code == 201

    def test_admin_fail_closed_without_key(self):
        """Sin key configurada, /v1/admin/* se niega en vez de quedar abierto."""
        _create("mem")
        r = client.post("/v1/admin/checkpoint")
        assert r.status_code == 503, r.text
        assert "M2M_API_KEY" in r.text
        r = client.post("/v1/admin/backup", json={"backup_path": "/tmp/x"})
        assert r.status_code == 503, r.text

    def test_admin_rejects_wrong_key(self, api_key):
        assert client.post("/v1/admin/checkpoint", headers={"X-API-Key": "nope"}).status_code == 401
        assert client.post("/v1/admin/checkpoint").status_code == 401

    def test_get_suggest_validates_key_instead_of_crashing(self, api_key):
        """Antes: explore(request=None) → AttributeError → 500 siempre que hubiera key."""
        _create("s")
        assert client.get("/v1/collections/s/suggest").status_code == 401
        r = client.get("/v1/collections/s/suggest", headers=AUTH)
        assert r.status_code == 400  # EBM no habilitado: respuesta real, no 500
        assert "EBM" in r.text


# ---------------------------------------------------------------------------
# 2. Límite de body antes de parsear + conteo de vectores antes de validar
# ---------------------------------------------------------------------------


class TestBodyLimit:
    def test_oversized_body_is_413_without_parsing(self):
        _create("big")
        payload = b"x" * (MAX_BODY + 1)  # no es JSON: si se parseara daría 400/422
        t0 = time.perf_counter()
        r = client.post(
            "/v1/collections/big/vectors",
            content=payload,
            headers={"Content-Type": "application/json"},
        )
        dt = time.perf_counter() - t0
        assert r.status_code == 413, r.text
        assert dt < 2.0, f"413 tardó {dt:.2f}s: el body se leyó/parseó"

    def test_body_under_limit_is_parsed_normally(self):
        _create("ok")
        r = client.post(
            "/v1/collections/ok/vectors",
            content=b"{" + b" " * 1000 + b"}",
            headers={"Content-Type": "application/json"},
        )
        assert r.status_code == 422  # falta 'vectors': llegó al validador
        assert r.status_code != 413

    def test_limit_is_configurable_via_env(self, monkeypatch):
        monkeypatch.setenv("M2M_MAX_BODY_BYTES", "1024")
        assert edge_api._max_body_bytes() == 1024
        # Valor inválido → default, sin romper el arranque.
        monkeypatch.setenv("M2M_MAX_BODY_BYTES", "no-es-un-numero")
        assert edge_api._max_body_bytes() == MAX_BODY
        monkeypatch.setenv("M2M_MAX_BODY_BYTES", "-5")
        assert edge_api._max_body_bytes() == MAX_BODY
        monkeypatch.delenv("M2M_MAX_BODY_BYTES")
        # El tope se fija al construir la app (como el resto de la config de
        # FastAPI): la app de este proceso lleva el default; un import fresco
        # con el env seteado levanta el valor configurado.
        assert edge_app.max_body_size == MAX_BODY
        src_root = os.path.dirname(os.path.dirname(os.path.abspath(m2m.__file__)))
        env = {**os.environ, "M2M_MAX_BODY_BYTES": "1024", "PYTHONPATH": src_root}
        out = subprocess.run(
            [sys.executable, "-c", "import m2m.api.edge_api as e; print(e.app.max_body_size)"],
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert out.returncode == 0, out.stderr
        assert out.stdout.strip() == "1024"

    def test_chunked_body_without_content_length_is_cut(self):
        """Sin Content-Length (chunked) el límite se aplica mientras se lee."""
        _create("chunked")
        chunk = b"x" * (1024 * 1024)
        r = client.post(
            "/v1/collections/chunked/vectors",
            content=(chunk for _ in range(MAX_BODY // len(chunk) + 2)),
            headers={"Content-Type": "application/json"},
        )
        assert r.status_code == 413, r.text

    def test_too_many_vectors_rejected_before_validating_arrays(self):
        """100.001 vectores → 422 en < 2 s (antes: 6-9 s de validación pydantic)."""
        _create("many")
        # JSON pre-codificado: el timed section mide sólo el request, no el
        # json.dumps del cliente.
        body = json.dumps({"vectors": [[0.0] * DIM] * (MAX_VECTORS + 1)}).encode()
        headers = {"Content-Type": "application/json"}
        t0 = time.perf_counter()
        r = client.post("/v1/collections/many/vectors", content=body, headers=headers)
        dt = time.perf_counter() - t0
        assert r.status_code == 422, r.text
        assert "100000" in r.text
        assert dt < 2.0, f"el rechazo tardó {dt:.2f}s: los arrays se validaron antes del conteo"
        # Señal robusta a la carga del host: cuesta lo mismo que /ingest, que
        # sólo decodifica el JSON y cuenta (sin validar los arrays).
        t0 = time.perf_counter()
        client.post("/ingest", content=body, headers=headers)
        dt_ingest = time.perf_counter() - t0
        assert dt <= 3 * dt_ingest + 0.5, (dt, dt_ingest)

    def test_vectors_at_limit_are_accepted(self):
        _create("exact")
        n = 50
        r = client.post("/v1/collections/exact/vectors", json={"vectors": [[1.0] * DIM] * n})
        assert r.status_code == 200 and r.json()["added"] == n


# ---------------------------------------------------------------------------
# 3. Rate limit con memoria acotada
# ---------------------------------------------------------------------------


class TestRateLimitMemory:
    def test_defaults_unchanged(self):
        assert edge_api._RATE_LIMIT_WINDOW == 60.0
        assert edge_api._RATE_LIMIT_MAX_REQUESTS == 100
        assert edge_api._RATE_LIMIT_MAX_VECTORS_PER_INSERT == 100_000

    def test_tracked_clients_are_capped(self, monkeypatch):
        monkeypatch.setattr(edge_api, "_RATE_LIMIT_MAX_CLIENTS", 100)
        monkeypatch.setattr(edge_api, "_RATE_LIMIT_SWEEP_INTERVAL", 0.0)
        monkeypatch.setattr(edge_api, "_rate_limit_last_sweep", 0.0)
        store = edge_api._rate_limit_store
        store.clear()
        for i in range(150):
            store[f"ip{i:03d}"] = [290.0 + i / 1000]  # pedidos recientes: el barrido por ventana no los purga
        edge_api._check_rate_limit("new", now=300.0)
        assert len(store) <= 100
        # El más reciente sigue trackeado; el de actividad más vieja se evictó.
        assert "new" in store and "ip149" in store
        assert "ip000" not in store

    def test_idle_entries_are_purged(self, monkeypatch):
        monkeypatch.setattr(edge_api, "_RATE_LIMIT_SWEEP_INTERVAL", 0.0)
        monkeypatch.setattr(edge_api, "_rate_limit_last_sweep", 0.0)
        store = edge_api._rate_limit_store
        store.clear()
        store["recent"] = [100.0]
        store["idle"] = [10.0]  # último pedido hace más de una ventana
        store["empty"] = []
        edge_api._check_rate_limit("recent", now=120.0)
        assert set(store) == {"recent"}
        assert store["recent"] == [100.0]

    def test_sweep_is_amortized(self, monkeypatch):
        """El barrido completo no corre en cada request (costo O(IPs))."""
        monkeypatch.setattr(edge_api, "_RATE_LIMIT_SWEEP_INTERVAL", 3600.0)
        monkeypatch.setattr(edge_api, "_rate_limit_last_sweep", 0.0)
        edge_api._rate_limit_store.clear()
        for i in range(1000):
            edge_api._check_rate_limit(f"ip{i}", now=5.0)
        assert edge_api._rate_limit_last_sweep == 0.0


# ---------------------------------------------------------------------------
# 4. Endpoints que persisten no se tragan excepciones
# ---------------------------------------------------------------------------


class TestAdminNoSwallow:
    def test_checkpoint_in_memory_collection_named_is_400(self, api_key):
        _create("mem")
        r = client.post("/v1/admin/checkpoint?name=mem", headers=AUTH)
        assert r.status_code == 400, r.text
        assert "storage_path" in r.text

    def test_checkpoint_all_collections_reports_in_memory_as_skipped(self, api_key):
        _create("mem")
        r = client.post("/v1/admin/checkpoint", headers=AUTH)
        assert r.status_code == 200, r.text
        assert r.json()["results"]["mem"]["status"] == "skipped"

    def test_checkpoint_unknown_collection_is_404(self, api_key):
        r = client.post("/v1/admin/checkpoint?name=nope", headers=AUTH)
        assert r.status_code == 404, r.text

    def test_checkpoint_real_failure_is_500_with_message(self, api_key):
        _install_boom("boom", OSError("disk full"))
        r = client.post("/v1/admin/checkpoint?name=boom", headers=AUTH)
        assert r.status_code == 500, r.text
        assert "disk full" in r.text
        r = client.post("/v1/admin/checkpoint", headers=AUTH)
        assert r.status_code == 500
        assert r.json()["results"]["boom"]["status"] == "error"

    def test_checkpoint_persistent_collection_ok(self, api_key, tmp_path):
        _create("p", storage_path=str(tmp_path / "st"))
        client.post("/v1/collections/p/vectors", json={"vectors": [[1.0] * DIM] * 3}, headers=AUTH)
        r = client.post("/v1/admin/checkpoint?name=p", headers=AUTH)
        assert r.status_code == 200, r.text
        assert r.json()["results"]["p"]["status"] == "ok"

    def test_backup_failure_is_500_with_reason(self, api_key, tmp_path):
        _install_boom("boom", OSError("disk full"))
        _create("mem")
        r = client.post("/v1/admin/backup", json={"backup_path": str(tmp_path / "bk")}, headers=AUTH)
        assert r.status_code == 500, r.text
        body = r.json()
        assert body["results"]["boom"] == {"status": "error", "reason": "disk full"}
        assert body["results"]["mem"]["status"] == "skipped"

    def test_stats_reports_collection_error(self):
        _install_boom("boom", RuntimeError("stats boom"))
        r = client.get("/v1/stats")
        assert r.status_code == 200
        entries = {c.get("name"): c for c in r.json()["collections"]}
        assert entries["boom"].get("error") == "stats boom"


# ---------------------------------------------------------------------------
# 5. save_metadata: una transacción por add()
# ---------------------------------------------------------------------------


def _count_commits(monkeypatch):
    commits = {"n": 0}
    real_connect = sqlite3.connect

    class _Conn:
        def __init__(self, c):
            self._c = c

        def commit(self):
            commits["n"] += 1
            return self._c.commit()

        def __getattr__(self, name):
            return getattr(self._c, name)

    monkeypatch.setattr(sqlite3, "connect", lambda *a, **k: _Conn(real_connect(*a, **k)))
    return commits


def _vecs(n, seed=0):
    return np.random.default_rng(seed).standard_normal((n, DIM), dtype=np.float32)


class TestMetadataBatch:
    def test_add_is_one_transaction(self, tmp_path, monkeypatch):
        db = m2m.SimpleVectorDB(latent_dim=DIM, storage_path=str(tmp_path / "st"), enable_wal=False)
        commits = _count_commits(monkeypatch)
        n = 1000
        added = db.add(vectors=_vecs(n))
        assert added == n
        assert commits["n"] == 1, f"{commits['n']} commits para un add() de {n} docs (esperado: 1)"

    def test_add_is_visible_to_a_new_instance_and_other_connection(self, tmp_path):
        st = str(tmp_path / "st")
        db = m2m.SimpleVectorDB(latent_dim=DIM, storage_path=st, enable_wal=False)
        ids = [f"d{i}" for i in range(50)]
        db.add(ids=ids, vectors=_vecs(50))
        # Otra conexión SQLite (no pasa por el objeto persistence) ve el lote entero.
        conn = sqlite3.connect(tmp_path / "st" / "data" / "metadata" / "metadata.db")
        try:
            n_rows = conn.execute("SELECT COUNT(*) FROM documents WHERE deleted=0").fetchone()[0]
        finally:
            conn.close()
        assert n_rows == 50
        fresh = m2m.SimpleVectorDB(latent_dim=DIM, storage_path=st, enable_wal=False)
        assert set(fresh._vectors) == set(ids)

    def test_save_metadata_many_is_one_atomic_transaction(self, tmp_path, monkeypatch):
        p = M2MPersistence(str(tmp_path / "st"), enable_wal=False)
        commits = _count_commits(monkeypatch)
        rows = [{"id": f"d{i}", "shard_idx": 1, "vector_idx": i, "metadata": {"i": i}, "document": None} for i in range(200)]
        p.save_metadata_many(rows)
        assert commits["n"] == 1
        docs = p.get_all_documents(include_deleted=False)
        assert [d["id"] for d in docs] == [r["id"] for r in rows]
        assert docs[7]["metadata"] == {"i": 7}

    def test_standalone_save_metadata_still_commits_immediately(self, tmp_path):
        st = tmp_path / "st"
        p = M2MPersistence(str(st), enable_wal=False)
        p.save_metadata("alone", shard_idx=1, vector_idx=0, metadata={"k": 1})
        conn = sqlite3.connect(st / "data" / "metadata" / "metadata.db")
        try:
            row = conn.execute("SELECT metadata FROM documents WHERE id='alone'").fetchone()
        finally:
            conn.close()
        assert row is not None and row[0] == '{"k": 1}'

    def test_incomplete_batch_is_flushed_by_the_next_storage_operation(self, tmp_path):
        st = tmp_path / "st"
        p = M2MPersistence(str(st), enable_wal=False)
        p.save_vectors(np.zeros((4, 3), dtype=np.float32), ["a", "b", "c", "d"])
        p.save_metadata("a", shard_idx=1, vector_idx=0)  # 1 de 4 filas: lote incompleto
        assert p.get_all_ids() == ["a"]
        conn = sqlite3.connect(st / "data" / "metadata" / "metadata.db")
        try:
            assert conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0] == 1
        finally:
            conn.close()


# ---------------------------------------------------------------------------
# 6. SplatStore.reset()
# ---------------------------------------------------------------------------


def _store(max_splats=64, dim=8):
    return SplatStore(M2MConfig(latent_dim=dim, max_splats=max_splats))


class TestSplatReset:
    def test_reset_matches_fresh_store_and_keeps_engine(self):
        store = _store()
        engine = store.engine
        fresh = _store()
        X = np.random.default_rng(1).standard_normal((20, 8), dtype=np.float32)
        assert store.add_splat(X) == 20
        store.build_index()
        store._m_sq_cache = np.ones(20, dtype=np.float32)
        store._m_sq_dirty = False
        assert store.n_active == 20 and store._next_id == 20

        store.reset()

        assert store.engine is engine  # sin reinstanciar
        assert store.n_active == fresh.n_active == 0
        assert store._next_id == 0
        assert getattr(store, "_last_trained_n", 0) == 0
        for attr in ("mu", "alpha", "kappa", "frequency"):
            np.testing.assert_array_equal(getattr(store, attr), getattr(fresh, attr))
        assert store._m_sq_cache is None
        assert store.engine._is_indexed is False
        assert store.engine.embeddings is None
        assert store.engine.coarse_assignments is None

        # Queda usable: ids desde 0 y mismo resultado que un store nuevo.
        assert store.add_splat(X[:5]) == 5
        assert store.n_active == 5 and store._next_id == 5
        np.testing.assert_array_equal(store.mu[:5], X[:5])
        store.build_index()
        assert store.engine._is_indexed is True

    def test_reset_returns_none(self):
        assert _store().reset() is None


# ---------------------------------------------------------------------------
# 7. Confinamiento de rutas en save/load de índice y shards
# ---------------------------------------------------------------------------


@pytest.fixture
def signed(monkeypatch):
    monkeypatch.setenv("M2M_HMAC_SECRET", "p14-hmac-secret")


def _storage(tmp_path):
    root = tmp_path / "root"
    return root, M2MPersistence(str(root / "st"), enable_wal=False)


def _signed_bytes(data, secret="p14-hmac-secret"):
    payload = pickle.dumps(data)
    return hmac.new(secret.encode(), payload, hashlib.sha256).digest() + payload


class TestPathConfinement:
    def test_index_name_with_dotdot_is_rejected(self, tmp_path, signed):
        root, p = _storage(tmp_path)
        with pytest.raises(ValueError):
            p.save_index({"a": 1}, name="../../escape")
        assert not (root / "escape.idx").exists()
        with pytest.raises(ValueError):
            p.save_index({"a": 1}, name="hrm2/../../escape")
        assert not (root / "escape.idx").exists()

    def test_load_index_with_dotdot_is_rejected(self, tmp_path, signed):
        _, p = _storage(tmp_path)
        with pytest.raises(ValueError):
            p.load_index("../../etc/secret")

    def test_absolute_index_name_is_rejected(self, tmp_path, signed):
        _, p = _storage(tmp_path)
        target = tmp_path / "abs.idx"
        with pytest.raises(ValueError):
            p.save_index({"a": 1}, name=str(target)[:-4])
        assert not target.exists()

    def test_index_dir_symlink_escape_is_rejected(self, tmp_path, signed):
        root, p = _storage(tmp_path)
        outside = tmp_path / "outside"
        outside.mkdir()
        idx = p.storage_path / "index"
        idx.rmdir()
        idx.symlink_to(outside, target_is_directory=True)
        with pytest.raises(ValueError):
            p.save_index({"a": 1})
        assert list(outside.iterdir()) == []
        # Un archivo FIRMADO válido afuera del storage tampoco se lee.
        (outside / "hrm2.idx").write_bytes(_signed_bytes({"stolen": True}))
        with pytest.raises(ValueError):
            p.load_index()

    def test_index_file_symlink_escape_is_rejected(self, tmp_path, signed):
        root, p = _storage(tmp_path)
        outside = tmp_path / "outside.idx"
        outside.write_bytes(_signed_bytes({"stolen": True}))
        (p.storage_path / "index" / "hrm2.idx").symlink_to(outside)
        with pytest.raises(ValueError):
            p.load_index()

    def test_shard_symlink_escape_is_rejected(self, tmp_path):
        root, p = _storage(tmp_path)
        outside = tmp_path / "outside.npy"
        np.save(outside, np.zeros((2, 3), dtype=np.float32))
        (p.storage_path / "data" / "vectors" / "shard_001.npy").symlink_to(outside)
        with pytest.raises(ValueError):
            p.load_vectors("shard_001")

    def test_shard_name_with_dotdot_is_rejected(self, tmp_path):
        root, p = _storage(tmp_path)
        with pytest.raises(ValueError):
            p.save_vectors(np.zeros((1, 3), dtype=np.float32), ["a"], shard_name="../x")
        assert not (root / "x.npy").exists()
        with pytest.raises(ValueError):
            p.load_vectors("../../x")

    def test_legit_paths_still_work(self, tmp_path, signed):
        root, p = _storage(tmp_path)
        p.save_index({"k": [1, 2]}, name="custom")
        assert p.load_index("custom") == {"k": [1, 2]}
        p.save_vectors(np.ones((2, 3), dtype=np.float32), ["a", "b"])
        assert p.load_vectors("shard_001").shape == (2, 3)
        p.save_metadata_many([{"id": "a", "shard_idx": 1, "vector_idx": 0}, {"id": "b", "shard_idx": 1, "vector_idx": 1}])
        assert p.write_manifest() is not None
        assert p.verify_manifest() is True
