"""Tests de verdad de claims del código — pack P7b (audit r3a-18, r3a-09, r3a-10, r3a-04).

Cada test reproduce un claim fuso que la auditoría encontró en el código:
- r3a-18: `python -m m2m` resolvía host 0.0.0.0 aunque SECURITY.md promete 127.0.0.1.
- r3a-09: logs de backend GPU no veraces; `device='cuda'` sin torch corría en CPU
  sin aviso y el selector no reportaba el backend efectivo.
- r3a-10: `m2m.exp_map`/`log_map` eran un fallback euclidiano que sombreaba a
  `m2m.geometry`; dos clases `M2MEngine` sin documentar la diferencia.
- r3a-04: `__version__` (2.3.0) != pyproject (2.3.1).
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import tomllib
import types
import warnings
from pathlib import Path

import numpy as np
import pytest

import m2m
import m2m.engine
import m2m.geometry

REPO_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# r3a-18 — python -m m2m escucha en 127.0.0.1 por defecto
# ---------------------------------------------------------------------------


def test_main_default_host_is_loopback(monkeypatch):
    """Sin M2M_HOST, `python -m m2m` resuelve 127.0.0.1 (lo que promete SECURITY.md)."""
    from m2m.__main__ import resolve_bind_addr

    monkeypatch.delenv("M2M_HOST", raising=False)
    monkeypatch.delenv("M2M_PORT", raising=False)
    assert resolve_bind_addr() == ("127.0.0.1", 8080)


def test_main_env_overrides_host_and_port(monkeypatch):
    from m2m.__main__ import resolve_bind_addr

    monkeypatch.setenv("M2M_HOST", "0.0.0.0")
    monkeypatch.setenv("M2M_PORT", "9000")
    assert resolve_bind_addr() == ("0.0.0.0", 9000)


def _fake_uvicorn(monkeypatch, calls):
    fake = types.ModuleType("uvicorn")
    fake.run = lambda app, **kw: calls.update(kw)
    monkeypatch.setitem(sys.modules, "uvicorn", fake)


def test_main_passes_resolved_host_to_uvicorn(monkeypatch, caplog):
    """main() le pasa a uvicorn el host resuelto; no levanta el servidor (uvicorn fingido)."""
    import m2m.__main__ as m

    calls = {}
    _fake_uvicorn(monkeypatch, calls)
    monkeypatch.delenv("M2M_HOST", raising=False)
    monkeypatch.delenv("M2M_PORT", raising=False)
    monkeypatch.delenv("M2M_API_KEY", raising=False)
    with caplog.at_level(logging.WARNING):
        m.main()
    assert (calls["host"], calls["port"]) == ("127.0.0.1", 8080)
    assert not [r for r in caplog.records if "M2M_API_KEY" in r.getMessage()]


def test_main_warns_when_binding_non_loopback_without_api_key(monkeypatch, caplog):
    import m2m.__main__ as m

    _fake_uvicorn(monkeypatch, {})
    monkeypatch.setenv("M2M_HOST", "0.0.0.0")
    monkeypatch.delenv("M2M_API_KEY", raising=False)
    with caplog.at_level(logging.WARNING):
        m.main()
    assert [r for r in caplog.records if r.levelno == logging.WARNING and "M2M_API_KEY" in r.getMessage()]


@pytest.mark.parametrize("dockerfile", ["Dockerfile", "deploy/Dockerfile"])
def test_dockerfiles_keep_container_reachable(dockerfile):
    """El default pasó a 127.0.0.1: los Dockerfiles setean M2M_HOST=0.0.0.0 explícito."""
    assert "ENV M2M_HOST=0.0.0.0" in (REPO_ROOT / dockerfile).read_text()


# ---------------------------------------------------------------------------
# r3a-09 — logs veraces del backend GPU + warning con device='cuda' sin torch
# ---------------------------------------------------------------------------


def _no_torch(monkeypatch):
    """`import torch` pasa a levantar ImportError aunque torch esté instalado."""
    monkeypatch.setitem(sys.modules, "torch", None)


def _db(device):
    return m2m.SimpleVectorDB(latent_dim=8, device=device)


def _reset_guard():
    getattr(m2m.engine, "_reset_backend_warning_guard", lambda: None)()


def test_cuda_without_torch_reports_cpu_as_effective_backend(monkeypatch):
    """El backend reportado coincide con lo que corre: sin torch nada de CUDA está activo."""
    _no_torch(monkeypatch)
    db = _db("cuda")
    splats = db.engine.m2m.splats
    router = splats.engine.router
    rng = np.random.default_rng(0)
    db.add(ids=[f"d{i}" for i in range(4)], vectors=rng.normal(size=(4, 8)).astype("float32"))
    db.search(rng.normal(size=8).astype("float32"), k=2)
    # lo que corre realmente
    assert router.gpu_router is None and router.use_cuda is False
    assert splats._cuda_searcher is None
    # lo que el selector reporta
    info = db.engine.get_backend_info()
    assert info["requested"] == "cuda"
    assert info["effective"] == "cpu"
    assert "torch" in info["reasons"]["cuda"]
    assert db.engine.backend == "cpu"
    assert router.backend == "cpu"
    assert router.get_backend_info() == info


def test_cuda_without_torch_warns_once(monkeypatch, caplog):
    """UserWarning explícito una sola vez por proceso + logger.warning en cada construcción."""
    _no_torch(monkeypatch)
    _reset_guard()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with caplog.at_level(logging.WARNING):
            _db("cuda")
            _db("cuda")
    uw = [w for w in caught if issubclass(w.category, UserWarning) and "cuda" in str(w.message).lower()]
    assert len(uw) == 1, [str(w.message) for w in caught]
    assert "cpu" in str(uw[0].message).lower()
    logs = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert sum("cuda" in m.lower() for m in logs) >= 2, logs


def test_vulkan_unavailable_is_not_logged_as_enabled(monkeypatch, caplog):
    """Con enable_vulkan y Vulkan caído, ningún log afirma que la aceleración está habilitada."""
    import m2m.gpu_vector_index as gvi

    class _BrokenVulkan:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("simulación: Vulkan no disponible")

    monkeypatch.setattr(gvi, "GPUVectorIndex", _BrokenVulkan)
    with caplog.at_level(logging.INFO, logger="m2m"):
        db = _db("vulkan")
    msgs = [r.getMessage() for r in caplog.records]
    enabled_claims = [m for m in msgs if "acceleration enabled" in m.lower()]
    assert enabled_claims == [], enabled_claims
    assert any("vulkan" in m.lower() and "cpu" in m.lower() for m in msgs), msgs
    assert db.engine.backend == "cpu"
    info = db.engine.get_backend_info()
    assert (info["requested"], info["effective"]) == ("vulkan", "cpu")
    assert info["reasons"]["vulkan"]


def test_cuda_available_reports_cuda_without_warning(monkeypatch, caplog):
    """Rama 'disponible': con CUDA usable el selector reporta cuda y no emite fallback."""
    import m2m.gpu_vector_index as gvi

    class _FakeCUDAIndex:
        def __init__(self, index, max_batch_size=1):
            self.index = index

    monkeypatch.setattr(gvi, "CUDAVectorIndex", _FakeCUDAIndex)
    monkeypatch.setattr(m2m.engine, "_cuda_unavailable_reason", lambda: None, raising=False)
    _reset_guard()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with caplog.at_level(logging.INFO, logger="m2m"):
            db = _db("cuda")
    assert db.engine.backend == "cuda"
    assert db.engine.get_backend_info() == {"requested": "cuda", "effective": "cuda", "reasons": {}}
    assert not [w for w in caught if issubclass(w.category, UserWarning)]
    assert any("CUDA" in r.getMessage() for r in caplog.records)


def test_cpu_default_is_not_a_fallback():
    db = _db("cpu")
    assert db.engine.get_backend_info() == {"requested": "cpu", "effective": "cpu", "reasons": {}}


# ---------------------------------------------------------------------------
# r3a-10 — sin sombreado de m2m.geometry ni duplicidad silenciosa de M2MEngine
# ---------------------------------------------------------------------------

GEOMETRY_NAMES = (
    "exp_map",
    "log_map",
    "geodesic_distance",
    "normalize_sphere",
    "project_to_tangent",
)


@pytest.mark.parametrize("name", GEOMETRY_NAMES)
def test_package_exports_real_geometry(name):
    """`from m2m import exp_map` debe dar la operación riemanniana de m2m.geometry."""
    assert getattr(m2m, name) is getattr(m2m.geometry, name)


def test_exp_map_stays_on_sphere():
    base = np.array([1.0, 0.0, 0.0])
    tangent = np.array([0.0, 0.5, 0.0])  # ortogonal a base
    out = m2m.exp_map(base, tangent)
    np.testing.assert_allclose(out, [np.cos(0.5), np.sin(0.5), 0.0], atol=1e-12)
    np.testing.assert_allclose(np.linalg.norm(out), 1.0, atol=1e-12)


def test_log_exp_roundtrip_on_sphere():
    rng = np.random.default_rng(0)
    base = m2m.normalize_sphere(rng.normal(size=5))
    point = m2m.normalize_sphere(rng.normal(size=5))
    np.testing.assert_allclose(m2m.exp_map(base, m2m.log_map(base, point)), point, atol=1e-9)


def test_normalize_sphere_matches_legacy_formula_bitwise():
    """El fallback de normalize_sphere era numéricamente idéntico al de geometry
    (mismo eje -1, mismo eps): el comportamiento de SimpleVectorDB no cambia."""
    rng = np.random.default_rng(0)
    for shape in [(6,), (1, 6), (17, 6)]:
        x = rng.normal(size=shape).astype("float32")
        legacy = x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)
        assert np.array_equal(m2m.normalize_sphere(x), legacy)


def test_all_is_consistent_and_includes_hrm2config():
    assert len(m2m.__all__) == len(set(m2m.__all__))
    for name in m2m.__all__:
        assert hasattr(m2m, name), name
    for name in GEOMETRY_NAMES + ("M2MEngine", "M2MConfig", "HRM2Config", "SimpleVectorDB", "__version__"):
        assert name in m2m.__all__, name
    assert m2m.HRM2Config is m2m.hrm2_engine.HRM2Config


def test_two_m2m_engine_classes_keep_their_module_and_are_documented():
    """Compat: m2m.M2MEngine sigue siendo el envoltorio de M2MMemory; el selector de
    hardware sigue siendo m2m.engine.M2MEngine (HRM2Engine.router)."""
    wrapper, hardware = m2m.M2MEngine, m2m.engine.M2MEngine
    assert wrapper is not hardware
    assert (wrapper.__module__, hardware.__module__) == ("m2m", "m2m.engine")
    assert "m2m.engine.M2MEngine" in wrapper.__doc__
    assert "m2m.M2MEngine" in hardware.__doc__
    db = _db("cpu")
    assert type(db.engine) is wrapper
    assert type(db.engine.m2m.splats.engine.router) is hardware
    assert db.engine.get_backend_info() == db.engine.m2m.splats.engine.router.get_backend_info()


def _top_id(db, result):
    """Id del primer resultado: DocResult (include_metadata=True) o la tupla legacy
    (vectors, scores, energy, indices posicionales) -> posición de splat -> id."""
    if hasattr(result[0], "id"):
        return result[0].id
    pos = int(np.asarray(result[-1]).reshape(-1)[0])
    return db._splat_id_order[pos]


def test_simple_vector_db_numeric_behaviour_smoke():
    rng = np.random.default_rng(0)
    vecs = rng.normal(size=(20, 8)).astype("float32")
    db = _db("cpu")
    db.add(ids=[f"d{i}" for i in range(20)], vectors=vecs)
    assert _top_id(db, db.search(vecs[7], k=1, include_metadata=True)) == "d7"
    db.update("d7", vector=vecs[3], upsert=True)
    assert _top_id(db, db.search(vecs[7], k=1, include_metadata=True)) != "d7"
    db.delete("d3", hard=True)
    assert _top_id(db, db.search(vecs[3], k=1, include_metadata=True)) != "d3"


def test_broken_core_import_raises_clear_import_error():
    """Si un módulo del núcleo no importa, `import m2m` falla con ImportError claro
    (antes: M2MEngine=None y el resto de nombres indefinidos -> NameError después)."""
    code = "import sys; sys.modules['m2m.splats'] = None\nimport m2m\n"
    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT / "src")}
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, cwd=str(REPO_ROOT), env=env
    )
    assert proc.returncode != 0, proc.stdout
    assert "la instalación está incompleta" in proc.stderr, proc.stderr[-800:]
