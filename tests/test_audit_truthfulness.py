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
