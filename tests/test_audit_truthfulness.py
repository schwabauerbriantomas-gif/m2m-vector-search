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
