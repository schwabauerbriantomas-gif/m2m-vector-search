"""Regresión del pack P7c — online learning + SOC.

Ítems de auditoría:
  - r2-14: el drift Hebbiano de ``OnlineUpdater`` no re-normaliza μ (‖μ‖ 1 → 0.073
    tras 500 feedbacks relevantes).
  - r3a-08: ``feedback()`` no está expuesto en ``SimpleVectorDB``/``AdvancedVectorDB``
    y recibe índices de splat (inestables) en vez de ids de documento.
  - r2-16 / r3a-07: ``SOCEngine`` con el uso documentado deja 0 clústeres,
    SUPERCRITICAL es inalcanzable, ``relax()`` sube la energía libre y avalanche /
    relax operan sobre copias desacopladas del store.

Decisiones del orquestador que fijan estas aserciones:
  - el update Hebbiano mantiene μ en la esfera (rechaza el paso degenerado);
  - ``feedback()`` va por IDS de documento, bajo ``_rw_lock``, resolviendo
    posiciones en el momento de la llamada;
  - la métrica de criticalidad usa μ/κ/α reales; SUPERCRITICAL es alcanzable por
    sobrecarga real (muchos splats de α bajo); ``relax()`` nunca sube F;
  - avalanche/relax mutan el store real y sobreviven a ``save()`` + reopen;
    ningún doc activo queda huérfano tras poda/avalanche/relax.
"""

from __future__ import annotations

import numpy as np
import pytest

import m2m
from m2m import AdvancedVectorDB, SimpleVectorDB
from m2m.ebm import EBMEnergy, SOCEngine
from m2m.ebm import soc as _soc
from m2m.ebm.soc import CriticalityState
from m2m.online_updates import FeedbackEvent, OnlineUpdater

# API nueva del pack: con getattr para que el log ROJO muestre la falla de CADA test
# (y no un ImportError de colección) cuando la API todavía no existe.
FeedbackResult = getattr(m2m, "FeedbackResult", None)
ConsolidationDecision = getattr(_soc, "ConsolidationDecision", None)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _unit(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def _random_unit(n: int, d: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return _unit(rng.normal(size=(n, d)).astype(np.float32))


def _clustered(n: int, d: int, seed: int = 0, k: int = 8, noise: float = 0.25):
    rng = np.random.default_rng(seed)
    centers = _unit(rng.normal(size=(k, d)).astype(np.float32))
    labels = rng.integers(0, k, size=n)
    x = centers[labels] + noise * rng.normal(size=(n, d)).astype(np.float32)
    return _unit(x), labels


def _norms(a: np.ndarray) -> np.ndarray:
    return np.linalg.norm(np.asarray(a, dtype=np.float64), axis=-1)


def _active_ids(db) -> list:
    return [d for d in db._vectors if d not in db._deleted]


def _brute_topk(x: np.ndarray, q: np.ndarray, k: int) -> list:
    sims = x @ q
    return [int(i) for i in np.argsort(-sims)[:k]]


# ---------------------------------------------------------------------------
# r2-14 — drift Hebbiano: μ se queda en la esfera
# ---------------------------------------------------------------------------


class TestR2_14HebbianDriftKeepsMuOnSphere:
    def _arrays(self, n=8, d=64, seed=0):
        rng = np.random.default_rng(seed)
        mu = _unit(rng.normal(size=(n, d)).astype(np.float32))
        alpha = np.ones(n, dtype=np.float32)
        kappa = np.full(n, 10.0, dtype=np.float32)
        return mu, alpha, kappa

    def test_500_relevant_feedbacks_keep_unit_norm(self):
        mu, alpha, kappa = self._arrays()
        mu0 = mu[0].copy()
        rng = np.random.default_rng(1)
        upd = OnlineUpdater(lr_mu=0.01)
        for _ in range(500):
            # queries unitarias de la cuenca del splat (como las que lo encuentran relevante)
            q = _unit(mu0 + 0.3 * rng.normal(size=mu.shape[1]).astype(np.float32))
            upd.apply_feedback(
                FeedbackEvent(query=q, splat_index=0, relevant=True), mu, alpha, kappa
            )
        assert np.isfinite(mu).all() and np.isfinite(alpha).all() and np.isfinite(kappa).all()
        np.testing.assert_allclose(_norms(mu[0]), 1.0, atol=1e-3)
        assert upd.alpha_min <= alpha[0] <= upd.alpha_max
        assert upd.kappa_min <= kappa[0] <= upd.kappa_max
        # los demás splats no se tocan
        np.testing.assert_allclose(_norms(mu[1:]), 1.0, atol=1e-6)

    def test_splat_moves_toward_the_feedback(self):
        mu, alpha, kappa = self._arrays()
        q = _unit(mu[0] + 0.5 * np.random.default_rng(2).normal(size=mu.shape[1]))
        upd = OnlineUpdater(lr_mu=0.05)
        before = float(mu[0] @ q)
        last = before
        for _ in range(20):
            upd.apply_feedback(
                FeedbackEvent(query=q, splat_index=0, relevant=True), mu, alpha, kappa
            )
            now = float(mu[0] @ q)
            assert now >= last - 1e-7  # monótono hacia la query
            last = now
        assert last > before + 1e-3
        np.testing.assert_allclose(_norms(mu[0]), 1.0, atol=1e-5)

    def test_degenerate_step_is_rejected_without_corrupting_the_splat(self):
        mu, alpha, kappa = self._arrays()
        mu0 = mu[0].copy()
        # μ' = μ + 0.5·(−μ − μ) = 0: paso degenerado (norma < 1e-6)
        upd = OnlineUpdater(lr_mu=0.5)
        upd.apply_feedback(
            FeedbackEvent(query=-mu0, splat_index=0, relevant=True), mu, alpha, kappa
        )
        np.testing.assert_array_equal(mu[0], mu0)
        assert np.isfinite(mu[0]).all()
        np.testing.assert_allclose(_norms(mu[0]), 1.0, atol=1e-6)
        # α/κ son reglas independientes y sí se aplican
        assert alpha[0] > 1.0 and kappa[0] > 10.0

    def test_non_finite_query_does_not_corrupt_mu(self):
        mu, alpha, kappa = self._arrays()
        mu0 = mu[0].copy()
        q = mu0.copy()
        q[0] = np.nan
        OnlineUpdater(lr_mu=0.1).apply_feedback(
            FeedbackEvent(query=q, splat_index=0, relevant=True), mu, alpha, kappa
        )
        np.testing.assert_array_equal(mu[0], mu0)
