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

    @pytest.mark.parametrize("cls", [SimpleVectorDB, AdvancedVectorDB])
    def test_db_feedback_keeps_every_splat_on_the_sphere(self, cls):
        db = cls(latent_dim=32, enable_ebm=True) if cls is SimpleVectorDB else cls(latent_dim=32)
        x = _random_unit(20, 32, seed=3)
        ids = [f"d{i}" for i in range(20)]
        db.add(ids=ids, vectors=x)
        rng = np.random.default_rng(4)
        for _ in range(500):
            q = _unit(x[0] + 0.3 * rng.normal(size=32).astype(np.float32))
            db.feedback(q, relevant_ids=["d0", "d1"])
        st = db.engine.m2m.splats
        n = st.n_active
        assert np.isfinite(st.mu[:n]).all() and np.isfinite(st.alpha[:n]).all()
        assert np.isfinite(st.kappa[:n]).all()
        assert np.all(_norms(st.mu[:n]) >= 0.999) and np.all(_norms(st.mu[:n]) <= 1.001)
        assert st.alpha[:n].max() <= 10.0 + 1e-6 and st.kappa[:n].max() <= 100.0 + 1e-6


# ---------------------------------------------------------------------------
# r3a-08 — feedback() público por ids de documento
# ---------------------------------------------------------------------------


class TestR3a_08FeedbackApi:
    @pytest.mark.parametrize("cls", [SimpleVectorDB, AdvancedVectorDB])
    def test_feedback_exists_and_reports_unknown_ids(self, cls):
        db = cls(latent_dim=32, enable_ebm=True) if cls is SimpleVectorDB else cls(latent_dim=32)
        x, _ = _clustered(12, 32, seed=5)
        ids = [f"d{i}" for i in range(12)]
        db.add(ids=ids, vectors=x)
        db.delete("d7")  # soft-deleted: no es un target válido

        res = db.feedback(
            x[0],
            relevant_ids=["d0", "nope", "d1", "d0"],
            irrelevant_ids=["d2", "gone", "d7"],
        )
        assert isinstance(res, FeedbackResult)
        assert res.n_relevant == 2  # d0 (una vez), d1
        assert res.n_irrelevant == 1  # d2
        assert res.unknown_ids == ["d7", "gone", "nope"]  # ordenados
        st = db.engine.m2m.splats
        p = {d: i for i, d in enumerate(db._splat_id_order)}
        assert st.alpha[p["d0"]] > 1.0 and st.alpha[p["d1"]] > 1.0
        assert st.alpha[p["d2"]] < 1.0
        assert np.isclose(st.alpha[p["d3"]], 1.0)
        # el EBM ve el estado aprendido
        np.testing.assert_allclose(db._ebm_energy.splat_alpha, st.alpha[: st.n_active])

    def test_feedback_validates_query(self):
        db = SimpleVectorDB(latent_dim=8, enable_ebm=True)
        with pytest.raises(ValueError, match="dimension"):
            db.feedback(np.zeros(4, dtype=np.float32), relevant_ids=[])
        with pytest.raises(ValueError, match="NaN"):
            q = np.zeros(8, dtype=np.float32)
            q[0] = np.nan
            db.feedback(q, relevant_ids=[])

    def test_feedback_loop_improves_ranking_without_hurting_recall(self):
        """Un loop de feedback relevante mejora el ranking del doc para ESA consulta
        (rank menor y score mayor, medidos) sin empeorar recall@10 del resto > 1%."""
        x, _ = _clustered(400, 64, seed=6)
        ids = [f"d{i}" for i in range(400)]
        db = SimpleVectorDB(latent_dim=64, enable_ebm=True)
        db.add(ids=ids, vectors=x)

        rng = np.random.default_rng(7)
        q_idx = rng.choice(400, 100, replace=False)
        queries = _unit(x[q_idx] + 0.15 * rng.normal(size=(100, 64)).astype(np.float32))
        q = queries[0]
        others = list(range(1, 100))

        def recall():
            hits = tot = 0
            for j in others:
                got = {r.id for r in db.search(queries[j], k=10, include_metadata=True)}
                truth = {ids[i] for i in _brute_topk(x, queries[j], 10)}
                hits += len(got & truth)
                tot += 10
            return hits / tot

        before = db.search(q, k=10, include_metadata=True)
        rank0 = [r.id for r in before].index("d%d" % q_idx[0])  # el propio doc: rank 0
        # target: el 3er resultado (no el primero) para que el rank pueda mejorar
        target = before[2].id
        rank0 = 2
        score0 = before[2].score
        recall0 = recall()

        for _ in range(300):
            db.feedback(q, relevant_ids=[target])

        after = db.search(q, k=10, include_metadata=True)
        ids_after = [r.id for r in after]
        rank1 = ids_after.index(target) if target in ids_after else len(ids_after)
        score1 = next(r.score for r in after if r.id == target)
        recall1 = recall()
        print(
            f"\n[feedback] rank {rank0}->{rank1} | score {score0:.4f}->{score1:.4f} | "
            f"recall@10 otros {recall0:.4f}->{recall1:.4f}"
        )
        assert score1 > score0 + 1e-3
        assert rank1 < rank0
        assert recall0 >= 0.95
        assert recall1 >= recall0 - 0.01
        st = db.engine.m2m.splats
        n = st.n_active
        assert np.all(_norms(st.mu[:n]) >= 0.999) and np.all(_norms(st.mu[:n]) <= 1.001)
        assert np.isfinite(st.alpha[:n]).all() and np.isfinite(st.kappa[:n]).all()

    def test_irrelevant_feedback_then_consolidate_prunes_those_docs(self):
        db = AdvancedVectorDB(latent_dim=32)
        x, _ = _clustered(20, 32, seed=8)
        ids = [f"d{i}" for i in range(20)]
        db.add(ids=ids, vectors=x)
        for _ in range(60):
            db.feedback(x[0], irrelevant_ids=["d3", "d4", "d5"])
        st = db.engine.m2m.splats
        p = {d: i for i, d in enumerate(db._splat_id_order)}
        assert max(st.alpha[p[d]] for d in ("d3", "d4", "d5")) <= 0.01 + 1e-6
        removed = db.consolidate(threshold=0.05)
        assert removed == 3
        assert not ({"d3", "d4", "d5"} & set(db._vectors))
        got = [r.id for r in db.search(x[0], k=20, include_metadata=True)]
        assert len(got) == 17 and not ({"d3", "d4", "d5"} & set(got))
