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


# ---------------------------------------------------------------------------
# r2-16 — constructor documentado, métrica real, SUPERCRITICAL alcanzable, relax
# ---------------------------------------------------------------------------


class TestR2_16SocConstructorMetricAndRelax:
    def test_documented_constructor_forms_clusters(self):
        x, _ = _clustered(120, 32, seed=9)
        alpha = np.ones(120, dtype=np.float32)
        kappa = np.full(120, 10.0, dtype=np.float32)
        # exactamente el uso del docstring de SOCEngine
        soc = SOCEngine(EBMEnergy(x, alpha, kappa), x, alpha, kappa)
        assert soc.get_statistics()["n_clusters"] > 0
        rep = soc.check_criticality()
        assert 0 < rep.n_clusters <= 120
        assert rep.size_variance > 0.0  # clústeres reales: los tamaños varían
        assert rep.energy_variance >= 0.0
        assert rep.state is not CriticalityState.SUPERCRITICAL  # store fresco
        assert rep.decayed_fraction == 0.0 and rep.kappa_dispersion == 0.0

    def test_metric_uses_mu_alpha_and_kappa(self):
        x = _random_unit(200, 32, seed=10)
        alpha = np.ones(200, dtype=np.float32)
        kappa = np.full(200, 10.0, dtype=np.float32)
        soc = SOCEngine(EBMEnergy(x, alpha, kappa), x, alpha, kappa)
        base = soc.check_criticality()
        assert 0.0 <= base.index < 0.3

        # κ: dispersión real de concentración
        kappa[:] = np.linspace(0.5, 100.0, 200).astype(np.float32)
        soc.update_splats(x, alpha, kappa)
        rep_k = soc.check_criticality()
        assert rep_k.kappa_dispersion > 0.5
        assert rep_k.components["kappa"] > 0.0
        assert rep_k.index > base.index

        # α: la mitad de los splats de un mismo grupo geométrico queda "caliente"
        soc.update_splats(x, np.ones(200, dtype=np.float32), np.full(200, 10.0, dtype=np.float32))
        alpha2 = np.ones(200, dtype=np.float32)
        alpha2[:20] = 10.0  # los 20 primeros splats comparten dirección (misma cuenca)
        x_tight = x.copy()
        x_tight[:20] = _unit(x[0] + 0.01 * np.random.default_rng(11).normal(size=(20, 32)))
        soc.update_splats(x_tight, alpha2, np.full(200, 10.0, dtype=np.float32))
        rep_a = soc.check_criticality()
        assert rep_a.energy_variance > 0.0
        assert rep_a.components["energy"] > 0.0
        assert rep_a.index > 0.0

        # μ: la misma α/κ pero toda la masa en una dirección -> tamaños desbalanceados
        x_skew = x.copy()
        x_skew[:170] = _unit(x[0] + 0.01 * np.random.default_rng(12).normal(size=(170, 32)))
        soc.update_splats(x_skew, np.ones(200, dtype=np.float32), np.full(200, 10.0, dtype=np.float32))
        rep_m = soc.check_criticality()
        assert rep_m.components["size"] > base.components["size"]
        assert rep_m.index > base.index

    def test_supercritical_reachable_standalone(self):
        x = _random_unit(100, 32, seed=13)
        alpha = np.r_[np.full(90, 0.01), np.full(10, 1.0)].astype(np.float32)
        kappa = np.full(100, 10.0, dtype=np.float32)
        soc = SOCEngine(EBMEnergy(x, alpha, kappa), x, alpha, kappa)
        rep = soc.check_criticality()
        assert rep.decayed_fraction == pytest.approx(0.9)
        assert rep.index >= 0.7
        assert rep.state is CriticalityState.SUPERCRITICAL
        assert rep.needs_relaxation() and rep.needs_monitoring()

    def test_supercritical_under_real_overload_via_public_api(self):
        """Sobrecarga real: muchos docs de α bajo por feedback 'irrelevante'."""
        db = AdvancedVectorDB(latent_dim=32)
        x, _ = _clustered(200, 32, seed=14)
        ids = [f"d{i}" for i in range(200)]
        db.add(ids=ids, vectors=x)
        assert db.check_criticality().state is not CriticalityState.SUPERCRITICAL
        for _ in range(60):
            db.feedback(x[0], irrelevant_ids=ids[:150])
        rep = db.check_criticality()
        print(f"\n[soc] tras sobrecarga: {rep.state.value} index={rep.index:.3f} {rep.components}")
        assert rep.decayed_fraction == pytest.approx(0.75)
        assert rep.state is CriticalityState.SUPERCRITICAL
        # nada es automático por defecto
        assert len(db._vectors) == 200
        assert db.auto_consolidate is False

    def test_relax_never_raises_free_energy(self):
        db = AdvancedVectorDB(latent_dim=32)
        x = _random_unit(60, 32, seed=15)
        db.add(ids=[f"d{i}" for i in range(60)], vectors=x)
        f0 = db.get_stats()["free_energy"]
        res = db.relax(iterations=10)
        f1 = db.get_stats()["free_energy"]
        print(f"\n[relax] F {f0:.5f} -> {f1:.5f} (delta={res.energy_delta:+.2e})")
        assert f1 <= f0 + 1e-9
        assert res.final_energy <= res.initial_energy + 1e-9
        assert res.iterations == 10

    def test_relax_strictly_lowers_free_energy_when_not_relaxed(self):
        db = AdvancedVectorDB(latent_dim=32)
        x = _random_unit(60, 32, seed=16)
        db.add(ids=[f"d{i}" for i in range(60)], vectors=x)
        st = db.engine.m2m.splats
        n = st.n_active
        st.kappa[:n] = 100.0  # sobre-afilado (lo que satura el feedback relevante)
        db._update_ebm_splats()
        f0 = db.get_stats()["free_energy"]
        res = db.relax(iterations=10)
        f1 = db.get_stats()["free_energy"]
        print(f"\n[relax no relajado] F {f0:.5f} -> {f1:.5f} | kappa max {st.kappa[:n].max():.2f}")
        assert f1 < f0 - 1e-6
        assert res.improved
        assert np.all(st.kappa[:n] < 100.0)
        # μ no se toca (es el embedding del documento) y α tampoco
        np.testing.assert_allclose(_norms(st.mu[:n]), 1.0, atol=1e-5)
        assert np.allclose(st.alpha[:n], 1.0)
        # un segundo relax sigue sin subir F
        f2 = db.get_stats()["free_energy"] if db.relax(iterations=10) else None
        assert f2 <= f1 + 1e-9


# ---------------------------------------------------------------------------
# r3a-07 — SOC sobre el store real, persistencia, invariante de seguridad
# ---------------------------------------------------------------------------


class TestR3a_07SocOperatesOnTheRealStore:
    def test_standalone_engine_writes_through_to_the_caller_arrays(self):
        x = _random_unit(40, 16, seed=17)
        alpha = np.ones(40, dtype=np.float32)
        kappa = np.full(40, 10.0, dtype=np.float32)
        soc = SOCEngine(EBMEnergy(x, alpha, kappa), x, alpha, kappa)
        a0 = alpha.copy()
        res = soc.trigger_avalanche()
        assert res.affected_clusters > 0 and res.energy_released > 0.0
        assert np.any(alpha < a0 - 1e-6)  # el array del llamador cambió
        np.testing.assert_allclose(soc.energy_api.splat_alpha, alpha)

    def test_avalanche_and_relax_mutate_real_store_and_survive_reopen(self, tmp_path):
        sp = str(tmp_path / "db")
        db = AdvancedVectorDB(latent_dim=32, storage_path=sp)
        x = _random_unit(40, 32, seed=18)
        db.add(ids=[f"d{i}" for i in range(40)], vectors=x)
        st = db.engine.m2m.splats
        n = st.n_active
        a0 = st.alpha[:n].copy()

        res = db.trigger_avalanche()
        assert res.affected_clusters > 0 and res.energy_released > 0.0
        a1 = st.alpha[:n].copy()
        assert np.any(a1 < a0 - 1e-6)
        assert np.all(a1 >= 0.01 - 1e-7)
        np.testing.assert_allclose(db._ebm_energy.splat_alpha, a1)  # el EBM ve el store

        st.kappa[:n] = 100.0
        db._update_ebm_splats()
        r = db.relax(iterations=10)
        k1 = st.kappa[:n].copy()
        assert r.improved and np.all(k1 < 100.0)

        db.save()
        db2 = AdvancedVectorDB(latent_dim=32, storage_path=sp)
        st2 = db2.engine.m2m.splats
        assert st2.n_active == n
        np.testing.assert_allclose(st2.alpha[:n], a1, rtol=1e-5, atol=1e-7)
        np.testing.assert_allclose(st2.kappa[:n], k1, rtol=1e-5, atol=1e-7)

    def test_memory_db_snapshot_carries_the_learned_state(self, tmp_path):
        db = AdvancedVectorDB(latent_dim=16)
        x = _random_unit(20, 16, seed=19)
        db.add(ids=[f"d{i}" for i in range(20)], vectors=x)
        db.trigger_avalanche()
        st = db.engine.m2m.splats
        a1 = st.alpha[: st.n_active].copy()
        assert np.any(a1 < 1.0 - 1e-6)  # la avalanche tocó el store real
        out = db.save(str(tmp_path / "snap"))
        db2 = AdvancedVectorDB(latent_dim=16, storage_path=out)
        st2 = db2.engine.m2m.splats
        np.testing.assert_allclose(st2.alpha[: st2.n_active], a1, rtol=1e-5, atol=1e-7)

    def test_no_orphans_after_consolidate_relax_and_avalanche(self, tmp_path):
        sp = str(tmp_path / "db")
        db = AdvancedVectorDB(latent_dim=32, storage_path=sp)
        x, _ = _clustered(60, 32, seed=20)
        ids = [f"d{i}" for i in range(60)]
        db.add(ids=ids, vectors=x)  # 1 splat por doc
        for _ in range(60):
            db.feedback(x[0], irrelevant_ids=ids[10:20])

        def check(d):
            order = d._splat_id_order
            store = d.engine.m2m.splats
            active = _active_ids(d)
            assert len(order) == store.n_active
            assert set(order) == set(active), "posiciones sin doc o docs sin splat"
            for doc_id in active:
                assert order.count(doc_id) >= 1
                got = [r.id for r in d.search(d._vectors[doc_id], k=len(active), include_metadata=True)]
                assert doc_id in got, f"{doc_id} activo pero no encontrado por search()"
                assert set(got) <= set(active)

        check(db)
        removed = db.consolidate(threshold=0.05)
        assert removed == 10
        check(db)
        f_before = db.get_stats()["free_energy"]
        assert db.relax().final_energy <= f_before + 1e-9
        check(db)
        db.trigger_avalanche()
        check(db)
        db.save()
        db2 = AdvancedVectorDB(latent_dim=32, storage_path=sp)
        check(db2)
        assert not (set(ids[10:20]) & set(db2._vectors)), "docs podados resucitaron del storage"

    def test_maybe_consolidate_explicit_and_idempotent(self):
        db = AdvancedVectorDB(latent_dim=32)
        x, _ = _clustered(60, 32, seed=21)
        ids = [f"d{i}" for i in range(60)]
        db.add(ids=ids, vectors=x)
        for _ in range(60):
            db.feedback(x[0], irrelevant_ids=ids[:45])
        assert len(db._vectors) == 60  # nada automático
        assert db.check_criticality().state is CriticalityState.SUPERCRITICAL

        dec = db.maybe_consolidate()
        assert isinstance(dec, ConsolidationDecision)
        print(f"\n[maybe_consolidate] {dec}")
        assert dec.triggered is True and dec.removed == 45
        assert len(db._vectors) == 15
        assert dec.state is not CriticalityState.SUPERCRITICAL
        for doc_id in _active_ids(db):
            got = [r.id for r in db.search(db._vectors[doc_id], k=15, include_metadata=True)]
            assert doc_id in got

        dec2 = db.maybe_consolidate()
        assert dec2.triggered is False and dec2.removed == 0
        assert len(db._vectors) == 15

    def test_maybe_consolidate_without_soc_raises(self):
        db = AdvancedVectorDB(latent_dim=16, enable_soc=False, enable_energy_features=False)
        with pytest.raises(RuntimeError, match="SOC"):
            db.maybe_consolidate()

    def test_auto_consolidate_is_opt_in(self):
        db = AdvancedVectorDB(latent_dim=32, auto_consolidate=True)
        x, _ = _clustered(60, 32, seed=22)
        ids = [f"d{i}" for i in range(60)]
        db.add(ids=ids, vectors=x)
        for _ in range(60):
            db.feedback(x[0], irrelevant_ids=ids[:45])
        assert len(db._vectors) == 15
        assert set(_active_ids(db)) == set(ids[45:])
