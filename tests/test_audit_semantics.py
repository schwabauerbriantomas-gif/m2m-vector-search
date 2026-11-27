"""Pack P7a — semántica de score / energía / confianza y temperatura de Boltzmann.

Ítems: r1-13, r3a-02 (reports r1/r3a), r2-15, r2-17 (report r2).

Decisiones del orquestador (cambios INTENCIONALES de semántica, citadas en cada test):
  D1. ``DocResult.score`` = similitud COSENO en [-1, 1] entre la query normalizada a la
      esfera y el vector almacenado (normalizado), IGUAL en todos los caminos:
      search() con/sin metadata, con/sin filtro, con/sin energía, modo LSH y
      search_batch(include_metadata=True). Antes: alpha (1.0 constante) sin filtro,
      -dist con filtro, -sqrt(d2) contra la query cruda en batch, distancia del LSH.
  D2. Campos nuevos y opcionales: ``alpha`` (peso del splat) y ``distance`` (L2 a la
      query normalizada). Sobre la esfera d² = 2 − 2·cos.
  D3. UNA implementación canónica de energía (``m2m.ebm.energy_api.splat_energy``) con
      la misma normalización de entrada (esfera) en todas las APIs; tope OOD único.
  D4. ``confidence = sigmoid(−E) = 1 / (1 + e^E)`` ∈ (0, 1): monotona decreciente en la
      energía, independiente de k, sin set de referencia. Antes: 1/(1+E) → negativa.
  D5. Boltzmann: T < 0 (o NaN) → ValueError; T ≥ 0 válida por chica que sea; T = 0 →
      límite determinista sin lanzar.
"""

import math
from types import SimpleNamespace

import numpy as np
import pytest

from m2m import SimpleVectorDB
from m2m.ebm import energy_api as ea
from m2m.ebm import EBMEnergy, EBMExploration
from m2m.energy import EnergyFunction


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _unit(x):
    """La MISMA normalización que aplica add_splats a μ: x / (‖x‖ + 1e-8)."""
    return x / (np.linalg.norm(x) + 1e-8)


def _random_db(n=300, d=16, seed=0, **kw):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d)).astype(np.float32)
    meta = [{"cat": i % 3, "r": float(v)} for i, v in enumerate(rng.random(n))]
    db = SimpleVectorDB(latent_dim=d, **kw)
    db.add(ids=[f"d{i}" for i in range(n)], vectors=X, metadata=meta)
    return db, X


def _assert_cosine(q, res):
    """D1 + D2 para una lista de DocResult contra la query q (cruda)."""
    qn = _unit(np.asarray(q, dtype=np.float64))
    assert len(res) > 0
    for r in res:
        assert -1.0 <= r.score <= 1.0
        np.testing.assert_allclose(r.score, float(np.dot(qn, r.vector)), atol=1e-5)
        assert r.alpha is None or r.alpha > 0
        assert r.distance is not None
        np.testing.assert_allclose(
            r.distance, float(np.linalg.norm(r.vector - qn)), atol=1e-5
        )
        # sobre la esfera: d² = 2 − 2·cos
        np.testing.assert_allclose(r.distance**2, 2.0 - 2.0 * r.score, atol=1e-5)


def _ebm_fixture():
    """5 splats D=4 con α y κ variados (como la sonda p5_online_ebm.py del audit)."""
    rng = np.random.default_rng(0)
    mu = _unit(rng.standard_normal((5, 4))).astype(np.float32)
    alpha = np.array([0.5, 1.0, 10.0, 2.0, 0.1], dtype=np.float32)
    kappa = np.array([5.0, 12.0, 10.0, 18.5, 8.0], dtype=np.float32)
    ebm = EBMEnergy(mu, alpha, kappa)
    splats = SimpleNamespace(mu=mu, alpha=alpha, kappa=kappa, n_active=5)
    cfg = SimpleNamespace(energy_splat_weight=1.0, energy_geom_weight=0.1)
    return ebm, EnergyFunction(cfg), splats




def test_r2_15_confidence_bounded_high_at_center_low_far():
    # Antes: centro con α=10 → E=-2.3256, confidence=-0.7544; α=0.5 → -1.0091;
    # E=-1 → división por cero.
    ebm = EBMEnergy(np.array([[1.0, 0, 0, 0]], dtype=np.float32), np.array([10.0]), np.array([10.0]))
    center = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    r = ebm.get_result(center)
    assert r.energy == pytest.approx(-math.log(10.0), abs=1e-5)
    assert 0.5 < r.confidence <= 1.0

    ebm_half = EBMEnergy(center[None].copy(), np.array([0.5]), np.array([10.0]))
    assert 0.0 <= ebm_half.get_result(center).confidence <= 1.0

    # a 1 radián del centro (sobre la esfera) la confianza baja
    tangent = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    far = math.cos(1.0) * center + math.sin(1.0) * tangent
    rf = ebm.get_result(far)
    assert rf.confidence < 0.1 < r.confidence

    # barrido: acotada, estrictamente monotona decreciente en E, sin singularidad en E=-1
    grid = np.linspace(-5.0, 15.0, 41)
    cs = [ea.confidence_from_energy(float(e)) for e in grid]
    assert all(0.0 <= c <= 1.0 for c in cs)
    assert all(a > b for a, b in zip(cs, cs[1:]))
    assert ea.confidence_from_energy(-1.0) == pytest.approx(1.0 / (1.0 + math.exp(-1.0)))
    # el tope OOD (10) y "sin splats" también caen dentro de (0, 1)
    assert 0.0 < ea.confidence_from_energy(ea.OOD_ENERGY) < 0.01


def test_r2_15_energy_apis_agree_same_input():
    # Antes: mismo punto OOD → EBMEnergy.energy = 10.0 vs EnergyFunction.E_splats = 69.08;
    # sin splats → 1.0 vs 10.0.
    ebm, ef, splats = _ebm_fixture()
    rng = np.random.default_rng(7)
    mu = splats.mu
    pts = [mu[i] for i in range(5)]
    pts += [mu[0] * 3.7, (mu[1] + mu[2]) / 2.0, rng.standard_normal(4) * 2.0]
    pts += [np.full(4, 50.0), np.zeros(4), np.array([3.0, -4.0, 0.0, 0.0])]
    for x in pts:
        x = np.asarray(x, dtype=np.float32)
        e_api = ebm.energy(x)
        e_fn = float(ef.E_splats(x, splats)[0])
        np.testing.assert_allclose(e_api, e_fn, atol=1e-5)
        np.testing.assert_allclose(float(ebm.energy_batch(x[None, :])[0]), e_fn, atol=1e-5)
        assert math.isfinite(e_api) and e_api <= ea.OOD_ENERGY + 1e-9
    # OOD (densidad < e^-10 bajo los splats considerados): antes 10.0 (EBMEnergy) vs
    # 69.08 = -log(1e-30) (E_splats). Sobre la esfera el punto más lejano a un splat es
    # su antípoda (d² = 4 → E = 40 sin tope).
    one = EBMEnergy(np.array([[1.0, 0, 0, 0]], dtype=np.float32), np.array([1.0]), np.array([10.0]))
    one_splats = SimpleNamespace(
        mu=one.splat_mu, alpha=one.splat_alpha, kappa=one.splat_kappa, n_active=1
    )
    antipode = np.array([-1.0, 0, 0, 0], dtype=np.float32)
    assert one.energy(antipode) == ea.OOD_ENERGY == float(ef.E_splats(antipode, one_splats)[0])
    assert one.energy(antipode * 37.0) == ea.OOD_ENERGY  # la entrada se normaliza
    # sin splats: antes 1.0 vs 10.0
    empty = EBMEnergy()
    assert empty.energy(antipode) == ea.OOD_ENERGY == float(ef.E_splats(antipode, None)[0])
    np.testing.assert_array_equal(ebm.energy_batch(np.zeros((0, 4))), np.zeros(0, dtype=np.float32))
    np.testing.assert_array_equal(empty.energy_batch(np.zeros((3, 4))), np.full(3, ea.OOD_ENERGY))


def test_r2_15_energy_normalizes_input_like_mu():
    # Causa 1 de r3a-02: μ se normaliza al insertar pero la energía se evaluaba con el
    # vector crudo (10.0 con el crudo, -0.0 con el normalizado). Misma entrada
    # re-escalada → mismo valor en ambas APIs.
    ebm, ef, splats = _ebm_fixture()
    rng = np.random.default_rng(11)
    for x in rng.standard_normal((6, 4)):
        x = x.astype(np.float32)
        ref_api = ebm.energy(_unit(x))
        ref_fn = float(ef.E_splats(_unit(x), splats)[0])
        for scale in (1.0, 3.7, 0.2, 100.0):
            np.testing.assert_allclose(ebm.energy(x * scale), ref_api, atol=1e-5)
            np.testing.assert_allclose(float(ef.E_splats(x * scale, splats)[0]), ref_fn, atol=1e-5)
        # el gradiente se evalúa en el mismo punto normalizado que la energía
        np.testing.assert_allclose(ebm.energy_gradient(x * 3.7), ebm.energy_gradient(_unit(x)), atol=1e-5)


def test_r2_15_energy_per_splat_matches_canonical():
    # La energía por resultado que usa search() es la canónica aplicada al splat del resultado.
    rng = np.random.default_rng(3)
    mu = _unit(rng.standard_normal((6, 5))).astype(np.float32)
    alpha = np.array([1.0, 2.0, 0.5, 1.0, 10.0, 1.0], dtype=np.float32)
    kappa = np.array([10.0, 12.0, 8.0, 10.0, 5.0, 18.5], dtype=np.float32)
    ebm = EBMEnergy(mu, alpha, kappa)
    x = rng.standard_normal(5).astype(np.float32)
    per = ebm.energy_per_splat(x)
    assert per.shape == (6,)
    for i in range(6):
        np.testing.assert_allclose(
            per[i], ea.splat_energy(x, mu[i : i + 1], alpha[i : i + 1], kappa[i : i + 1]), atol=1e-6
        )
    # E(x) = -log Σᵢ exp(-Eᵢ(x)), con Eᵢ SIN tope (el tope se aplica una sola vez, al total)
    raw_e = ea.splat_energies(x, mu, alpha, kappa)
    total = ebm.energy(x)
    np.testing.assert_allclose(
        min(-np.log(np.sum(np.exp(-raw_e))), ea.OOD_ENERGY), total, atol=1e-6
    )
    assert total <= float(per.min()) + 1e-9




def _exploration_fixture(seed=1, kappa=1.0):
    rng = np.random.default_rng(seed)
    X = _unit(rng.standard_normal((300, 8))).astype(np.float32)
    ebm = EBMEnergy(X[:20].copy(), np.ones(20, dtype=np.float32), np.full(20, kappa, dtype=np.float32))
    expl = EBMExploration(ebm, all_vectors=X, all_ids=[str(i) for i in range(len(X))])
    return expl, rng


@pytest.mark.parametrize("temperature", [1.0, 0.1, 0.01, 1e-4, 1e-6, 0.0])
def test_r2_17_sample_uncertain_accepts_low_and_zero_temperature(temperature):
    # Antes: T <= 0.01 → ValueError 'Fewer non-zero entries in p than size'; T=0 → ídem.
    expl, _ = _exploration_fixture()
    pts = expl.sample_uncertain(k=5, temperature=temperature)
    assert len(pts) == 5
    assert all(np.isfinite(p).all() for p in pts)


def test_r2_17_negative_or_nan_temperature_raises():
    # Antes: T=-1 se convertía en silencio en 1e-10 (max(T, 1e-10)).
    expl, _ = _exploration_fixture()
    with pytest.raises(ValueError, match="temperature"):
        expl.sample_uncertain(k=5, temperature=-1.0)
    with pytest.raises(ValueError, match="temperature"):
        expl.sample_uncertain(k=5, temperature=float("nan"))


@pytest.mark.parametrize("temperature", [0.0, 1e-12, 1e-320])
def test_r2_17_zero_temperature_is_deterministic_limit(monkeypatch, temperature):
    # T → 0: el límite de la distribución ∝ exp(+E/T) son los k candidatos de MAYOR
    # energía (sample_uncertain busca regiones inciertas), sin lanzar.
    expl, rng = _exploration_fixture()
    cands = _unit(rng.standard_normal((60, 8))).astype(np.float32)
    monkeypatch.setattr(expl, "_sample_global", lambda n, dim: [c.copy() for c in cands])
    E = expl.energy_api.energy_batch(cands).astype(np.float64)
    assert len(set(np.round(E, 6))) == len(E)  # sin empates: el top-k es único
    top = set(int(i) for i in np.argsort(-E)[:5])
    got = expl.sample_uncertain(k=5, temperature=temperature)
    idx = {int(np.argmin(((cands - g) ** 2).sum(axis=1))) for g in got}
    assert len(got) == 5 and idx == top
