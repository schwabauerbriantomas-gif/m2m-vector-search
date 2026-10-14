"""Tests del gate de modulus vendorizado (src/m2m/gate_modulus.py) y su
integración en HRM2Engine (k_selection='gate').

Generadores sintéticos calibrados empíricamente (probe v5-v8, scratch):
- mono: 32 conos de tema + 20% background — a N=2000 la octava fina
  (175->250, ~5.5-7.8x el numero de clusters) cae en la meseta saturada
  -> firma mono-escala (slope_fino > -0.75, como synth64 de la calibración).
- hier: arbol 12 temas x 5 subtemas + micro-grupos near-dup en hojas,
  ruido conectivo sig=0.30 -> curva lisa sin knee -> jerarquico=True
  (slope_fino <= -0.75, como el corpus real: -1.46).

El gate cuesta ~13-15s por corpus a N=2000 (index-time). Marcar -m slow si
la suite exige velocidad; por ahora viven en la suite normal.
"""

import json
import os

import numpy as np
import pytest

from m2m.gate_modulus import (
    GateKSelection,
    SLOPE_FINO_UMBRAL,
    _classify_fine,
    gate_select_k,
)
from m2m.hrm2_engine import HRM2Engine

DATA_NPY = "/tmp/m2mvs/embeddings.npy"
GOLDEN_K = os.path.join(os.path.dirname(__file__), "_gate_golden_k.json")

N_SYNTH = 2000
D_SYNTH = 384


# ---------------------------------------------------------------------------
# Generadores (congelados: NO retocar sin re-validar el gate)
# ---------------------------------------------------------------------------

def make_mono(n=N_SYNTH, d=D_SYNTH, n_clu=32, seed=7, bg=0.20, sig=0.35):
    """Corpus mono-escala: n_clu clusters equilibrados + background difuso."""
    r = np.random.default_rng(seed)
    C = r.normal(size=(n_clu, d)).astype(np.float32)
    C /= np.linalg.norm(C, axis=1, keepdims=True)
    nb = int(n * bg)
    nc = n - nb
    pts = np.empty((n, d), dtype=np.float32)
    idx = r.choice(n_clu, size=nc)
    pts[:nc] = C[idx] + sig * r.normal(size=(nc, d)).astype(np.float32)
    pts[nc:] = r.normal(size=(nb, d)).astype(np.float32)
    pts /= (np.linalg.norm(pts, axis=1, keepdims=True) + 1e-12)
    return pts


def make_hier(n=N_SYNTH, d=D_SYNTH, seed=41, sig=0.30):
    """Corpus jerarquico: temas -> subtemas -> micro-grupos near-dup,
    puntos en TODOS los niveles (docs generales, de subtema y de hoja)."""
    r = np.random.default_rng(seed)
    n_sup, n_sub = 12, 5
    pts = []
    for s_ in range(n_sup):
        S = r.normal(size=(d,)).astype(np.float32)
        S /= np.linalg.norm(S)
        m_sup = max(20, n // (n_sup * 6))
        pts.append(S + sig * r.normal(size=(m_sup, d)).astype(np.float32))
        for b_ in range(n_sub):
            M = S + 0.5 * (r.normal(size=(d,)).astype(np.float32) / np.sqrt(d))
            m_sub = max(15, n // (n_sup * n_sub * 6))
            pts.append(M + sig * r.normal(size=(m_sub, d)).astype(np.float32))
            for g_ in range(7):
                G = M + 0.25 * (r.normal(size=(d,)).astype(np.float32) / np.sqrt(d))
                m_g = int(r.integers(2, 5))
                pts.append(G + 0.5 * sig * r.normal(size=(m_g, d)).astype(np.float32))
    X = np.vstack(pts)[:n]
    X /= (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    return X.astype(np.float32)


# ---------------------------------------------------------------------------
# (a) clasificación: regla de decisión del slope fino (función pura, sin
#     KMeans -> determinística en cualquier versión de numpy/BLAS)
# ---------------------------------------------------------------------------

def _traj_consta(med_dict, eps=0.0, n_pairs=30):
    """Trayectoria sintética con mediana EXACTA por escala.

    eps=0 -> valores constantes (IC degenerado de ancho cero).
    eps>0 -> valores alternos m*(1±eps) (mediana m, IC de ancho > 0).
    """
    out = {}
    for K, m in med_dict.items():
        if eps == 0:
            vals = [m] * n_pairs
        else:
            vals = [m * (1 + eps), m * (1 - eps)] * (n_pairs // 2)
        out[K] = [float(v) for v in vals]
    return out


def test_classify_flat_plateau_is_mono():
    # curva PLANA en toda la banda fina -> slope 0 con IC [0,0] -> mono
    med = {44: 100.0, 88: 100.0, 175: 100.0, 250: 100.0}
    traj = _traj_consta(med)
    cls = _classify_fine(traj, med, "175->250", seed=42)
    assert cls["clasificacion"] == "mono"
    assert cls["jerarquico"] is False
    assert cls["slope_fino"] == 0.0
    assert cls["IC95"] == [0.0, 0.0]


def test_classify_powerlaw_is_hierarchical():
    # ley de potencias steep en la banda fina -> slope ~ -0.96, IC estrecho
    # -> jerárquico (el IC completo queda del lado del umbral)
    med = {44: 800.0, 88: 400.0, 175: 200.0, 250: 142.0}
    traj = _traj_consta(med)
    cls = _classify_fine(traj, med, "175->250", seed=42)
    assert cls["clasificacion"] == "jerarquico"
    assert cls["jerarquico"] is True
    assert cls["IC95"][1] <= SLOPE_FINO_UMBRAL


def test_classify_borderline_is_inconclusive():
    # slope puntual ~ -0.8 (del lado jerárquico del umbral) PERO con dispersión
    # ±25% por escala -> el IC cruza -0.75 -> inconclusivo (honesto)
    m_b = 100.0
    m_a = m_b * (2 ** -0.8)
    med = {88: m_b, 175: m_a}
    traj = _traj_consta(med, eps=0.25)
    cls = _classify_fine(traj, med, "88->175", seed=42)
    assert cls["slope_fino"] is not None and cls["slope_fino"] < SLOPE_FINO_UMBRAL
    assert cls["IC95"][0] <= SLOPE_FINO_UMBRAL < cls["IC95"][1]
    assert cls["clasificacion"] == "inconclusivo"
    assert cls["jerarquico"] is False


# ---------------------------------------------------------------------------
# (b) selección de K: matriz de decisión sobre veredictos INYECTADOS vía
#     verdict_cache (sin física -> determinística; la clasificación ya está
#     testeada arriba y contra anclas reales en swarm11)
# ---------------------------------------------------------------------------

def _fake_X(n=64, d=8, seed=0):
    r = np.random.default_rng(seed)
    return r.normal(size=(n, d)).astype(np.float32)


def _fake_verdict(tmp_path, N, D, jerarquico, knee, slope=-1.2, ic=(-1.5, -0.9)):
    verdict = {
        "N": N, "D": D,
        "medianas": {16: 10.0, 22: 9.0},
        "octave_slopes": {"16->22": -0.15},
        "slope_octava_fina": {"octava": "16->22", "slope": slope, "IC95": list(ic)},
        "growth_fino_grueso": {"par": [22, 16], "ratio": 0.9, "IC95": [0.8, 1.0]},
        "clasificacion": "jerarquico" if jerarquico else "mono",
        "jerarquico": jerarquico,
        "knee": knee,
        "regla": "fake",
        "t_s": 0.1,
    }
    path = tmp_path / "verdict_cache.json"
    path.write_text(json.dumps({
        "target_docs_per_cluster": 25, "seed": 42, "verdict": verdict,
    }))
    return str(path)


def test_selection_hierarchical_uses_operational_k(tmp_path):
    X = _fake_X()
    cache = _fake_verdict(tmp_path, len(X), X.shape[1], jerarquico=True,
                          knee={"knee_K": 10, "slope_seg": -5.0})
    sel = gate_select_k(X, target_docs_per_cluster=25, verdict_cache=cache, seed=42)
    # jerárquico: K operacional N//target aunque exista knee estructural
    assert sel.jerarquico is True
    assert sel.k == len(X) // 25


def test_selection_structural_knee_wins_on_mono(tmp_path):
    X = _fake_X()
    cache = _fake_verdict(tmp_path, len(X), X.shape[1], jerarquico=False,
                          knee={"knee_K": 10, "slope_seg": -5.0})
    sel = gate_select_k(X, target_docs_per_cluster=25, verdict_cache=cache, seed=42)
    assert sel.jerarquico is False
    # mono con knee ESTRUCTURAL (slope_seg < -3): el knee manda
    assert sel.k == 10


def test_selection_phantom_knee_discarded_on_mono(tmp_path):
    X = _fake_X()
    cache = _fake_verdict(tmp_path, len(X), X.shape[1], jerarquico=False,
                          knee={"knee_K": 10, "slope_seg": -1.0})
    sel = gate_select_k(X, target_docs_per_cluster=25, verdict_cache=cache, seed=42)
    assert sel.jerarquico is False
    # knee fantasma (argmax de desviación sobre curva plana): descartado,
    # K operacional (el knee 16-vs-44 entre versiones era exactamente esto)
    assert sel.k == len(X) // 25


def test_selection_no_knee_operational_k(tmp_path):
    X = _fake_X()
    cache = _fake_verdict(tmp_path, len(X), X.shape[1], jerarquico=False, knee=None)
    sel = gate_select_k(X, target_docs_per_cluster=25, verdict_cache=cache, seed=42)
    assert sel.jerarquico is False
    assert sel.k == len(X) // 25


# ---------------------------------------------------------------------------
# (c) integración: HRM2Engine con k_selection='gate' sobre el dataset real
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not os.path.exists(DATA_NPY), reason="dataset de validación ausente")
def test_engine_gate_integration_recall():
    X = np.load(DATA_NPY).astype(np.float32)
    Xn = X / np.linalg.norm(X, axis=1, keepdims=True)
    n = Xn.shape[0]

    engine = HRM2Engine(
        metric="cosine", batch_size=1000, random_state=42, k_selection="gate"
    )
    engine.index(precomputed_embeddings=Xn)

    # El gate sobre este corpus (validado en swarm11): mono-escala, SIN knee
    # estructural (el knee por argmax de desviacion sobre curva plana es
    # fantasma: 16 en py3.10 vs 44 en py3.12 en el mismo corpus) -> K
    # OPERACIONAL = N//25 = 26. El criterio se pinnea, no el numero fragil.
    sel = engine._gate_selection
    assert sel is not None
    assert sel.jerarquico is False
    assert sel.slope_fino is not None and sel.slope_fino > -0.75  # plano en la octava fina
    assert sel.k == n // 25
    assert sel.k == 26
    assert engine.n_coarse == 26

    # recall@10 vs brute-force con config default de búsqueda (n_probe=5, lod=2)
    rng = np.random.default_rng(0)
    qidx = rng.choice(n, 100, replace=False)
    hits = 0
    for qi in qidx:
        q = Xn[qi]
        gt = set(np.argsort(-(Xn @ q))[:10].tolist())  # top-10 INCLUYENDO el propio punto
        res = engine.query(q, k=10, lod=2)
        got = {r[0] for r in res}
        hits += len(gt & got)
    recall = hits / (100 * 10)
    assert recall >= 0.95, f"recall@10={recall:.3f} < 0.95"


# ---------------------------------------------------------------------------
# (d) regresión: silhouette produce EXACTAMENTE el mismo K que antes del cambio
# ---------------------------------------------------------------------------

def _golden():
    with open(GOLDEN_K) as fh:
        return json.load(fh)


@pytest.mark.skipif(not os.path.exists(DATA_NPY), reason="dataset de validación ausente")
def test_regression_silhouette_k_real():
    X = np.load(DATA_NPY).astype(np.float32)
    eng = HRM2Engine(metric="cosine", batch_size=1000, k_selection="silhouette")
    k_new = eng._auto_detect_k(X)
    assert k_new == _golden()["real_659"]


def test_regression_silhouette_k_synthetic():
    # mismo generador con que se capturó el golden pre-cambio
    rng = np.random.default_rng(7)
    C, D, npc = 64, 384, 32
    centers = rng.normal(size=(C, D))
    Xs = np.vstack([centers[c] + 0.28 * rng.normal(size=(npc, D)) for c in range(C)])
    Xs = (Xs / np.linalg.norm(Xs, axis=1, keepdims=True)).astype(np.float32)
    eng = HRM2Engine(metric="cosine", batch_size=1000, k_selection="silhouette")
    k_new = eng._auto_detect_k(Xs)
    assert k_new == _golden()["mono_2048"]


def test_k_selection_validation_and_compat():
    # valor inválido -> error claro
    with pytest.raises(ValueError):
        HRM2Engine(k_selection="bogus")
    # auto_k=True legacy -> silhouette (backward-compat)
    eng = HRM2Engine(auto_k=True)
    assert eng.k_selection == "silhouette"
    # default -> fixed, sin auto-detección (comportamiento previo sin auto_k)
    eng2 = HRM2Engine()
    assert eng2.k_selection == "fixed"
