"""
Gate de modulus para la elección de K (vendorizado de swarm11/gate_modulus.py).

Decide, a partir de la trayectoria del módulo combinatorio del grafo kNN bajo
particiones K-Means, la clasificación del corpus (por IC95 del slope fino,
NUNCA por valor puntual — cruza umbrales entre versiones de numpy/BLAS):

- **jerárquico** (IC95 completo <= umbral): sin K* interior; K es un knob
  operacional K = N // target_docs_per_cluster.
- **mono** (IC95 completo > umbral): si existe knee ESTRUCTURAL
  (slope_seg < SLOPE_KNEE_STRUCTURAL), K = knee; si no, K operacional.
  El knee por argmax de desviación sobre curva plana es FANTASMA y se
  descarta (dio 16 vs 44 entre py3.10/py3.12 en el mismo corpus).
- **inconclusivo** (el IC cruza el umbral): se declara honesto, K operacional.

API pública:
    gate_select_k(X, target_docs_per_cluster=25, verdict_cache=None, seed=42)
        -> GateKSelection(k, jerarquico, slope_fino, octava, regla, verdict)

Calibración del umbral slope-fino -0.75 (IC95 disjuntos, N=4600;
swarm11/gate_calibration_N4600.json):
    jerárquico real  -1.46 [-1.98, -1.01]
    mono-escala synth +0.02 [-0.22, +0.15]
    uniforme         -0.21 [-0.77, -0.03]
FP 4.1% / FN 0.3% en la ventana (-1.01, -0.75). NO usar -0.5 (31% FP).

Dependencias: numpy, scipy.sparse (+ scipy.sparse.linalg), scikit-learn.
Costo: ~O(N^2·D) en kNN + 17 escalas × 30 spsolve — index-time únicamente
(~2.1 s @ 659×384, 4 hilos CPU). NUNCA correr en el hot path de query.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from sklearn.cluster import MiniBatchKMeans

# Umbral calibrado (ver docstring del módulo).
SLOPE_FINO_UMBRAL = -0.75
# Un knee solo manda K si es ESTRUCTURAL (caida abrupta del modulo), no si es
# el argmax de desviacion sobre una curva plana (eso es ruido: en el corpus
# real dio knee=16 con slope_seg -1.45 en py3.10 vs k=44 en py3.12).
SLOPE_KNEE_STRUCTURAL = -3.0
# Escalas base del sweep de trayectoria (sub-octavas de 2 aprox.).
BASE_SCALES: Tuple[int, ...] = (
    16, 22, 30, 44, 56, 64, 88, 105, 125, 144, 175, 250, 350, 700, 1400, 2750, 5500,
)


@dataclass
class GateKSelection:
    """Resultado del gate: K elegido + por qué (para auditoría del index-time)."""

    k: int
    jerarquico: bool
    slope_fino: Optional[float]
    octava: Optional[str]
    regla: str
    verdict: Dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Internos (no forman parte de la API pública; names estables para tests)
# ---------------------------------------------------------------------------

def _knn_edges(X: np.ndarray, k: int = 10, chunk: int = 512) -> np.ndarray:
    """Aristas del grafo kNN (pares únicos i<j) vía productos escalares."""
    n = X.shape[0]
    pairs = set()
    for s in range(0, n, chunk):
        S = X[s:s + chunk] @ X.T
        for row in range(S.shape[0]):
            i = s + row
            top = np.argpartition(-S[row], k + 1)[:k + 1]
            for j in top:
                if j != i:
                    pairs.add((min(i, int(j)), max(i, int(j))))
        del S
    return np.array(sorted(pairs), dtype=np.int64)


def _one_pair_energy(gu, gv, gw, n_cl, A, B, idx_map) -> Optional[float]:
    """Energía del corte (A,B): sistema de Laplace en el grafo de buckets."""
    keep = ~(((gu == A) & (gv == B)) | ((gu == B) & (gv == A)))
    eu, ev, ew = gu[keep], gv[keep], gw[keep]
    iu = idx_map[eu]
    iv = idx_map[ev]
    n_int = n_cl - 2
    both = (iu >= 0) & (iv >= 0)
    W = sp.coo_matrix(
        (np.r_[ew[both], ew[both]],
         (np.r_[iu[both], iv[both]], np.r_[iv[both], iu[both]])),
        shape=(n_int, n_int),
    ).tocsc()
    deg = np.zeros(n_int)
    fu = iu >= 0
    fv = iv >= 0
    np.add.at(deg, iu[fu], ew[fu])
    np.add.at(deg, iv[fv], ew[fv])
    L = (sp.diags(deg + 1e-12) - W).tocsc()
    rhs = np.zeros(n_int)
    toA = (eu == A) & (iv >= 0)
    np.add.at(rhs, iv[toA], ew[toA])
    toA2 = (ev == A) & (iu >= 0)
    np.add.at(rhs, iu[toA2], ew[toA2])
    v = spla.spsolve(L, rhs)
    if not np.isfinite(v).all() or v.max() > 1 + 1e-6 or v.min() < -1e-6:
        return None
    val = np.empty(n_cl)
    val[:] = np.nan
    val[np.where(idx_map >= 0)[0]] = v
    val[A] = 1.0
    val[B] = 0.0
    dv = val[gu] - val[gv]
    return float(np.sum(gw * dv * dv))


def _trajectory(
    X: np.ndarray,
    scales: List[int],
    n_pairs: int = 30,
    seed: int = 42,
) -> Dict[int, List[float]]:
    """Median-sample de energías de corte por escala K."""
    r = np.random.default_rng(seed)
    E = _knn_edges(X)
    out: Dict[int, List[float]] = {}
    for K in scales:
        if K >= X.shape[0] // 4:
            continue
        km = MiniBatchKMeans(
            n_clusters=K, batch_size=2048, n_init=3, max_iter=100, random_state=seed
        )
        lab = km.fit_predict(X).astype(np.int64)
        ca, cb = lab[E[:, 0]], lab[E[:, 1]]
        m = ca != cb
        key = np.stack([np.minimum(ca[m], cb[m]), np.maximum(ca[m], cb[m])], 1)
        uniq, cnt = np.unique(key, axis=0, return_counts=True)
        gu, gv, gw = uniq[:, 0], uniq[:, 1], cnt.astype(np.float64)
        es: List[float] = []
        tries = 0
        while len(es) < n_pairs and tries < n_pairs * 4:
            tries += 1
            A, B = r.integers(0, K, 2)
            if A == B:
                continue
            idx_map = np.full(K, -1, np.int64)
            others = np.where((np.arange(K) != A) & (np.arange(K) != B))[0]
            idx_map[others] = np.arange(len(others))
            try:
                e = _one_pair_energy(gu, gv, gw, K, A, B, idx_map)
                if e is not None and np.isfinite(e) and e > 0:
                    es.append(e)
            except Exception:
                continue
        if es:
            out[K] = es
        del km, lab
    return out


def _knee_detector(med: Dict[int, float], N: Optional[int] = None) -> Optional[Dict]:
    """Knee de la curva log-log: segmento con desvío máximo del slope global."""
    ks = sorted(med)
    if len(ks) < 4:
        return None
    # Guard anti-artefacto: descartar el segmento más fino si K > N/10
    # (buckets < ~10 docs -> corte A-B diminuto -> caída espuria del módulo).
    if N is not None:
        ks = [k for k in ks if k <= max(44, N // 10)] or ks
    if len(ks) < 4:
        return None
    glob = (np.log(med[ks[-1]]) - np.log(med[ks[0]])) / (np.log(ks[-1]) - np.log(ks[0]))
    segs = [
        (ks[i], ks[i + 1],
         (np.log(med[ks[i + 1]]) - np.log(med[ks[i]])) / (np.log(ks[i + 1]) - np.log(ks[i])))
        for i in range(len(ks) - 1)
    ]
    dev = max(segs, key=lambda s: abs(s[2] - glob))
    return {
        "knee_K": int(dev[0]),
        "segmento": [int(dev[0]), int(dev[1])],
        "slope_seg": float(dev[2]),
        "slope_global": float(glob),
    }


def _ratio_ci(traj, Khi, Klo, B=10000, seed=42):
    """IC95 bootstrap del ratio de medianas fino->grueso."""
    r = np.random.default_rng(seed)
    a = np.asarray(traj[Khi], float)
    b = np.asarray(traj[Klo], float)
    ia = r.integers(0, len(a), (B, len(a)))
    ib = r.integers(0, len(b), (B, len(b)))
    rats = np.median(a[ia], axis=1) / np.median(b[ib], axis=1)
    return float(np.median(rats)), [float(np.percentile(rats, 2.5)),
                                    float(np.percentile(rats, 97.5))]


def _slope_fino_ci(traj, fine_oct, B=10000, seed=42):
    """IC95 bootstrap del slope de la octava fina (log-log entre 2 escalas).

    El slope puntual de una sub-octava es ruidoso (30 pares por escala) y
    cruza umbrales entre builds/BLAS; la decisión se toma con el intervalo.
    """
    if not fine_oct:
        return None
    try:
        klo_s, khi_s = fine_oct.split("->")
        klo, khi = int(klo_s), int(khi_s)
    except ValueError:
        return None
    if klo not in traj or khi not in traj:
        return None
    a = np.asarray(traj[khi], float)
    b = np.asarray(traj[klo], float)
    if len(a) < 5 or len(b) < 5:
        return None
    lr = np.log(khi / klo)
    r = np.random.default_rng(seed)
    ia = r.integers(0, len(a), (B, len(a)))
    ib = r.integers(0, len(b), (B, len(b)))
    slopes = (np.log(np.median(a[ia], axis=1)) - np.log(np.median(b[ib], axis=1))) / lr
    return [float(np.percentile(slopes, 2.5)), float(np.percentile(slopes, 97.5))]


def _classify_fine(traj, med, fine_oct, seed=42):
    """Clasificación en 3 estados por IC95 del slope fino (función pura).

    - 'jerarquico': IC95 completo del slope fino <= umbral (señal robusta).
    - 'mono': IC95 completo > umbral (plano con margen).
    - 'inconclusivo': el IC cruza el umbral (corpus chico / banda fina
      ruidosa). HONESTO: no se finge clasificación; K va operacional.

    La decisión es por IC y NO por valor puntual: el slope de una única
    sub-octava amplifica el ruido de endpoints (brazo corto, 30 pares por
    escala) y cruza umbrales entre versiones de numpy/BLAS (bug py3.10 vs
    3.12: -0.62 vs -1.33 en el mismo corpus con medianas idénticas).
    """
    osl = _octave_slopes(med)
    slope_fine = osl.get(fine_oct)
    slope_ci = _slope_fino_ci(traj, fine_oct, seed=seed)
    if slope_fine is None or slope_ci is None:
        clasificacion = "inconclusivo"
    elif slope_ci[1] <= SLOPE_FINO_UMBRAL:
        clasificacion = "jerarquico"
    elif slope_ci[0] > SLOPE_FINO_UMBRAL:
        clasificacion = "mono"
    else:
        clasificacion = "inconclusivo"
    return {
        "clasificacion": clasificacion,
        "jerarquico": clasificacion == "jerarquico",
        "slope_fino": slope_fine,
        "IC95": slope_ci,
    }


def _octave_slopes(med: Dict[int, float]) -> Dict[str, float]:
    """Slope log-log entre escalas consecutivas."""
    ks = sorted(med)
    out: Dict[str, float] = {}
    for i in range(len(ks) - 1):
        out[f"{ks[i]}->{ks[i + 1]}"] = float(
            np.log(med[ks[i + 1]] / med[ks[i]]) / np.log(ks[i + 1] / ks[i])
        )
    return out


def _verdict(X: np.ndarray, seed: int = 42) -> Dict:
    """Veredicto completo del gate sobre vectores L2-normalizados."""
    import time as _time

    t0 = _time.time()
    cap = max(12, X.shape[0] // 4)
    scales = [s for s in BASE_SCALES if s <= cap]
    traj = _trajectory(X, scales, seed=seed)
    if len(traj) < 2:
        raise ValueError(
            f"gate: trayectoria degenerada ({len(traj)} escalas con energía) "
            f"para N={X.shape[0]} — corpus demasiado chico o colapsado"
        )
    med = {K: float(np.median(v)) for K, v in traj.items()}
    knee = _knee_detector(med, X.shape[0])
    # growth fino->grueso referencial: 350 si existe, si no un tercio superior
    Kref = 350 if 350 in med else sorted(med)[max(0, len(med) // 3)]
    g, ci = _ratio_ci(traj, max(med), Kref, seed=seed)
    osl = _octave_slopes(med)
    # Señal libre de escala: slope de la (sub)octava más fina con buckets
    # >= ~10 docs. Jerárquico real: slope <= SLOPE_FINO_UMBRAL (mono: ~0).
    # DECISIÓN POR IC, no por valor puntual: el slope puntual sobre una única
    # sub-octava es ruidoso y cruza el umbral entre versiones de numpy/BLAS
    # (bug py3.10 vs 3.12: -0.62 vs -1.33 en el mismo corpus). El bootstrap
    # de pares (misma unidad que el resto del gate) da el error estándar y
    # la decisión se toma si el IC95 completo queda del lado jerárquico.
    ks_sorted = sorted(med)
    fine_oct = None
    for i in range(len(ks_sorted) - 1):
        if ks_sorted[i] <= max(44, X.shape[0] // 10):
            fine_oct = f"{ks_sorted[i]}->{ks_sorted[i + 1]}"
    slope_fine = osl.get(fine_oct)
    cls = _classify_fine(traj, med, fine_oct, seed=seed)
    clasificacion = cls["clasificacion"]
    slope_fine = cls["slope_fino"]
    slope_ci = cls["IC95"]
    jerarquico = cls["jerarquico"]
    res = {
        "N": int(X.shape[0]),
        "D": int(X.shape[1]),
        "medianas": med,
        "octave_slopes": osl,
        "slope_octava_fina": {"octava": fine_oct, "slope": slope_fine, "IC95": slope_ci},
        "growth_fino_grueso": {"par": [int(max(med)), int(Kref)], "ratio": g, "IC95": ci},
        "clasificacion": clasificacion,
        "jerarquico": jerarquico,
        "knee": knee,
        "regla": (
            "NO buscar K* interior: corpus jerárquico de curva lisa, barrer K o router jerárquico"
            if jerarquico
            else f"corpus mono-escala o knee/artefacto: knee={knee['knee_K'] if knee else '?'} "
                 f"slope_seg={knee['slope_seg']:.2f} -> revisar bracket; codebook K ~ bracket "
                 f"del knee si es real"
        ),
        "t_s": round(_time.time() - t0, 1),
    }
    return res


# ---------------------------------------------------------------------------
# API pública
# ---------------------------------------------------------------------------

def gate_select_k(
    X: np.ndarray,
    target_docs_per_cluster: int = 25,
    verdict_cache: Optional[str] = None,
    seed: int = 42,
) -> GateKSelection:
    """Elegir K para el codebook según el veredicto del gate de modulus.

    Args:
        X: embeddings (N, D). Se L2-normalizan internamente (copia).
        target_docs_per_cluster: K operacional para corpus jerárquicos
            (k = N // target). Ignorado en mono-escala (manda el knee).
        verdict_cache: path JSON para cachear el veredicto (index-time, una
            vez por corpus). Si existe y coincide (N, D, target, seed), se
            reutiliza; si no, se computa y guarda.
        seed: semilla reproducible (KMeans + pares aleatorios + bootstrap).

    Returns:
        GateKSelection con k elegido, jerarquico, slope_fino, octava, regla
        y el veredicto completo (medianas, slopes, knee, IC).
    """
    X = np.ascontiguousarray(np.asarray(X, dtype=np.float32))
    if X.ndim != 2 or X.shape[0] < 64:
        raise ValueError(f"gate: necesito X (N>=64, D), recibí {X.shape}")
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    Xn = X / np.maximum(norms, 1e-12)

    verdict: Optional[Dict] = None
    cache_ok = False
    if verdict_cache:
        import json as _json
        import os as _os

        if _os.path.exists(verdict_cache):
            try:
                cached = _json.load(open(verdict_cache))
                cv = cached.get("verdict", cached)
                if (
                    cached.get("target_docs_per_cluster") == int(target_docs_per_cluster)
                    and cached.get("seed") == int(seed)
                    and cv.get("N") == int(Xn.shape[0])
                    and cv.get("D") == int(Xn.shape[1])
                ):
                    verdict = cv
                    cache_ok = True
            except Exception:
                verdict = None  # cache corrupta -> recomputar

    if verdict is None:
        verdict = _verdict(Xn, seed=seed)
        if verdict_cache:
            import json as _json

            payload = {
                "target_docs_per_cluster": int(target_docs_per_cluster),
                "seed": int(seed),
                "verdict": verdict,
            }
            with open(verdict_cache, "w") as fh:
                _json.dump(payload, fh, indent=1)

    n = Xn.shape[0]
    knee = verdict.get("knee")
    knee_structural = bool(knee and knee["slope_seg"] < SLOPE_KNEE_STRUCTURAL)
    if verdict["jerarquico"]:
        k = n // max(1, int(target_docs_per_cluster))
        regla = (
            f"jerárquico (slope_fino {verdict['slope_octava_fina']['slope']:.2f} "
            f"IC95 {verdict['slope_octava_fina'].get('IC95')} <= {SLOPE_FINO_UMBRAL} "
            f"en octava {verdict['slope_octava_fina']['octava']}): "
            f"K operacional = N//{target_docs_per_cluster} = {k}; NO optimizar K"
        )
    elif knee_structural:
        k = int(knee["knee_K"])
        regla = (
            f"mono-escala con knee ESTRUCTURAL: K ~ knee {k} "
            f"(slope_seg {knee['slope_seg']:.2f} < {SLOPE_KNEE_STRUCTURAL})"
        )
    else:
        k = n // max(1, int(target_docs_per_cluster))
        if knee:
            knee_nota = (
                f"knee={knee['knee_K']} descartado como fantasma "
                f"(slope_seg {knee['slope_seg']:.2f} >= {SLOPE_KNEE_STRUCTURAL})"
            )
        else:
            knee_nota = "sin knee detectado"
        regla = (
            f"corpus mono-escala: {knee_nota} -> K operacional = "
            f"N//{target_docs_per_cluster} = {k}"
        )

    # Saneo: KMeans necesita k >= 2 y buckets no vacíos masivamente.
    k = int(max(2, min(k, max(2, n // 4))))

    return GateKSelection(
        k=k,
        jerarquico=bool(verdict["jerarquico"]),
        slope_fino=verdict["slope_octava_fina"]["slope"],
        octava=verdict["slope_octava_fina"]["octava"],
        regla=regla,
        verdict=verdict,
    )
