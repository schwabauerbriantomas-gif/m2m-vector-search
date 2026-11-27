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

Trayectoria degenerada (menos de 2 escalas con energía, corpus muy chico):
NO lanza — devuelve un veredicto 'inconclusivo' con degenerado=True y K
operacional [audit r2-10].

Escalas descartadas [audit r2-12]: si el grafo de clústeres queda
desconectado (kNN k=10 no conecta clústeres bien separados -> Laplaciana
singular -> sin energía), la escala no entra en la trayectoria. El veredicto
lo reporta en verdict['escalas'] / verdict['dropped_scales'] y, si se perdió
alguna escala de la región fina, NO se fija la clasificación: pasa a
'inconclusivo' (clasificacion_sin_escalas_perdidas guarda la que habría
salido). La estabilidad frente a la semilla se mide sólo si se pide
(gate_select_k(..., stability_seeds>0)) y se reporta en
verdict['estabilidad_semilla'].

API pública:
    gate_select_k(X, target_docs_per_cluster=25, verdict_cache=None, seed=42,
                  stability_seeds=0)
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

import hashlib
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
# Versión del algoritmo/clave del cache de veredictos [audit r2-11]: los
# caches escritos por versiones anteriores (sin fingerprint) no se reusan.
_CACHE_ALGO = 2


@dataclass
class GateKSelection:
    """Resultado del gate: K elegido + por qué (para auditoría del index-time)."""

    k: int
    jerarquico: bool
    slope_fino: Optional[float]
    octava: Optional[str]
    regla: str
    verdict: Dict = field(default_factory=dict)
    # [audit r2-12] Escalas evaluadas que quedaron fuera de la trayectoria
    # (sin energía: grafo de clústeres desconectado). == verdict['dropped_scales'].
    dropped_scales: List[int] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Internos (no forman parte de la API pública; names estables para tests)
# ---------------------------------------------------------------------------

def _data_fingerprint(X: np.ndarray) -> str:
    """SHA-1 del contenido de X (float32 C-contiguo, por chunks) + shape.

    Clave del cache de veredictos [audit r2-11]: la validez se fijaba sólo en
    (N, D, target, seed) y reusaba el veredicto de un corpus para otro
    distinto de igual tamaño.
    """
    X = np.ascontiguousarray(X, dtype=np.float32)
    # Clave de cache, no criptográfico; el digest NO cambia (rompería los caches
    # ya escritos) (anotado para bandit B324) [audit P15].
    h = hashlib.sha1(usedforsecurity=False)
    h.update(f"{X.shape[0]}x{X.shape[1]} float32;".encode())
    for s in range(0, X.shape[0], 65536):
        h.update(X[s:s + 65536].tobytes())
    return h.hexdigest()


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
    report: Optional[Dict[int, Dict]] = None,
) -> Dict[int, List[float]]:
    """Median-sample de energías de corte por escala K.

    `report` (opcional, [audit r2-12]): dict que se llena con el destino de
    CADA escala pedida — por K: {'estado': 'ok'|'descartada'|'omitida',
    'motivo': str, 'n_pares': int, 'pares_descartados': int}. Antes las
    escalas sin energía (grafo de clústeres desconectado) se descartaban en
    silencio y el veredicto no lo decía.
    """
    r = np.random.default_rng(seed)
    E = _knn_edges(X)
    out: Dict[int, List[float]] = {}
    n_quarter = X.shape[0] // 4
    for K in scales:
        if K >= n_quarter:
            if report is not None:
                report[K] = {"estado": "omitida", "motivo": f"K >= N/4 ({n_quarter})"}
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
        n_desc = 0  # pares (A != B) sin energía finita > 0
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
                else:
                    n_desc += 1
            except Exception:
                n_desc += 1
                continue
        if report is not None:
            if es:
                report[K] = {"estado": "ok", "motivo": "", "n_pares": len(es),
                             "pares_descartados": n_desc}
            else:
                report[K] = {
                    "estado": "descartada",
                    "motivo": ("sin energía: ningún par (A,B) de clústeres con corte "
                               "finito (grafo de clústeres desconectado: el kNN k=10 "
                               "no conecta clústeres bien separados)"),
                    "n_pares": 0,
                    "pares_descartados": n_desc,
                }
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


def _escalas_block(
    scales: List[int], traj: Dict[int, List[float]], report: Dict[int, Dict]
) -> Dict:
    """Destino de cada escala del sweep [audit r2-12] (JSON-safe: claves str)."""
    evaluadas = [int(K) for K in scales if report.get(K, {}).get("estado") != "omitida"]
    descartadas = [int(K) for K in evaluadas if K not in traj]
    return {
        "evaluadas": evaluadas,
        "con_energia": sorted(int(K) for K in traj),
        # Per-scale detail: estado/motivo/n_pares/pares_descartados (los pares
        # descartados en escalas 'ok' indican desconexión parcial).
        "detalle": {str(int(K)): dict(r) for K, r in report.items()},
        "descartadas": {str(K): dict(report[K]) for K in descartadas},
        "omitidas_K_ge_N4": sorted(int(K) for K, r in report.items()
                                   if r.get("estado") == "omitida"),
        "finas_perdidas": [],
    }


def _verdict(X: np.ndarray, seed: int = 42) -> Dict:
    """Veredicto completo del gate sobre vectores L2-normalizados.

    Con trayectoria degenerada (< 2 escalas con energía) no lanza: devuelve
    'inconclusivo' con degenerado=True [audit r2-10; antes ValueError].
    """
    import time as _time

    t0 = _time.time()
    N = int(X.shape[0])
    cap = max(12, N // 4)
    scales = [s for s in BASE_SCALES if s <= cap]
    report: Dict[int, Dict] = {}
    traj = _trajectory(X, scales, seed=seed, report=report)
    escalas = _escalas_block(scales, traj, report)
    descartadas = [int(K) for K in escalas["descartadas"]]
    if len(traj) < 2:
        return {
            "N": N,
            "D": int(X.shape[1]),
            "medianas": {int(K): float(np.median(v)) for K, v in traj.items()},
            "octave_slopes": {},
            "slope_octava_fina": {"octava": None, "slope": None, "IC95": None},
            "growth_fino_grueso": {"par": [None, None], "ratio": None, "IC95": [None, None]},
            "clasificacion": "inconclusivo",
            "jerarquico": False,
            "knee": None,
            "degenerado": True,
            "n_escalas_con_energia": len(traj),
            "escalas": escalas,
            "dropped_scales": descartadas,
            "regla": (
                f"trayectoria degenerada ({len(traj)} escalas con energía) para "
                f"N={N} — corpus demasiado chico o colapsado -> inconclusivo, "
                f"K operacional"
            ),
            "t_s": round(_time.time() - t0, 1),
        }
    med = {K: float(np.median(v)) for K, v in traj.items()}
    knee = _knee_detector(med, N)
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
    fine_cap = max(44, N // 10)
    fine_oct = None
    for i in range(len(ks_sorted) - 1):
        if ks_sorted[i] <= fine_cap:
            fine_oct = f"{ks_sorted[i]}->{ks_sorted[i + 1]}"
    slope_fine = osl.get(fine_oct)
    cls = _classify_fine(traj, med, fine_oct, seed=seed)
    clasificacion = cls["clasificacion"]
    clasificacion_cruda = clasificacion
    slope_fine = cls["slope_fino"]
    slope_ci = cls["IC95"]
    jerarquico = cls["jerarquico"]
    # [audit r2-12] Escalas de la región fina perdidas por grafo desconectado:
    # la 'octava fina' se estaría midiendo por encima de la escala natural
    # (p.ej. hojas nítidas: 350->700 en vez de por debajo de n_hojas). La
    # clasificación no es confiable -> inconclusivo, guardando la cruda.
    finas_perdidas = [K for K in descartadas if K <= fine_cap]
    escalas["finas_perdidas"] = finas_perdidas
    if finas_perdidas and clasificacion != "inconclusivo":
        clasificacion = "inconclusivo"
        jerarquico = False
    nota_perdidas = (
        f" | inconclusivo: {len(finas_perdidas)} escalas finas perdidas por grafo de "
        f"clústeres desconectado {finas_perdidas} (clasificación sin esas "
        f"pérdidas: '{clasificacion_cruda}')"
        if finas_perdidas else ""
    )
    res = {
        "N": N,
        "D": int(X.shape[1]),
        "medianas": med,
        "octave_slopes": osl,
        "slope_octava_fina": {"octava": fine_oct, "slope": slope_fine, "IC95": slope_ci},
        "growth_fino_grueso": {"par": [int(max(med)), int(Kref)], "ratio": g, "IC95": ci},
        "clasificacion": clasificacion,
        "jerarquico": jerarquico,
        "knee": knee,
        "degenerado": False,
        "n_escalas_con_energia": len(traj),
        "escalas": escalas,
        "dropped_scales": descartadas,
        "clasificacion_sin_escalas_perdidas": clasificacion_cruda,
        "regla": (
            "NO buscar K* interior: corpus jerárquico de curva lisa, barrer K o router jerárquico"
            if jerarquico
            else f"corpus mono-escala o knee/artefacto: knee={knee['knee_K'] if knee else '?'} "
                 # [audit r2-10] knee es None con < 4 escalas (N <= ~177):
                 # este formato se evaluaba sin guardia -> TypeError.
                 f"slope_seg={(knee['slope_seg'] if knee else float('nan')):.2f} -> revisar "
                 f"bracket; codebook K ~ bracket del knee si es real" + nota_perdidas
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
    stability_seeds: int = 0,
) -> GateKSelection:
    """Elegir K para el codebook según el veredicto del gate de modulus.

    Args:
        X: embeddings (N, D). Se L2-normalizan internamente (copia).
        target_docs_per_cluster: K operacional para corpus jerárquicos
            (k = N // target). Ignorado en mono-escala (manda el knee).
        verdict_cache: path JSON para cachear el veredicto (index-time, una
            vez por corpus). Se reutiliza sólo si coincide (N, D, target,
            seed) Y el fingerprint del contenido (SHA-1 de X) y la versión
            del algoritmo [audit r2-11: antes la clave era sólo (N, D, target,
            seed) y reusaba el veredicto de un corpus para otro distinto];
            si no, se computa y guarda.
        seed: semilla reproducible (KMeans + pares aleatorios + bootstrap).
        stability_seeds: semillas ADICIONALES para medir la estabilidad de la
            clasificación frente a la semilla [audit r2-12]. Cada una cuesta
            una trayectoria completa; 0 (default) = no se evalúa y el
            veredicto lo dice ('estabilidad_semilla.evaluada' = False).

    Returns:
        GateKSelection con k elegido, jerarquico, slope_fino, octava, regla
        y el veredicto completo (medianas, slopes, knee, IC, escalas
        descartadas, estabilidad por semilla).

    Raises:
        ValueError: sólo si X no es 2-D o N < 64 (entrada inválida). Con
            trayectoria degenerada NO lanza: devuelve 'inconclusivo'.
    """
    X = np.ascontiguousarray(np.asarray(X, dtype=np.float32))
    if X.ndim != 2 or X.shape[0] < 64:
        raise ValueError(f"gate: necesito X (N>=64, D), recibí {X.shape}")
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    Xn = X / np.maximum(norms, 1e-12)
    fingerprint = _data_fingerprint(X)

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
                    # [audit r2-11] mismo corpus (fingerprint) y mismo algoritmo
                    and cached.get("fingerprint") == fingerprint
                    and cached.get("algo") == _CACHE_ALGO
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
                "fingerprint": fingerprint,
                "algo": _CACHE_ALGO,
                "verdict": verdict,
            }
            with open(verdict_cache, "w") as fh:
                _json.dump(payload, fh, indent=1)

    verdict["fingerprint"] = fingerprint

    # [audit r2-12] Estabilidad frente a la semilla: tasa de flip de la
    # clasificación en `stability_seeds` semillas adicionales (opt-in).
    clases = {str(int(seed)): verdict["clasificacion"]}
    estabilidad = {
        "evaluada": False,
        "n_seeds": 1,
        "semilla": int(seed),
        "clasificaciones": clases,
        "flip_rate": None,
        "estable": None,
        "nota": ("no evaluada (stability_seeds=0): cerca del umbral la "
                 "clasificación puede cambiar con la semilla"),
    }
    n_alt = int(stability_seeds)
    if n_alt > 0:
        for s in range(int(seed) + 1, int(seed) + n_alt + 1):
            clases[str(s)] = _verdict(Xn, seed=s)["clasificacion"]
        n_flip = sum(c != clases[str(int(seed))] for s, c in clases.items()
                     if s != str(int(seed)))
        estabilidad = {
            "evaluada": True,
            "n_seeds": 1 + n_alt,
            "semilla": int(seed),
            "clasificaciones": clases,
            "flip_rate": n_flip / n_alt,
            "estable": n_flip == 0,
        }
    verdict["estabilidad_semilla"] = estabilidad

    n = Xn.shape[0]
    knee = verdict.get("knee")
    knee_structural = bool(knee and knee["slope_seg"] < SLOPE_KNEE_STRUCTURAL)
    finas_perdidas = list(verdict.get("escalas", {}).get("finas_perdidas", []))
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

    # [audit r2-10/r2-12] Avisos honestos en la regla operativa.
    if verdict.get("degenerado"):
        regla = (
            f"trayectoria degenerada ({verdict.get('n_escalas_con_energia', 0)} escalas "
            f"con energía para N={n}): inconclusivo -> K operacional = {k}"
        )
    avisos = []
    if finas_perdidas:
        avisos.append(
            f"inconclusivo: escalas finas perdidas por grafo de clústeres "
            f"desconectado {finas_perdidas} (veredicto.verdict.escalas)"
        )
    if estabilidad["evaluada"] and not estabilidad["estable"]:
        avisos.append(
            f"semilla inestable: flip_rate {estabilidad['flip_rate']:.2f} "
            f"{estabilidad['clasificaciones']}"
        )
    if avisos:
        regla = f"[{' | '.join(avisos)}] {regla}"

    return GateKSelection(
        k=k,
        jerarquico=bool(verdict["jerarquico"]),
        slope_fino=verdict["slope_octava_fina"]["slope"],
        octava=verdict["slope_octava_fina"]["octava"],
        regla=regla,
        verdict=verdict,
        dropped_scales=[int(s) for s in verdict.get("dropped_scales", [])],
    )
