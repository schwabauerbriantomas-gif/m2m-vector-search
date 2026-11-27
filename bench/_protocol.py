"""Protocolo común de benchmarks (audit r4-17): lo usan TODOS los productores de ``bench/``.

Qué garantiza
-------------
* Datasets sintéticos con semilla fija (``generar``): ``clustered`` (C = round(sqrt(n))
  centros ~ N(0, I_d)*2, cada punto = centro + N(0, I_d)) y ``uniform`` (N(0, I_d));
  todo L2-normalizado y casteado a float32. Las queries salen del mismo generador con
  otra semilla (``generar_queries``).
* Ground truth de recall@10 por fuerza bruta en float64 sobre los MISMOS datos
  (``ground_truth``: producto punto contra todas las filas, opcionalmente restringido
  a los docs que pasan un filtro).
* Calentamiento + repeticiones adaptativas (``medir``): mínimo ``MIN_REPS`` repeticiones y
  ``MIN_TIEMPO_S`` de medición. Se informa latencia por query mediana / p5 / p95 (ms) y
  throughput QPS = queries de la repetición / tiempo de pared de ESA repetición, mediana
  entre repeticiones (nunca 1/latencia media ni N/suma de latencias: audit r4-09).
* RSS pico muestreado cada ``PERIODO_RSS_S`` s (``RssSampler``, lee ``/proc/self/statm``).
* Meta en cada fila (``meta``): commit del árbol que provee ``m2m`` (resuelto desde
  ``m2m.__file__``, NO desde el cwd), si ``src/`` está sucio, versiones, CPU, hilos
  BLAS=1, ``uptime`` antes/después y carga de CPU del sistema durante la medición.
* Lo que no se midió se escribe como ``estado = "n/d"`` con su motivo (``nd``). Nunca un
  número inventado, extrapolado ni copiado de otra corrida.
"""
from __future__ import annotations

import gc
import json
import os
import platform
import resource
import subprocess
import sys
import threading
import time
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

K = 10
SEED_DATOS = 42
SEED_QUERIES = 7
N_QUERIES = 200
BATCHES = (1, 8, 200, 1000)
WARMUP = 1
MIN_REPS = 5
MAX_REPS = 200
MIN_TIEMPO_S = 1.0
PERIODO_RSS_S = 0.005
RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(RAIZ, "benchmarks", "results")
ND = "n/d"
PREFIJO_ID = "doc"

PROTOCOLO = {
    "k": K,
    "n_queries": N_QUERIES,
    "seed_datos": SEED_DATOS,
    "seed_queries": SEED_QUERIES,
    "warmup": WARMUP,
    "min_reps": MIN_REPS,
    "max_reps": MAX_REPS,
    "min_tiempo_s": MIN_TIEMPO_S,
    "dataset_clustered": (
        "C = round(sqrt(n)) centros ~ N(0, I_d)*2; punto = centro + N(0, I_d); "
        "L2-normalizado, float32"
    ),
    "dataset_uniform": "N(0, I_d) L2-normalizado, float32",
    "queries": "mismo generador que el dataset, seed_queries, n_queries",
    "gt": (
        "fuerza bruta float64 (producto punto contra todas las filas, argpartition + "
        "orden); recall@10 = |top-10 interseccion GT| / |GT|"
    ),
    "latencia": "ms por query: mediana / p5 / p95 sobre todas las queries de todas las repeticiones",
    "qps": (
        "queries_por_repeticion / tiempo de pared de la repeticion; mediana entre "
        "repeticiones (no es 1/latencia media)"
    ),
    "batch": "una repeticion = una llamada con B queries; latencia por query = tiempo / B",
    "rss": "pico de RSS del proceso muestreado cada %.3f s (/proc/self/statm)" % PERIODO_RSS_S,
    "carga": (
        "carga_cpu_sistema_pct = CPU ocupada del sistema (/proc/stat, todos los CPUs) "
        "durante la medicion; cpu_propio_pct = (utime+stime del proceso)/pared"
    ),
}


# --------------------------------------------------------------------------- hilos
def requerir_hilos_1() -> None:
    """El protocolo exige BLAS en 1 hilo (con más hilos las ops chicas hacen spin-wait)."""
    for var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"):
        if os.environ.get(var) != "1":
            sys.exit(f"protocolo: corré con {var}=1 (y TMPDIR en un disco local)")


# ------------------------------------------------------------------------ datasets
def n_clusters(n: int) -> int:
    return max(8, int(round(n ** 0.5)))


def _normalizar(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    return (X / np.linalg.norm(X, axis=1, keepdims=True)).astype(np.float32)


def generar(kind: str, n: int, d: int, seed: int = SEED_DATOS) -> Dict[str, np.ndarray]:
    """Dataset con semilla. Devuelve {"X": [n,d] float32, "labels": [n], "centros": [C,d]}."""
    rng = np.random.default_rng(seed)
    if kind == "clustered":
        c = n_clusters(n)
        centros = rng.standard_normal((c, d), dtype=np.float64) * 2.0
        labels = rng.integers(0, c, size=n)
        X = centros[labels] + rng.standard_normal((n, d), dtype=np.float64)
    elif kind == "uniform":
        centros = np.zeros((1, d), dtype=np.float64)
        labels = np.zeros(n, dtype=np.int64)
        X = rng.standard_normal((n, d), dtype=np.float64)
    else:
        raise ValueError(f"kind debe ser 'clustered' o 'uniform' (got {kind!r})")
    return {"X": _normalizar(X), "labels": labels, "centros": centros}


def generar_queries(kind: str, ds: Dict[str, np.ndarray], n: int = N_QUERIES,
                    seed: int = SEED_QUERIES) -> np.ndarray:
    rng = np.random.default_rng(seed)
    d = ds["X"].shape[1]
    if kind == "clustered":
        labels = rng.integers(0, len(ds["centros"]), size=n)
        Q = ds["centros"][labels] + rng.standard_normal((n, d), dtype=np.float64)
    elif kind == "uniform":
        Q = rng.standard_normal((n, d), dtype=np.float64)
    else:
        raise ValueError(f"kind debe ser 'clustered' o 'uniform' (got {kind!r})")
    return _normalizar(Q)


def dataset_fila(kind: str, n: int, d: int) -> Dict[str, object]:
    return {"kind": kind, "n": n, "d": d, "seed": SEED_DATOS}


# ------------------------------------------------------------------ ground truth
def ground_truth(X: np.ndarray, Q: np.ndarray, k: int = K,
                 permitidos: Optional[np.ndarray] = None, chunk: int = 256) -> np.ndarray:
    """Top-k exacto por producto punto en float64. ``permitidos`` (bool[n]) restringe el GT."""
    X64 = np.asarray(X, dtype=np.float64)
    Q64 = np.asarray(Q, dtype=np.float64)
    m = Q64.shape[0]
    gt = np.full((m, k), -1, dtype=np.int64)
    for s in range(0, m, chunk):
        S = Q64[s:s + chunk] @ X64.T
        if permitidos is not None:
            S[:, ~permitidos] = -np.inf
        gt[s:s + chunk] = topk(S, k)
    return gt


def topk(S: np.ndarray, k: int = K) -> np.ndarray:
    """Top-k por fila (mayor = mejor), ordenado. Requiere k <= S.shape[1]."""
    parte = np.argpartition(-S, k - 1, axis=1)[:, :k]
    filas = np.arange(S.shape[0])[:, None]
    orden = np.argsort(-S[filas, parte], axis=1, kind="stable")
    return parte[filas, orden]


def recall_at_k(ids: Sequence[Sequence[int]], gt: np.ndarray, k: int = K) -> float:
    """``ids[i]`` = índices devueltos para la query i (len <= k; -1 = ausente).
    Denominador por query = |GT válido| = min(k, docs que pasan el filtro)."""
    aciertos = total = 0
    for fila, verdadero in zip(ids, gt):
        validos = {int(v) for v in verdadero if int(v) >= 0}
        if not validos:
            continue
        total += len(validos)
        aciertos += len({int(v) for v in fila if int(v) >= 0} & validos)
    return aciertos / total if total else float("nan")


# ------------------------------------------------------------------------ sistema
def rss_mb() -> float:
    with open("/proc/self/statm") as f:
        residente = int(f.read().split()[1])
    return residente * os.sysconf("SC_PAGE_SIZE") / 2 ** 20


class RssSampler(threading.Thread):
    """Muestrea el RSS del proceso cada ``periodo_s``; ``pico`` = máximo visto."""

    def __init__(self, periodo_s: float = PERIODO_RSS_S):
        super().__init__(daemon=True)
        self.periodo_s = periodo_s
        self.pico = 0.0
        self._parar = threading.Event()

    def run(self) -> None:
        while not self._parar.is_set():
            self.pico = max(self.pico, rss_mb())
            self._parar.wait(self.periodo_s)

    def __enter__(self) -> "RssSampler":
        self.pico = rss_mb()
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self._parar.set()
        self.join()
        self.pico = max(self.pico, rss_mb())


def _leer_proc_stat() -> np.ndarray:
    with open("/proc/stat") as f:
        for linea in f:
            if linea.startswith("cpu "):
                return np.array([int(v) for v in linea.split()[1:]], dtype=np.int64)
    return np.zeros(10, dtype=np.int64)


def _busy_frac(antes: np.ndarray, despues: np.ndarray) -> float:
    d = despues - antes
    total = int(d.sum())
    if total <= 0:
        return float("nan")
    idle = int(d[3]) + (int(d[4]) if len(d) > 4 else 0)
    return 100.0 * (1.0 - idle / total)


def _proc_cpu_s() -> float:
    ru = resource.getrusage(resource.RUSAGE_SELF)
    return ru.ru_utime + ru.ru_stime


def _uptime() -> str:
    try:
        return subprocess.run(["uptime"], capture_output=True, text=True, check=False).stdout.strip()
    except OSError:
        return ""


def _loadavg() -> str:
    try:
        with open("/proc/loadavg") as f:
            return f.read().strip()
    except OSError:
        return ""


def _cpu() -> str:
    try:
        with open("/proc/cpuinfo") as f:
            for linea in f:
                if linea.startswith("model name"):
                    return linea.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or ND


def _git(arbol: str, *args: str) -> str:
    try:
        return subprocess.run(["git", "-C", arbol, *args], capture_output=True, text=True,
                              check=False).stdout.strip()
    except OSError:
        return ""


def meta(m2m=None) -> Dict[str, object]:
    """Meta común de la corrida. El commit sale del árbol que provee ``m2m`` (m2m.__file__)."""
    archivo = getattr(m2m, "__file__", None)
    if archivo:
        arbol = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(archivo))))
    else:
        arbol = RAIZ
    commit = _git(arbol, "rev-parse", "HEAD")
    hilos = {v: os.environ.get(v) for v in
             ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")}
    try:
        nproc = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        nproc = os.cpu_count()
    return {
        "commit": commit,
        "commit_corto": commit[:9],
        "arbol": os.path.basename(arbol),
        "src_sucio": bool(_git(arbol, "status", "--porcelain", "--", "src")),
        "m2m_file": os.path.relpath(os.path.abspath(archivo), arbol) if archivo else None,
        "cpu": _cpu(),
        "nproc": nproc,
        "cpu_count": os.cpu_count(),
        "hilos": hilos,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "plataforma": platform.platform(),
        "uptime_antes": _uptime(),
        "loadavg_antes": _loadavg(),
    }


def cerrar_meta(m: Dict[str, object]) -> Dict[str, object]:
    out = dict(m)
    out["uptime_despues"] = _uptime()
    out["loadavg_despues"] = _loadavg()
    return out


# ------------------------------------------------------------------------ medición
def _carga(st0: float, cs0: np.ndarray, t0: float) -> Dict[str, float]:
    pared = time.perf_counter() - t0
    propio = _proc_cpu_s() - st0
    return {
        "carga_cpu_sistema_pct": _busy_frac(cs0, _leer_proc_stat()),
        "cpu_propio_pct": 100.0 * propio / pared if pared > 0 else float("nan"),
        "tiempo_medido_s": pared,
    }


def medir(rep: Callable[[], Optional[Sequence[float]]], queries_por_rep: int,
          warmup: int = WARMUP, min_reps: int = MIN_REPS, max_reps: int = MAX_REPS,
          min_tiempo_s: float = MIN_TIEMPO_S) -> Dict[str, object]:
    """``rep()`` ejecuta UNA repetición y devuelve la lista de latencias por query (s) en
    los paths single, o ``None`` en batch (latencia por query = tiempo / queries_por_rep)."""
    for _ in range(warmup):
        rep()
    uptime_antes = _uptime()
    st0, cs0, t0 = _proc_cpu_s(), _leer_proc_stat(), time.perf_counter()
    rep_s: List[float] = []
    latencias: List[float] = []
    while True:
        t1 = time.perf_counter()
        lat = rep()
        rep_s.append(time.perf_counter() - t1)
        if isinstance(lat, list):
            latencias.extend(lat)
        if len(rep_s) >= min_reps and sum(rep_s) >= min_tiempo_s:
            break
        if len(rep_s) >= max_reps:
            break
    out = _carga(st0, cs0, t0)
    if latencias:
        lat = np.asarray(latencias, dtype=np.float64) * 1e3
    else:
        lat = np.asarray(rep_s, dtype=np.float64) / queries_por_rep * 1e3
    qps = [queries_por_rep / t for t in rep_s]
    out.update({
        "n_repeticiones": len(rep_s),
        "n_muestras_latencia": int(lat.size),
        "queries_por_rep": queries_por_rep,
        "mediana": float(np.median(lat)),
        "p5": float(np.percentile(lat, 5)),
        "p95": float(np.percentile(lat, 95)),
        "qps": float(np.median(qps)),
        "qps_min": float(min(qps)),
        "qps_max": float(max(qps)),
        "uptime_antes": uptime_antes,
        "uptime_despues": _uptime(),
    })
    return out


def medir_single(consulta: Callable[[np.ndarray], object], Q: np.ndarray, **kw) -> Dict[str, object]:
    """Una repetición = recorrer ``Q`` llamando ``consulta(q)`` (latencia por query medida)."""
    perf = time.perf_counter

    def rep():
        lat = []
        for q in Q:
            t0 = perf()
            consulta(q)
            lat.append(perf() - t0)
        return lat

    return medir(rep, len(Q), **kw)


def medir_frio(preparar: Callable[[], None], consulta: Callable[[np.ndarray], object], Q: np.ndarray,
               n: int = 20) -> Dict[str, object]:
    """``n`` evaluaciones FRÍAS: ``preparar()`` (no cronometado) invalida el estado cacheado
    por una vía pública (p. ej. una escritura) y ``consulta(q)`` es la 1ª evaluación después."""
    lat = []
    uptime_antes = _uptime()
    st0, cs0, t0 = _proc_cpu_s(), _leer_proc_stat(), time.perf_counter()
    for j in range(n):
        preparar()
        q = Q[j % len(Q)]
        t1 = time.perf_counter()
        consulta(q)
        lat.append(time.perf_counter() - t1)
    out = _carga(st0, cs0, t0)
    lat_ms = np.asarray(lat, dtype=np.float64) * 1e3
    mediana_ms = float(np.median(lat_ms))
    out.update({
        "n_repeticiones": n,
        "n_muestras_latencia": n,
        "queries_por_rep": 1,
        "mediana": mediana_ms,
        "p5": float(np.percentile(lat_ms, 5)),
        "p95": float(np.percentile(lat_ms, 95)),
        "qps": 1000.0 / mediana_ms if mediana_ms > 0 else float("inf"),
        "qps_definicion": "1 / latencia mediana de evaluaciones frías individuales",
        "uptime_antes": uptime_antes,
        "uptime_despues": _uptime(),
    })
    return out


# --------------------------------------------------------------------- helpers m2m
def construir_db(X: np.ndarray, metadata: Optional[list] = None, **kw):
    """``SimpleVectorDB`` + ``add()`` de todo X con ids "doc<i>". Devuelve (db, build_s)."""
    import time as _time

    import m2m
    db = m2m.SimpleVectorDB(latent_dim=int(X.shape[1]), **kw)
    ids = [f"{PREFIJO_ID}{i}" for i in range(len(X))]
    t0 = _time.perf_counter()
    db.add(ids=ids, vectors=X, metadata=metadata)
    return db, _time.perf_counter() - t0


def doc_int(doc_id) -> int:
    s = str(doc_id)
    return int(s[len(PREFIJO_ID):]) if s.startswith(PREFIJO_ID) else -1


def ids_desde_posiciones(db, posiciones) -> List[List[int]]:
    """Posiciones (splat / ``_splat_id_order``) -> índice entero del doc (-1 si falta)."""
    orden = db._splat_id_order
    out = []
    for fila in np.atleast_2d(np.asarray(posiciones)):
        ids = []
        for p in fila:
            p = int(p)
            ids.append(-1 if p < 0 or p >= len(orden) else doc_int(orden[p]))
        out.append(ids)
    return out


def ids_docresult(lista) -> List[int]:
    return [doc_int(x.id) for x in lista]


def info_indice(db) -> Dict[str, object]:
    info = {}
    try:
        info["n_docs"] = len(db._splat_id_order)
    except Exception:  # pragma: no cover
        pass
    try:
        info["n_probe"] = int(db.n_probe)
    except Exception:  # pragma: no cover
        pass
    try:
        info["n_coarse"] = int(len(db.engine.m2m.splats.engine.coarse_cluster_embeddings))
    except Exception:  # pragma: no cover
        info["n_coarse"] = ND
    return info


# ------------------------------------------------------------------------- salida
def _num(v, decimales: int = 1):
    """Redondea; None o NaN -> None (n/d), nunca un número inventado."""
    if v is None:
        return None
    v = float(v)
    return None if v != v else round(v, decimales)


def resumen_de_tiempos(tiempos_s: Sequence[float], queries_por_rep: Optional[int] = None,
                       **extra) -> Dict[str, object]:
    """Resumen (misma forma que ``medir``) a partir de duraciones por repetición EN SEGUNDOS.
    Para paths que no son búsquedas (build, reopen, escrituras): ``qps`` = queries_por_rep /
    mediana, o None si no aplica."""
    t = np.asarray(tiempos_s, dtype=np.float64)
    med_s = float(np.median(t))
    out = {
        "n_repeticiones": int(t.size),
        "n_muestras_latencia": int(t.size),
        "queries_por_rep": queries_por_rep,
        "mediana": med_s * 1e3,
        "p5": float(np.percentile(t, 5)) * 1e3,
        "p95": float(np.percentile(t, 95)) * 1e3,
        "qps": (queries_por_rep / med_s) if (queries_por_rep and med_s > 0) else None,
        "carga_cpu_sistema_pct": None,
        "cpu_propio_pct": None,
        "uptime_antes": None,
        "uptime_despues": _uptime(),
    }
    out.update(extra)
    return out


def fila(tabla: str, meta_comun: Dict, dataset: Dict, parametro: str, m: Dict,
         recall: Optional[float] = None, rss_pico_mb: Optional[float] = None,
         build_s: Optional[float] = None, **extra) -> Dict:
    out = {
        "tabla": tabla,
        "commit": meta_comun["commit_corto"],
        "dataset": dict(dataset),
        "parametro": parametro,
        "n_repeticiones": m["n_repeticiones"],
        "mediana": round(float(m["mediana"]), 4),
        "p5": round(float(m["p5"]), 4),
        "p95": round(float(m["p95"]), 4),
        "qps": _num(m.get("qps"), 1),
        "recall@10": _num(recall, 4),
        "rss_pico_mb": _num(rss_pico_mb, 1),
        "build_s": _num(build_s, 3),
        "meta": {
            "cpu": meta_comun["cpu"],
            "hilos": meta_comun["hilos"],
            "uptime_antes": m.get("uptime_antes"),
            "uptime_despues": m.get("uptime_despues"),
            "carga_cpu_sistema_pct": _num(m.get("carga_cpu_sistema_pct"), 1),
            "cpu_propio_pct": _num(m.get("cpu_propio_pct"), 1),
        },
    }
    if m.get("qps_definicion"):
        out["qps_definicion"] = m["qps_definicion"]
    out.update(extra)
    return out


def nd(tabla: str, meta_comun: Dict, dataset: Dict, parametro: str, motivo: str) -> Dict:
    return {"tabla": tabla, "commit": meta_comun["commit_corto"], "dataset": dict(dataset),
            "parametro": parametro, "estado": ND, "motivo": motivo}


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (set, tuple)):
        return list(o)
    return str(o)


def escribir(tabla: str, filas: List[Dict], meta_comun: Dict, out_dir: Optional[str] = None,
             **extra) -> str:
    destino_dir = out_dir or RESULTS
    os.makedirs(destino_dir, exist_ok=True)
    doc = {"tabla": tabla, "protocolo": PROTOCOLO, "meta": cerrar_meta(meta_comun),
           "resultados": filas}
    doc.update(extra)
    destino = os.path.join(destino_dir, f"{tabla}.json")
    with open(destino, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=1, ensure_ascii=False, default=_json_default)
    return destino


def resumen(filas: Sequence[Dict]) -> None:
    for f in filas:
        if f.get("estado") == ND:
            print(f"  [n/d] {f['parametro']}: {f['motivo']}")
        else:
            qps = "n/a" if f["qps"] is None else f"{f['qps']:>9.1f}"
            rec = "  n/a" if f["recall@10"] is None else f"{f['recall@10']:.4f}"
            print(f"  {f['parametro']:<44} qps={qps} med={f['mediana']:>9.3f}ms "
                  f"p95={f['p95']:>9.3f}ms R@10={rec} n={f['n_repeticiones']}")


def dir_mb(path: str) -> float:
    total = 0
    for raiz, _, archivos in os.walk(path):
        for a in archivos:
            total += os.path.getsize(os.path.join(raiz, a))
    return total / 2 ** 20


def rss_tras_gc() -> float:
    gc.collect()
    return rss_mb()
