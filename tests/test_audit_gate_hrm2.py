"""Tests de regresión del pack P11a — gate de modulus + HRM2 + guardia GPU.

Ítems (~/swarm_m2m_vs_audit/reports/r2.json): r2-10, r2-11, r2-12, r2-18,
r2-19, r2-20. Un test por criterio de aceptación del BACKLOG (P11a). Sondas
base del auditor (solo lectura): ~/swarm_m2m_vs_audit/work/r2/p4_gate.py,
p4b_gate_threads.py y p6_hrm2.py.

Todo corre por el camino CPU (sin GPU en esta máquina).
"""

import json
import os
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

import m2m.gate_modulus as gm
import m2m.gpu_vector_index as gvi
from m2m.gate_modulus import BASE_SCALES, GateKSelection, gate_select_k
from m2m.hrm2_engine import HRM2Config, HRM2Engine

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ---------------------------------------------------------------------------
# Generadores
# ---------------------------------------------------------------------------

def _unit(A):
    A = np.asarray(A, dtype=np.float32)
    return (A / np.linalg.norm(A, axis=1, keepdims=True)).astype(np.float32)


def _uniform(n, d, seed=0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, d)).astype(np.float32)


def _clustered(n_centers=20, per_center=150, d=32, sigma=0.3, seed=0):
    rng = np.random.default_rng(seed)
    C = rng.standard_normal((n_centers, d)).astype(np.float32) * 3.0
    X = C[rng.integers(0, n_centers, n_centers * per_center)]
    X = X + sigma * rng.standard_normal(X.shape).astype(np.float32)
    return _unit(X)


def _sharp_leaves(n_leaves=30, per_leaf=40, d=64, sigma=0.05, seed=0):
    """Hojas nítidas y bien separadas: el kNN (k=10) no conecta las hojas ->
    grafo de clústeres desconectado a escalas K > n_leaves (caso r2-12)."""
    rng = np.random.default_rng(seed)
    C = _unit(rng.standard_normal((n_leaves, d)).astype(np.float32))
    X = C[rng.integers(0, n_leaves, n_leaves * per_leaf)]
    X = X + sigma * rng.standard_normal(X.shape).astype(np.float32)
    return _unit(X)


def _ids(r):
    return [int(x[0]) for x in r]


def _dists(r):
    return [float(x[1]) for x in r]


def _recall(res, gt, k=10):
    return float(np.mean([len(set(r) & set(g.tolist())) / k for r, g in zip(res, gt)]))


def _expected_lod01(eng, q_raw, k, lod):
    """Contrato de lod 0/1 (BACKLOG P11a #5): por orden de cercanía del
    centroide grueso, los k MÁS CERCANOS (distancia exacta) de cada clúster
    sondeado — lod 1: de los clústeres finos más cercanos hasta reunir k por
    clúster grueso — hasta reunir k candidatos; después top-k por distancia."""
    q = np.ascontiguousarray(np.asarray(q_raw, dtype=np.float32).flatten())
    if eng.metric == "cosine":
        q = q / np.linalg.norm(q)
    cent = eng.coarse_model.centroids_  # misma fórmula de ruteo que query()
    cd = np.sqrt(np.maximum(((cent - q) ** 2).sum(axis=1), 0.0))
    closest = np.argsort(cd)[: eng._adaptive_n_probe(cd)]
    cands = []
    for c in closest:
        idx = np.where(eng.coarse_assignments == c)[0]
        if lod == 0:
            d = np.linalg.norm(eng.embeddings[idx] - q, axis=1)
            order = np.argsort(d, kind="stable")[:k]
            cands += [(int(idx[i]), float(d[i])) for i in order]
        else:
            added = 0
            fa = eng.fine_assignments[c]
            fd = eng.fine_models[c].transform(q.reshape(1, -1))[0]
            for f in np.argsort(fd, kind="stable"):
                mem = idx[fa == f]
                if len(mem) == 0:
                    continue
                d = np.linalg.norm(eng.embeddings[mem] - q, axis=1)
                order = np.argsort(d, kind="stable")[:k]
                cands += [(int(mem[i]), float(d[i])) for i in order]
                added += len(order)
                if added >= k:
                    break
        if len(cands) >= k:
            break
    return sorted(cands, key=lambda t: t[1])[:k]


# ===========================================================================
# Criterio 1 (r2-10): gate_select_k no lanza; determinista 1 vs 4 hilos
# ===========================================================================

@pytest.mark.parametrize("n", [64, 100, 128, 176, 177, 256, 512, 1000, 4600])
def test_gate_select_k_no_crash(n):
    X = _uniform(n, 32)
    sel = gate_select_k(X, seed=42)  # hoy: TypeError 100 <= N <= 177, ValueError N=64
    assert isinstance(sel, GateKSelection)
    assert sel.verdict["clasificacion"] in ("jerarquico", "mono", "inconclusivo")
    assert 2 <= sel.k <= max(2, n // 4)
    print(f"\n[r2-10] N={n} k={sel.k} jerarquico={sel.jerarquico} "
          f"cls={sel.verdict['clasificacion']} escalas={sel.verdict['n_escalas_con_energia']}")


@pytest.mark.parametrize("n", [8, 16, 32])
def test_gate_select_k_tiny_n_is_documented_valueerror(n):
    # N < 64 es entrada inválida documentada (ValueError explícito), no un crash.
    with pytest.raises(ValueError, match="N>=64"):
        gate_select_k(_uniform(n, 8))


def test_gate_degenerate_trajectory_returns_honest_verdict():
    # N=64: ninguna escala elegible (16 >= N/4) -> antes ValueError; hoy
    # veredicto 'inconclusivo' marcado degenerado (K operacional).
    sel = gate_select_k(_uniform(64, 16), seed=42)
    v = sel.verdict
    assert v["degenerado"] is True
    assert v["clasificacion"] == "inconclusivo" and sel.jerarquico is False
    assert v["n_escalas_con_energia"] == 0
    assert v["escalas"]["omitidas_K_ge_N4"] == [16]
    assert sel.k == 2


def test_gate_hrm2_integration_small_corpus_does_not_crash(capsys):
    # HRM2 con k_selection='gate' heredaba el crash para corpus chicos.
    eng = HRM2Engine(k_selection="gate", embedding_dim=16, metric="cosine")
    eng.index(precomputed_embeddings=_uniform(100, 16))
    assert eng._is_indexed and 2 <= eng._stats.n_coarse_clusters <= 10
    r = eng.query(_uniform(1, 16)[0], k=5)
    assert len(r) == 5


_THREAD_PROBE = r"""
import hashlib, json, numpy as np
from m2m.gate_modulus import _verdict
rng = np.random.default_rng(0)
X = rng.standard_normal((256, 16)).astype(np.float32)
X /= np.linalg.norm(X, axis=1, keepdims=True)
v = _verdict(X, seed=42)
v = {k: val for k, val in v.items() if k != "t_s"}
print(hashlib.md5(json.dumps(v, sort_keys=True, default=str).encode()).hexdigest(), v["clasificacion"])
"""


def test_gate_verdict_deterministic_1_vs_4_threads(tmp_path):
    """md5 del veredicto (sin t_s) idéntico con 1 y 4 hilos de BLAS."""
    hashes = {}
    for n_threads in ("1", "4"):
        env = dict(
            os.environ,
            OPENBLAS_NUM_THREADS=n_threads,
            OMP_NUM_THREADS=n_threads,
            MKL_NUM_THREADS=n_threads,
            PYTHONPATH=os.path.join(ROOT, "src"),
        )
        r = subprocess.run([sys.executable, "-c", _THREAD_PROBE], env=env,
                           capture_output=True, text=True, timeout=600)
        assert r.returncode == 0, r.stderr[-2000:]
        md5, cls = r.stdout.strip().split()
        hashes[n_threads] = md5
        print(f"\n[r2-10 threads={n_threads}] md5={md5} cls={cls}")
    assert hashes["1"] == hashes["4"]


# ===========================================================================
# Criterio 2 (r2-11): el cache no reusa veredictos entre corpus distintos
# ===========================================================================

def test_gate_cache_does_not_reuse_verdict_across_corpora(tmp_path, monkeypatch):
    N, D = 400, 16
    A = _uniform(N, D, seed=1)
    B = _uniform(N, D, seed=2)  # mismo (N, D, target, seed), corpus distinto
    cache = str(tmp_path / "verdict.json")
    calls = []
    real = gm._verdict

    def spy(X, seed=42):
        calls.append(tuple(X[0][:3]))
        return real(X, seed=seed)

    monkeypatch.setattr(gm, "_verdict", spy)
    sA = gate_select_k(A, verdict_cache=cache)
    assert len(calls) == 1
    # Cache de UNA entrada: mismo corpus -> reusa (no recomputa)
    sA2 = gate_select_k(A, verdict_cache=cache)
    assert len(calls) == 1, "mismo corpus: debería reusar el cache"
    assert sA2.verdict["fingerprint"] == gm._data_fingerprint(A)
    assert {str(k): v for k, v in sA.verdict["medianas"].items()} == sA2.verdict["medianas"]
    # Otro corpus de mismo (N, D): NO reusa
    sB = gate_select_k(B, verdict_cache=cache)  # hoy: reusa el veredicto de A
    assert len(calls) == 2, "el cache reusó el veredicto de otro corpus de igual (N, D)"
    assert sB.verdict["medianas"] != sA.verdict["medianas"]
    on_disk = json.loads(open(cache).read())
    assert on_disk["fingerprint"] == gm._data_fingerprint(B)
    assert on_disk["algo"] == gm._CACHE_ALGO


def test_gate_cache_rejects_legacy_cache_without_fingerprint(tmp_path, monkeypatch):
    # Cache del formato viejo (sin fingerprint/algo): no se confía en él.
    X = _uniform(400, 16, seed=3)
    legacy = {
        "target_docs_per_cluster": 25, "seed": 42,
        "verdict": {"N": 400, "D": 16, "jerarquico": True,
                    "slope_octava_fina": {"octava": "16->22", "slope": -2.0, "IC95": [-2, -1]},
                    "knee": None, "clasificacion": "jerarquico"},
    }
    cache = tmp_path / "legacy.json"
    cache.write_text(json.dumps(legacy))
    calls = []
    real = gm._verdict
    monkeypatch.setattr(gm, "_verdict", lambda X, seed=42: (calls.append(1), real(X, seed=seed))[1])
    sel = gate_select_k(X, verdict_cache=str(cache))
    assert len(calls) == 1, "cache legacy sin fingerprint fue aceptado"
    assert "fingerprint" in json.loads(cache.read_text())


# ===========================================================================
# Criterio 3 (r2-12): escalas descartadas reportadas + estabilidad por semilla
# ===========================================================================

def test_gate_reports_dropped_scales_on_sharp_leaves():
    X = _sharp_leaves()
    scales = [s for s in BASE_SCALES if s <= max(12, len(X) // 4)]
    rep = {}
    traj = gm._trajectory(X, scales, seed=42, report=rep)
    evaluadas = [K for K in scales if rep[K]["estado"] != "omitida"]
    assert set(rep) == set(scales), "toda escala pedida debe tener destino reportado"
    for K in evaluadas:
        assert rep[K]["estado"] in ("ok", "descartada")
        if rep[K]["estado"] == "descartada":
            assert rep[K]["motivo"] and K not in traj
        else:
            assert rep[K]["n_pares"] == len(traj[K])
    # La desconexión de hojas nítidas tiene que verse en algún lado.
    assert any(r["estado"] == "descartada" or r.get("pares_descartados", 0) > 0
               for r in rep.values())
    sel = gate_select_k(X, seed=42)
    v = sel.verdict
    assert v["dropped_scales"] == sorted(K for K in evaluadas if K not in traj)
    assert sel.dropped_scales == v["dropped_scales"]
    fine_cap = max(44, len(X) // 10)
    assert v["escalas"]["finas_perdidas"] == [K for K in v["dropped_scales"] if K <= fine_cap]
    for K in v["dropped_scales"]:
        assert v["escalas"]["descartadas"][str(K)]["motivo"]
    print(f"\n[r2-12] con_energia={v['escalas']['con_energia']} "
          f"descartadas={v['dropped_scales']} finas_perdidas={v['escalas']['finas_perdidas']} "
          f"cls={v['clasificacion']} k={sel.k}")
    # Honesto: si se perdió una escala fina, nunca 'mono'/'jerárquico' sin avisar.
    if v["escalas"]["finas_perdidas"]:
        assert v["clasificacion"] == "inconclusivo" and sel.jerarquico is False


@pytest.mark.parametrize("forzada", ["mono", "jerarquico"])
def test_gate_downgrades_to_inconclusive_when_fine_scales_lost(monkeypatch, forzada):
    X = _unit(_uniform(600, 16))
    scales = [s for s in BASE_SCALES if s <= max(12, len(X) // 4)]
    full = gm._trajectory(X, scales, seed=42)
    real = gm._trajectory

    def lossy(X_, scales_, n_pairs=30, seed=42, report=None):
        traj = real(X_, scales_, n_pairs=n_pairs, seed=seed, report=report)
        for K in sorted(traj)[:2]:  # "desconectamos" las 2 escalas más finas
            traj.pop(K)
            if report is not None:
                report[K] = {"estado": "descartada", "motivo": "test: grafo desconectado",
                             "n_pares": 0, "pares_descartados": 120}
        return traj

    # Clasificación forzada (independiente de los datos) para aislar la regla
    fake = {"clasificacion": forzada, "jerarquico": forzada == "jerarquico",
            "slope_fino": 0.0, "IC95": [0.0, 0.1]}
    monkeypatch.setattr(gm, "_classify_fine", lambda *args, **kw: dict(fake))
    v_ok = gm._verdict(X, seed=42)  # sin pérdidas: la clasificación pasa tal cual
    assert v_ok["escalas"]["finas_perdidas"] == []
    assert v_ok["clasificacion"] == forzada
    monkeypatch.setattr(gm, "_trajectory", lossy)
    v = gm._verdict(X, seed=42)
    assert v["escalas"]["finas_perdidas"] == sorted(full)[:2]
    assert v["dropped_scales"] == v["escalas"]["finas_perdidas"]
    assert v["clasificacion"] == "inconclusivo" and v["jerarquico"] is False
    assert v["clasificacion_sin_escalas_perdidas"] == forzada
    assert "inconclusivo" in v["regla"]


def test_gate_reports_seed_stability():
    X = _uniform(300, 16)
    sel0 = gate_select_k(X, seed=42)
    assert sel0.verdict["estabilidad_semilla"]["evaluada"] is False
    sel = gate_select_k(X, seed=42, stability_seeds=2)
    est = sel.verdict["estabilidad_semilla"]
    assert est["evaluada"] is True and est["n_seeds"] == 3
    assert set(est["clasificaciones"]) == {"42", "43", "44"}
    flips = sum(c != est["clasificaciones"]["42"] for c in est["clasificaciones"].values())
    assert est["flip_rate"] == pytest.approx(flips / 2)
    assert est["estable"] is (flips == 0)
    if flips:
        assert "inestable" in sel.regla
    print(f"\n[r2-12] estabilidad_semilla={est}")


# ===========================================================================
# Criterio 4 (r2-18): query_batch == query fila a fila (norma != 1)
# ===========================================================================

def _build(metric, n=4000, d=32, n_coarse=50, n_probe=5):
    eng = HRM2Engine(n_coarse=n_coarse, n_fine=20, embedding_dim=d, n_probe=n_probe,
                     metric=metric)
    eng.index(precomputed_embeddings=_unit(_uniform(n, d, seed=7)))
    return eng


def _non_unit_queries(d=32, n=500, seed=11):
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((n, d)) * 3.0).astype(np.float32)


@pytest.mark.parametrize("metric", ["cosine", "euclidean"])
def test_query_batch_matches_query_row_by_row_non_unit_queries(metric):
    eng = _build(metric)
    Q = _non_unit_queries(d=32)
    assert np.all(np.abs(np.linalg.norm(Q, axis=1) - 1) > 1)
    rb = eng.query_batch(Q, k=10)
    rq = [eng.query(q, k=10) for q in Q]
    assert len(rb) == len(rq) == 500
    mism = sum(_ids(a) != _ids(b) for a, b in zip(rb, rq))
    print(f"\n[r2-18 metric={metric}] filas distintas={mism}/500")
    for a, b in zip(rb, rq):
        assert _ids(a) == _ids(b)
        np.testing.assert_array_equal(_dists(a), _dists(b))


def test_query_batch_matches_query_with_adaptive_probing():
    eng = _build("cosine")
    eng.adaptive_probing = True
    eng.adaptive_threshold = 1.0   # casi todas las queries "fáciles" -> 1 sonda
    eng.adaptive_min_probe = 1
    Q = _non_unit_queries(d=32, n=300)
    rb = eng.query_batch(Q, k=10)
    rq = [eng.query(q, k=10) for q in Q]
    mism = sum(_ids(a) != _ids(b) for a, b in zip(rb, rq))
    print(f"\n[r2-18 adaptive] filas distintas={mism}/300")
    for a, b in zip(rb, rq):
        assert _ids(a) == _ids(b)
        np.testing.assert_array_equal(_dists(a), _dists(b))


@pytest.mark.parametrize("lod", [0, 1])
def test_query_batch_lod01_delegates_with_parity(lod):
    eng = _build("cosine", n=1500, d=16, n_coarse=20)
    Q = _non_unit_queries(d=16, n=40)
    rb = eng.query_batch(Q, k=10, lod=lod)
    rq = [eng.query(q, k=10, lod=lod) for q in Q]
    for a, b in zip(rb, rq):
        assert _ids(a) == _ids(b)
        np.testing.assert_allclose(_dists(a), _dists(b), rtol=0, atol=1e-6)


# ===========================================================================
# Criterio 5 (r2-19): HRM2Config cableado; lod 0/1 devuelven los más cercanos
# ===========================================================================

def test_hrm2config_is_wired_into_engine():
    cfg = HRM2Config(adaptive_probing=True, adaptive_threshold=1.2, adaptive_min_probe=2,
                     n_probe=7, n_coarse=12, n_fine=17, metric="cosine", random_state=7)
    eng = HRM2Engine(config=cfg)  # hoy: adaptive_probing False, n_probe 5
    assert eng.adaptive_probing is True
    assert eng.adaptive_threshold == 1.2 and eng.adaptive_min_probe == 2
    assert eng.n_probe == 7 and eng.n_coarse == 12 and eng.n_fine == 17
    assert eng.metric == "cosine" and eng.random_state == 7
    # El adaptive probing del config tiene efecto en el ruteo
    assert eng._adaptive_n_probe(np.array([1.0, 2.0], dtype=np.float32)) == 2  # ratio 2 > 1.2 -> fácil
    assert eng._adaptive_n_probe(np.array([1.0, 1.1], dtype=np.float32)) == 7
    # Kwarg explícito (distinto del default) manda sobre el config
    eng2 = HRM2Engine(config=cfg, n_probe=3, metric="euclidean")
    assert eng2.n_probe == 3 and eng2.metric == "euclidean" and eng2.adaptive_probing is True
    # Sin config (o con M2MConfig): sin cambios de comportamiento
    eng3 = HRM2Engine(n_coarse=10)
    assert eng3.adaptive_probing is False and eng3.n_probe == 5 and eng3.batch_size == 10000
    from m2m import engine as _engine_mod
    eng4 = HRM2Engine(n_coarse=10, config=_engine_mod.M2MConfig())
    assert eng4.adaptive_probing is False and eng4.n_probe == 5


@pytest.mark.parametrize("lod", [0, 1])
def test_lod01_returns_nearest_within_probed_clusters(lod):
    X = _clustered(n_centers=20, per_center=150, d=32)
    rng = np.random.default_rng(5)
    Q = _unit(X[rng.integers(0, len(X), 100)]) * 2.0  # norma 2 (no unitaria)
    eng = HRM2Engine(n_coarse=20, n_fine=5, embedding_dim=32, n_probe=3, metric="cosine")
    eng.index(precomputed_embeddings=X)
    res = [eng.query(q, k=10, lod=lod) for q in Q]
    # Contrato exacto: los más cercanos de los clústeres sondeados, ordenados
    for q, r in zip(Q, res):
        exp = _expected_lod01(eng, q, 10, lod)
        assert _ids(r) == [i for i, _ in exp]
        np.testing.assert_allclose(_dists(r), [d for _, d in exp], rtol=0, atol=1e-5)
        assert _dists(r) == sorted(_dists(r))
    # query_with_details sigue el mismo contrato
    det = [eng.query_with_details(q, k=10, lod=lod) for q in Q[:30]]
    for r, d in zip(res[:30], det):
        assert _ids(r) == [x.splat_id for x in d]
    # Medición (sonda p6_hrm2.py, clustered): recall@10 vs brute force
    gt = np.argsort(-(_unit(Q) @ X.T), axis=1)[:, :10]
    rec = _recall([_ids(r) for r in res], gt)
    print(f"\n[r2-19 lod={lod}] recall@10 clustered (n_fine=5) = {rec:.3f}")
    # lod 0 (clúster grueso completo) recupera lo mismo que lod 2 en datos
    # clusterizados. lod 1 sólo escanea los clústeres finos más cercanos
    # (hasta k miembros por clúster grueso): techo estructural ~0.5 con
    # n_fine=5 (los 10-NN se reparten entre 2-5 clústeres finos); antes del
    # fix (primeros k por índice) la sonda p6_hrm2.py daba 0.166 (n_fine=20).
    assert rec >= (0.9 if lod == 0 else 0.4), f"lod={lod} recall@10 {rec:.3f}"


