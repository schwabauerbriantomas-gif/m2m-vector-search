"""Ciclo de vida del índice IVF: attach incremental, compact y consolidate.

Audit r1-11: tras un attach, `SplatStore.build_index` solo extendía las
máscaras de los clusters que recibieron puntos nuevos; las demás quedaban con
la longitud vieja → `_cluster_masks` heterogéneas → `update(id, vector=...)`
sobre un doc agregado después del último retrain lanzaba IndexError (indexa
`_cluster_masks[cid][pos]`) y los candidatos de HRM2Engine quedaban
desalineados respecto de `coarse_cluster_indices`.

Audit r1-25: `consolidate()` compacta las filas de `mu` y el `build_index()`
siguiente tomaba el camino attach (n <= 1.25 * _last_trained_n), dejando el
índice con posiciones pre-compaction → IndexError en toda búsqueda.
"""

import numpy as np
import pytest

from m2m import AdvancedVectorDB, SimpleVectorDB, normalize_sphere

D = 24
N_BASE, N_ATTACH = 4000, 500


def _norm(x):
    return normalize_sphere(np.asarray(x, dtype=np.float32))


def _gt_topk(db, q, k, pool=None):
    """GT exacto por fuerza bruta (float64) entre docs activos (o un subconjunto)."""
    ids = [d for d in db._vectors if d not in db._deleted]
    if pool is not None:
        ids = [i for i in ids if i in pool]
    X = _norm(np.stack([db._vectors[i] for i in ids])).astype(np.float64)
    d = ((X - np.asarray(q, dtype=np.float64)) ** 2).sum(1)
    order = np.argsort(d, kind="stable")[:k]
    return [ids[i] for i in order]


def _recall(returned, truth):
    t = set(truth)
    return len([r for r in returned if r in t]) / len(t)


def _assert_index_invariants(db):
    """Criterio 1: TODAS las estructuras del índice alineadas con n_active."""
    store = db.engine.m2m.splats
    eng = store.engine
    n = store.n_active
    assert eng._is_indexed
    assert len(eng.coarse_assignments) == n
    assert len(eng.embeddings) == n
    assert len(eng._cluster_masks) > 0
    lens = sorted({int(m.shape[0]) for m in eng._cluster_masks.values()})
    assert lens == [n], f"máscaras con longitudes heterogéneas: {lens} (n_active={n})"

    masks = eng._cluster_masks
    # Cada posición pertenece a exactamente un cluster...
    stack = np.stack([masks[c] for c in sorted(masks)])
    assert int(stack.sum()) == n
    assert np.array_equal(stack.sum(axis=0), np.ones(n, dtype=int))
    # ...y las listas / embeddings precomputados coinciden con las máscaras.
    all_idx = np.concatenate([np.asarray(v) for v in eng.coarse_cluster_indices.values()])
    assert len(all_idx) == n and len(np.unique(all_idx)) == n
    assert all_idx.min() >= 0 and all_idx.max() < n
    for cid, idx in eng.coarse_cluster_indices.items():
        assert cid in masks
        idx = np.asarray(idx)
        assert np.array_equal(np.sort(idx), np.where(masks[cid])[0])
        assert len(eng.coarse_cluster_embeddings[cid]) == len(idx)
        assert np.all(eng.coarse_assignments[idx] == cid)
    assert len(db._splat_id_order) == n
    return n


def _snapshot(db):
    store = db.engine.m2m.splats
    eng = store.engine
    return {
        "n_active": store.n_active,
        "last_trained_n": store._last_trained_n,
        "mask_lens": sorted({int(m.shape[0]) for m in eng._cluster_masks.values()}),
        "assignments": len(eng.coarse_assignments),
        "embeddings": len(eng.embeddings),
        "id_order": len(db._splat_id_order),
        "coarse_model_id": id(eng.coarse_model),
    }


def _check_engine_self_queries(eng, n, positions):
    """Criterio 4: query/query_batch/query_with_details válidas y consistentes.

    Para una posición p, la consulta por su propio embedding debe devolver p
    como top-1 (distancia 0) en las tres APIs, y toda posición devuelta debe
    estar en [0, n) y ser coherente con coarse_assignments.
    """
    for p in positions:
        q = np.asarray(eng.embeddings[p], dtype=np.float32)
        ids_q = [r[0] for r in eng.query(q, k=5)]
        ids_b = [r[0] for r in eng.query_batch(q[np.newaxis, :], k=5)[0]]
        det = eng.query_with_details(q, k=5)
        ids_d = [s.splat_id for s in det]
        assert len(ids_q) == 5 and len(ids_b) == 5 and len(ids_d) == 5
        assert ids_q[0] == p and ids_b[0] == p and ids_d[0] == p
        for i in ids_q + ids_b + ids_d:
            assert 0 <= i < n
        for s in det:
            assert int(eng.coarse_assignments[s.splat_id]) == int(s.coarse_cluster)
            assert s.splat_id in set(eng.coarse_cluster_indices[int(s.coarse_cluster)].tolist())


def _build_attached(seed=7):
    """add 4000 (retrain) → add 500 (attach) → add 1 (attach)."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((N_BASE + N_ATTACH + 1, D)).astype(np.float32)
    ids = [f"t{i}" for i in range(len(X))]
    db = SimpleVectorDB(latent_dim=D)
    db.add(ids=ids[:N_BASE], vectors=X[:N_BASE])
    s1 = _snapshot(db)
    db.add(ids=ids[N_BASE : N_BASE + N_ATTACH], vectors=X[N_BASE : N_BASE + N_ATTACH])
    s2 = _snapshot(db)
    db.add(ids=ids[-1:], vectors=X[-1:])
    s3 = _snapshot(db)
    return db, X, ids, (s1, s2, s3)


@pytest.fixture(scope="module")
def attached():
    return _build_attached()


# ---------------------------------------------------------------------------
# r1-11 — máscaras desalineadas tras attach
# ---------------------------------------------------------------------------


def test_r1_11_attach_keeps_all_masks_aligned(attached):
    """Criterio 1: tras add 4000 → add 500 → add 1, una sola longitud en todo."""
    db, X, ids, (s1, s2, s3) = attached

    # Los dos adds posteriores van por el camino ATTACH (no retrain).
    assert s1["n_active"] == N_BASE and s1["last_trained_n"] == N_BASE
    assert s2["n_active"] == N_BASE + N_ATTACH == s2["last_trained_n"]
    assert s2["coarse_model_id"] == s1["coarse_model_id"], "add 500 debía ser attach"
    assert s3["n_active"] == N_BASE + N_ATTACH + 1 == s3["last_trained_n"]
    assert s3["coarse_model_id"] == s1["coarse_model_id"], "add 1 debía ser attach"

    # Hoy: solo las máscaras de los clusters que recibieron puntos nuevos se
    # extienden → longitudes {4000, 4500} tras el attach de 500.
    assert s2["mask_lens"] == [N_BASE + N_ATTACH], s2
    assert s2["assignments"] == s2["embeddings"] == s2["id_order"] == N_BASE + N_ATTACH
    assert s3["mask_lens"] == [N_BASE + N_ATTACH + 1], s3
    assert s3["assignments"] == s3["embeddings"] == s3["id_order"] == N_BASE + N_ATTACH + 1

    n = _assert_index_invariants(db)
    assert n == N_BASE + N_ATTACH + 1


# r1-11 y [audit r1-24]: update(vector=) tras un attach mantiene alineadas las
# asignaciones, los embeddings de cluster y las máscaras
def test_r1_11_update_vector_after_attach():
    """Criterio 2: update(id, vector=) sobre docs agregados post-índice, 50 vectores."""
    db, X, ids, _ = _build_attached()
    store = db.engine.m2m.splats
    eng = store.engine
    rng = np.random.default_rng(11)
    new_vecs = (
        rng.standard_normal((100, D)).astype(np.float32) * np.linspace(0.5, 3.0, 100)[:, None]
    )

    # (a) el doc agregado de a uno (posición 4500, la que excede las máscaras cortas)
    zid = ids[-1]
    pos = db._splat_id_order.index(zid)
    assert pos == N_BASE + N_ATTACH
    # (b) 50 docs del lote attachado
    batch = ids[N_BASE : N_BASE + 50]
    batch_pos = [db._splat_id_order.index(b) for b in batch]

    for j in range(50):
        for did, p in ((zid, pos), (batch[j], batch_pos[j])):
            v = new_vecs[j] if did == zid else new_vecs[50 + j]
            old_cid = int(eng.coarse_assignments[p])
            db.update(did, vector=v)  # hoy: IndexError en _cluster_masks[cid][pos]
            new_cid = int(eng.coarse_assignments[p])
            assert p in set(eng.coarse_cluster_indices[new_cid].tolist())
            assert eng._cluster_masks[new_cid][p]
            if new_cid != old_cid:
                assert p not in set(eng.coarse_cluster_indices[old_cid].tolist())
                assert not eng._cluster_masks[old_cid][p]
            assert len(eng.coarse_cluster_embeddings[new_cid]) == len(
                eng.coarse_cluster_indices[new_cid]
            )
            assert np.allclose(store.mu[p], _norm(v), atol=1e-6)
            assert np.allclose(db._vectors[did], v)
            top = db.search(v, k=1, include_metadata=True)
            assert len(top) == 1 and top[0].id == did

    # El estado queda consistente (ninguna actualización a medias).
    _assert_index_invariants(db)
    for q in X[:20]:
        assert len(db.search(q, k=5, include_metadata=True)) == 5


def test_r1_11_engine_queries_consistent_after_attach(attached):
    """Criterio 4 (attach): posiciones válidas y consistentes en las 3 APIs."""
    db, X, ids, _ = attached
    eng = db.engine.m2m.splats.engine
    n = db.engine.m2m.splats.n_active
    positions = list(range(0, N_BASE, 100)) + list(range(N_BASE, n, 5)) + [n - 1]
    _check_engine_self_queries(eng, n, positions)


def test_r1_11_recall_after_attach_not_lower():
    """Criterio 5: recall@10 antes/después del attach, mismo seed y mismas queries.

    "Antes" = índice de 4000 docs (GT por fuerza bruta sobre esos 4000);
    "después" = mismo índice tras attach de 500 (GT sobre los 4500: un índice
    correcto debe encontrar los verdaderos top-10 aunque sean docs nuevos).
    Consultas aleatorias (datos sin estructura → recall IVF intrínsecamente
    bajo) y self-queries de docs base.
    """
    rng = np.random.default_rng(7)
    X = rng.standard_normal((N_BASE + N_ATTACH, D)).astype(np.float32)
    ids = [f"t{i}" for i in range(len(X))]
    Q_rand = np.random.default_rng(42).standard_normal((300, D)).astype(np.float32)
    queries = [("random", q) for q in Q_rand] + [("self", X[i]) for i in range(0, N_BASE, 13)]

    db = SimpleVectorDB(latent_dim=D)
    db.add(ids=ids[:N_BASE], vectors=X[:N_BASE])

    def measure():
        """Devuelve (recall medio por tipo, ids devueltos por query, GT por query)."""
        recs = {"random": [], "self": []}
        got_sets, gt_sets = [], []
        for kind, q in queries:
            got = {r.id for r in db.search(q, k=10, include_metadata=True)}
            gt = set(_gt_topk(db, _norm(q), 10))
            recs[kind].append(len(got & gt) / len(gt))
            got_sets.append(got)
            gt_sets.append(gt)
        return {k: float(np.mean(v)) for k, v in recs.items()}, got_sets, gt_sets

    before, got_before, _ = measure()
    assert db.engine.m2m.splats._last_trained_n == N_BASE
    db.add(ids=ids[N_BASE:], vectors=X[N_BASE:])
    assert db.engine.m2m.splats._last_trained_n == N_BASE + N_ATTACH
    after, got_after, gt_after = measure()

    # Exacto: un doc base que el índice devolvía antes y que sigue siendo un
    # verdadero top-10 (sobre los 4500) no puede perderse tras el attach — la
    # estructura de índice de los puntos viejos no cambia; los docs nuevos
    # solo pueden desplazar a los que ya NO son verdaderos top-10.
    checked = lost = 0
    for gb, ga, gt in zip(got_before, got_after, gt_after):
        keep = gb & gt
        checked += len(keep)
        lost += len(keep - ga)
    assert lost == 0, f"{lost}/{checked} vecinos base perdidos tras el attach"

    # Medición antes/después: el GT cambia con el dataset (4000 → 4500), así
    # que la cota admite ruido de muestreo, no una baja real.
    print(
        f"\n[r1-11] recall@10 random: antes={before['random']:.4f} "
        f"tras attach={after['random']:.4f}"
        f" | self: antes={before['self']:.4f} tras attach={after['self']:.4f}"
        f" | vecinos base perdidos: {lost}/{checked}"
    )
    for kind in ("random", "self"):
        assert after[kind] >= before[kind] - 0.01, (kind, before, after)


# ---------------------------------------------------------------------------
# r1-25 — consolidate() poda y deja el índice con posiciones pre-compaction
# ---------------------------------------------------------------------------

N_TOTAL, PRUNE_LO, PRUNE_HI = 6000, 1000, 2500  # alpha=0.001 → 1500 splats fuera


def _assert_engine_aligned(eng, n):
    """Estructuras del engine alineadas con n (sin depender de SimpleVectorDB)."""
    assert eng._is_indexed
    assert len(eng.coarse_assignments) == n
    assert len(eng.embeddings) == n
    assert len(eng._cluster_masks) > 0
    assert {int(m.shape[0]) for m in eng._cluster_masks.values()} == {n}
    stack = np.stack([eng._cluster_masks[c] for c in sorted(eng._cluster_masks)])
    assert np.array_equal(stack.sum(axis=0), np.ones(n, dtype=int))
    all_idx = np.concatenate([np.asarray(v) for v in eng.coarse_cluster_indices.values()])
    assert len(all_idx) == n and len(np.unique(all_idx)) == n
    assert all_idx.max() < n
    for cid, idx in eng.coarse_cluster_indices.items():
        assert np.array_equal(np.sort(np.asarray(idx)), np.where(eng._cluster_masks[cid])[0])
        assert len(eng.coarse_cluster_embeddings[cid]) == len(idx)


@pytest.fixture(scope="module")
def consolidated():
    """6000 docs → alpha[1000:2500]=0.001 → consolidate(threshold=0.01)."""
    rng = np.random.default_rng(7)
    X = rng.standard_normal((N_TOTAL, D)).astype(np.float32)
    ids = [f"v{i}" for i in range(N_TOTAL)]
    adv = AdvancedVectorDB(latent_dim=D, enable_soc=False, enable_energy_features=False)
    adv.add(ids=ids, vectors=X)
    adv.engine.m2m.splats.alpha[PRUNE_LO:PRUNE_HI] = 0.001
    n_removed = adv.consolidate(threshold=0.01)
    return adv, X, ids, n_removed


def test_r1_25_consolidate_keeps_index_aligned_and_searchable(consolidated):
    """Criterio 3: poda real → search no lanza, recall@5 == 1.0, invariantes de (1)."""
    adv, X, ids, n_removed = consolidated
    store = adv.engine.m2m.splats
    removed = set(ids[PRUNE_LO:PRUNE_HI])
    assert n_removed == PRUNE_HI - PRUNE_LO == 1500
    assert store.n_active == N_TOTAL - 1500
    assert len(adv._splat_id_order) == len(adv._vectors) == store.n_active
    assert not (set(adv._vectors) & removed)
    assert store._last_trained_n == store.n_active
    assert _assert_index_invariants(adv) == 4500

    recs = []
    for i in range(0, N_TOTAL, 15):
        if ids[i] in removed:
            continue
        res = adv.search(X[i], k=5, include_metadata=True)  # hoy: IndexError
        got = [r.id for r in res]
        assert len(got) == 5 and not (set(got) & removed)
        for r in res:
            assert np.allclose(r.vector, _norm(adv._vectors[r.id]), atol=1e-5)
        recs.append(_recall(got, _gt_topk(adv, _norm(X[i]), 5)))

    # (a) Paridad exacta query a query con un índice RECÍN construido sobre los
    # docs sobrevivientes: consolidar no debe dejar un índice peor que empezar
    # de cero. (Datos gaussianos sin estructura: el recall IVF absoluto es
    # intrínsecamente < 1 con n_probe=5, por eso la comparación es de paridad.)
    surv = [i for i in ids if i not in removed]
    fresh = AdvancedVectorDB(latent_dim=D, enable_soc=False, enable_energy_features=False)
    fresh.add(ids=surv, vectors=X[[ids.index(i) for i in surv]])
    assert _assert_index_invariants(fresh) == 4500
    recs_fresh = [
        _recall(
            [r.id for r in fresh.search(X[i], k=5, include_metadata=True)],
            _gt_topk(fresh, _norm(X[i]), 5),
        )
        for i in range(0, N_TOTAL, 15)
        if ids[i] in set(surv)
    ]
    print(
        f"\n[r1-25] recall@5 self-queries gaussianos: consolidate={np.mean(recs):.4f} "
        f"fresh={np.mean(recs_fresh):.4f} n={len(recs)} | paridad exacta: {recs == recs_fresh}"
    )
    assert len(recs) == len(recs_fresh) == 300
    assert recs == recs_fresh

    # (b) Con estructura de clústeres (el caso de uso de IVF) el recall@5 sobre
    # los docs restantes es 1.0 después de consolidar.
    rng = np.random.default_rng(5)
    centers = rng.standard_normal((40, D)).astype(np.float32) * 20.0
    Xc = centers[rng.integers(0, 40, N_TOTAL)] + rng.standard_normal((N_TOTAL, D)).astype(
        np.float32
    )
    cids = [f"c{i}" for i in range(N_TOTAL)]
    advc = AdvancedVectorDB(latent_dim=D, enable_soc=False, enable_energy_features=False)
    advc.add(ids=cids, vectors=Xc)
    advc.engine.m2m.splats.alpha[PRUNE_LO:PRUNE_HI] = 0.001
    assert advc.consolidate(threshold=0.01) == 1500
    assert _assert_index_invariants(advc) == 4500
    recs_c, removed_c = [], set(cids[PRUNE_LO:PRUNE_HI])
    for i in range(0, N_TOTAL, 15):
        if cids[i] in removed_c:
            continue
        res = advc.search(Xc[i], k=5, include_metadata=True)
        got = [r.id for r in res]
        assert len(got) == 5 and not (set(got) & removed_c)
        recs_c.append(_recall(got, _gt_topk(advc, _norm(Xc[i]), 5)))
    print(
        f"[r1-25] recall@5 self-queries clustered: mean={np.mean(recs_c):.4f} "
        f"min={np.min(recs_c):.4f} n={len(recs_c)}"
    )
    assert min(recs_c) == 1.0


def test_r1_25_engine_queries_consistent_after_consolidate(consolidated):
    """Criterio 4 (compact+retrain): posiciones válidas y consistentes en las 3 APIs,
    y el attach sigue siendo válido sobre el índice reconstruido."""
    adv, X, ids, _ = consolidated
    store = adv.engine.m2m.splats
    n = store.n_active
    _check_engine_self_queries(store.engine, n, list(range(0, n, 50)) + [n - 1])

    # vectores nuevos (no duplicados: con duplicados el top-1 de la self-query
    # es un empate y puede salir la otra posición)
    adv.add(
        ids=["post1", "post2"],
        vectors=np.random.default_rng(9).standard_normal((2, D)).astype(np.float32),
    )
    assert store._last_trained_n == n + 2
    _assert_index_invariants(adv)
    _check_engine_self_queries(store.engine, n + 2, [n, n + 1, 0, n // 2])


def test_r1_25_build_index_after_direct_compact_retrains():
    """Guardia en build_index: si n_active bajó desde el último entrenamiento
    (compact() directo, sin pasar por consolidate) el attach no es válido
    (las filas se desplazaron) → retrain completo y estructuras alineadas."""
    rng = np.random.default_rng(3)
    X = rng.standard_normal((3600, D)).astype(np.float32)
    db = SimpleVectorDB(latent_dim=D)
    db.add(ids=[f"c{i}" for i in range(len(X))], vectors=X)
    store = db.engine.m2m.splats
    cent_before = np.asarray(store.engine.coarse_model.centroids_).copy()
    store.alpha[500:1500] = 0.001
    assert store.compact() == 1000
    assert store.n_active == 2600 < store._last_trained_n
    store.build_index()
    assert store._last_trained_n == 2600
    _assert_engine_aligned(store.engine, 2600)
    # con retrain los centroides se re-adaptan al corpus compactado (y K cambia)
    cent_after = np.asarray(store.engine.coarse_model.centroids_)
    assert cent_after.shape != cent_before.shape or not np.allclose(cent_before, cent_after)
