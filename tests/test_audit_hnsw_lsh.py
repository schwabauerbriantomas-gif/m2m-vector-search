"""Tests de regresión del pass de corrección para HNSW y LSH (auditoría r2).

r2-07: ``HNSWIndex._search_layer`` devuelve (distancia, índice) del MÁS CERCANO al más
       lejano, y ``_insert`` entra a la capa inferior por el candidato más cercano
       (antes entraba por el más lejano de los ef_construction candidatos).
r2-09: el multiplicador de nivel por defecto es 1/ln(M) (Malkov & Yashunin 2016, §4.1):
       ~1/M de los nodos vive en capa >= 1. Antes era 1/ln(2) -> la mitad de los nodos.
r2-04: ``CrossPolytopeLSH`` no toca el RNG GLOBAL de numpy y su fallback (queries con
       menos de k candidatos) es determinista e independiente de ese RNG.
r2-03: con los defaults el recall@10 en datos uniformes es ~0.02. Decisión del
       orquestador: NO se promete speedup ni se inventa uno; el índice avisa con un
       ``UserWarning`` que el modo es experimental.
"""

import math
import warnings

import numpy as np
import pytest

from m2m import SimpleVectorDB
from m2m.hnsw_index import HNSWIndex
from m2m.lsh_index import CrossPolytopeLSH, LSHConfig


def _unit(a):
    return a / np.linalg.norm(a, axis=1, keepdims=True)


def _hnsw(n=300, d=16, seed=0, M=8, **kw):
    rng = np.random.default_rng(seed)
    X = _unit(rng.standard_normal((n, d)).astype(np.float32))
    h = HNSWIndex(dim=d, M=M, ef_construction=32, ef_search=32, seed=1, **kw)
    h.build(X)
    return h, X


def _lsh(dim=64, n=300, seed=42, **cfg):
    rng = np.random.default_rng(seed)
    X = _unit(rng.standard_normal((n, dim)).astype(np.float32))
    idx = CrossPolytopeLSH(LSHConfig(dim=dim, seed=seed, **cfg))
    idx.index(X)
    return idx, X


# ------------------------------------------------------------------ r2-07


def test_search_layer_returns_closest_first():
    h, X = _hnsw()
    for q in X[:5]:
        c = h._search_layer(q, h._entry_point, 16, 0)
        assert len(c) == 16
        dists = [d for d, _ in c]
        assert dists == sorted(
            dists
        ), f"_search_layer no devuelve closest-first: {dists[:4]} ... {dists[-4:]}"
        assert c[0][0] <= c[-1][0]


def test_insert_enters_next_layer_from_closest_candidate(monkeypatch):
    """Spy sobre _search_layer: para cada insert con capa >= 1, el entry point de la
    capa lev-1 debe ser el candidato MÁS CERCANO que devolvió la capa lev (el paper usa
    W completo; la implementación usa uno solo, y tenía que ser el más cercano)."""
    calls = []
    orig = HNSWIndex._search_layer

    def spy(self, query, entry, ef, level):
        out = orig(self, query, entry, ef, level)
        calls.append((query, level, entry, out))
        return out

    monkeypatch.setattr(HNSWIndex, "_search_layer", spy)
    # mult=2.0 -> P(nivel >= 1) = e^-0.5 ~ 61%: sobran inserts de varias capas.
    _hnsw(max_level_mult=2.0)

    checked = 0
    for upper, lower in zip(calls, calls[1:]):
        if upper[0] is lower[0] and upper[1] == lower[1] + 1:
            closest = min(upper[3])[1]
            farthest = max(upper[3])[1]
            assert lower[2] == closest, (
                f"capa {lower[1]}: entro por {lower[2]} (mas lejano={farthest}) "
                f"en vez del mas cercano {closest}"
            )
            checked += 1
    assert checked >= 10, f"solo {checked} inserts de varias capas: el spy no probó nada"


# ------------------------------------------------------------------ r2-09


def test_default_level_multiplier_is_one_over_ln_M():
    for M in (8, 16):
        h = HNSWIndex(dim=16, M=M)
        assert math.isclose(h.max_level_mult, 1.0 / math.log(M)), h.max_level_mult
    h = HNSWIndex(dim=16, M=16)
    levels = [h._random_level() for _ in range(4000)]
    frac = sum(lvl >= 1 for lvl in levels) / len(levels)
    assert 0.04 <= frac <= 0.09, f"fracción de nodos en capa >= 1 = {frac:.3f} (esperado ~1/16)"


def test_explicit_level_multiplier_is_honored():
    h = HNSWIndex(dim=16, M=16, max_level_mult=2.0)
    assert h.max_level_mult == 2.0
    levels = [h._random_level() for _ in range(2000)]
    assert 0.55 <= sum(lvl >= 1 for lvl in levels) / len(levels) <= 0.67  # e^-0.5 = 0.607


def test_built_hierarchy_has_paper_density():
    h, _ = _hnsw(n=400, M=16)
    levels = np.array([node.level for node in h._nodes])
    frac = float((levels >= 1).mean())
    assert (
        0.5 / 16 <= frac <= 1.5 / 16
    ), f"fracción en capa >= 1 = {frac:.3f} (esperado 1/16 = 0.0625)"
    assert levels.max() <= 3, f"nivel máximo {levels.max()} con N=400: jerarquía demasiado densa"
    # El grafo sigue siendo recorrible: todos los nodos conectados desde el entry point.
    seen, stack = {h._entry_point}, [h._entry_point]
    while stack:
        for nb in h._nodes[stack.pop()].neighbors[0]:
            if nb not in seen:
                seen.add(nb)
                stack.append(nb)
    assert len(seen) == len(h._nodes)


# ------------------------------------------------------------------ r2-04


def test_lsh_constructor_leaves_global_numpy_rng_untouched():
    np.random.seed(7)
    reference = np.random.rand(5).copy()
    np.random.seed(7)
    before = np.random.get_state()
    CrossPolytopeLSH(LSHConfig(dim=64, seed=42))
    after = np.random.get_state()
    assert before[0] == after[0] and before[2] == after[2]
    np.testing.assert_array_equal(before[1], after[1])
    np.testing.assert_array_equal(np.random.rand(5), reference)


def test_lsh_same_seed_same_results_twice():
    a, X = _lsh()
    b, _ = _lsh()
    for t in range(a.n_tables):
        for i in range(a.k):
            np.testing.assert_array_equal(a.rotations[t][i], b.rotations[t][i])
    for q in X[::37]:
        ia, da = a.query(q, k=5)
        ib, db = b.query(q, k=5)
        assert ia.tolist() == ib.tolist(), "dos índices con el mismo seed devuelven distinto top-k"
        np.testing.assert_array_equal(da, db)


def test_lsh_fallback_deterministic_and_independent_of_global_rng():
    # d=64 -> k=3 -> 128^3 buckets: casi toda query tiene < k candidatos -> fallback.
    idx, X = _lsh()
    assert idx.k == 3
    for q in X[::53]:
        np.random.seed(1)
        a = idx.query(q, k=5)[0].tolist()
        np.random.seed(2)
        b = idx.query(q, k=5)[0].tolist()
        assert a == b, "el fallback de query() depende del RNG global de numpy"
        assert idx.query(q, k=5)[0].tolist() == a, "la misma query dos veces da distinto resultado"


# ------------------------------------------------------------------ r2-03


def _recall_uniform(n=4000, dim=384, n_q=60, **cfg):
    rng = np.random.default_rng(0)
    X = _unit(rng.standard_normal((n, dim)).astype(np.float32))
    Q = _unit(rng.standard_normal((n_q, dim)).astype(np.float32))
    gt = np.argsort(-(Q @ X.T), axis=1)[:, :10]
    idx = CrossPolytopeLSH(LSHConfig(dim=dim, **cfg))
    idx.index(X)
    hits = sum(len(set(idx.query(q, k=10)[0].tolist()) & set(g.tolist())) for q, g in zip(Q, gt))
    return hits / (n_q * 10)


def test_lsh_uniform_defaults_low_recall_is_flagged_experimental():
    """Mide recall@10 con defaults en uniforme (N=4k, d=384). Si sigue < 0.5, el índice
    TIENE que avisar (UserWarning, una vez por instancia) que es experimental."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        CrossPolytopeLSH(LSHConfig(dim=384))
    flagged = [
        w
        for w in caught
        if issubclass(w.category, UserWarning) and "experimental" in str(w.message).lower()
    ]
    recall = _recall_uniform()
    if recall < 0.5:
        assert len(flagged) == 1, (
            f"recall@10={recall:.4f} con defaults y el constructor emite {len(flagged)} "
            "avisos 'experimental' (esperado 1 por instancia)"
        )


def test_simplevectordb_lsh_mode_warns_experimental():
    rng = np.random.default_rng(0)
    X = _unit(rng.standard_normal((2000, 640)).astype(np.float32))
    db = SimpleVectorDB(device="cpu", latent_dim=640, enable_lsh_fallback=True)
    with pytest.warns(UserWarning, match="(?i)experimental"):
        db.add(vectors=X)
    assert db._use_lsh and db.lsh is not None
