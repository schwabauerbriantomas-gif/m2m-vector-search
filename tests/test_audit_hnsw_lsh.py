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

import numpy as np

from m2m.hnsw_index import HNSWIndex


def _unit(a):
    return a / np.linalg.norm(a, axis=1, keepdims=True)


def _hnsw(n=300, d=16, seed=0, M=8, **kw):
    rng = np.random.default_rng(seed)
    X = _unit(rng.standard_normal((n, d)).astype(np.float32))
    h = HNSWIndex(dim=d, M=M, ef_construction=32, ef_search=32, seed=1, **kw)
    h.build(X)
    return h, X


# ------------------------------------------------------------------ r2-07


def test_search_layer_returns_closest_first():
    h, X = _hnsw()
    for q in X[:5]:
        c = h._search_layer(q, h._entry_point, 16, 0)
        assert len(c) == 16
        dists = [d for d, _ in c]
        assert dists == sorted(dists), f"_search_layer no devuelve closest-first: {dists[:4]} ... {dists[-4:]}"
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


