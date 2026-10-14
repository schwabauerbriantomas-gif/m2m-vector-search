"""Tests de diversidad Malkov (Alg. 4, arXiv:1603.09320) para HNSWIndex.

Cubre el contrato de T3:
  (a) test determinista del anillo: el edge al nodo lejano sobrevive la poda
      con Alg.4 (keepPrunedConnections + descarte por redundancia) donde la
      seleccion por cercania pura lo mataria;
  (b) corpus clusterizado sintetico (50 clusters x 20 pts, D=32, seed fija):
      recall@10 >= 0.95;
  (c) corpus uniforme: recall no baja mas de 0.01 vs implementacion vieja
      (baseline capturado ANTES del cambio con bench_t3.py, mismas seeds);
  (d) grado del grafo <= M (capas >0) y <= 2M (capa 0) en todos los casos.
"""
import os
from collections import deque

import numpy as np
import pytest

from m2m.hnsw_index import HNSWIndex

K = 10

# Baseline capturado con la implementacion vieja (Alg.3 simple) ANTES del
# cambio, misma data/seeds/config que TestUniformNoRegression (bench_t3.py).
UNIFORM_BASELINE_RECALL = 0.986


def _gt_topk(X, Q, k=K):
    sims = Q @ X.T
    return np.argsort(-sims, axis=1)[:, :k]


def _recall(X, Q, idx):
    res = [idx.search(q, k=K).indices for q in Q]
    gt = _gt_topk(X, Q)
    hits = sum(len(set(r.tolist()) & set(g.tolist())) for r, g in zip(res, gt))
    return hits / (len(gt) * gt.shape[1])


def _make_clustered(seed=7, n_clust=50, per=20, d=32):
    rng = np.random.RandomState(seed)
    vecs = []
    for c in range(n_clust):
        center = rng.randn(d) * 3.0
        for _ in range(per):
            vecs.append(center + rng.randn(d) * 0.35)
    X = np.array(vecs, dtype=np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X[rng.permutation(len(X))]


def _make_uniform(seed=42, n=1000, d=32):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, d).astype(np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X


def _queries(X, seed, n=100):
    rng = np.random.RandomState(seed)
    qidx = rng.choice(len(X), n, replace=False)
    Q = X[qidx] * 0.9 + X[rng.choice(len(X), n)] * 0.1
    return Q / np.linalg.norm(Q, axis=1, keepdims=True)


# ---------------------------------------------------------------- (a) anillo

class TestRingDiversity:
    def test_far_edge_survives_alg4_pruning(self):
        """Unit test determinista del patron anillo sobre _select_neighbors.

        Geometria euclidea (D=32): q en el origen es el nodo que poda.
        - 9 nodos 'anillo' (idx 0..8): circulo de radio 0.3 centrado a
          distancia 3 de q => mutuamente a <= 0.6 << 3 de distancia entre si:
          para Alg.4 son REDUNDANTES entre ellos (sobrevive solo el primero).
        - nodo 'lejano' (idx 9): a distancia 6 de q (el mas lejano) pero en
          direccion ORTOGONAL al anillo => d(lejano, anillo) ~ 6.71 > 6:
          NO redundante => Alg.4 lo acepta.

        Con seleccion por cercania pura (Alg.3) los M mas cercanos son nodos
        del anillo (d ~ 3.0 < 6) y el lejano queda FUERA. Con Alg.4 el
        diversifica: [anillo_0, lejano] + keepPrunedConnections completa el
        grado con el anillo descartado mas cercano. El edge al lejano
        sobrevive la poda solo con Alg.4.
        """
        D = 32
        h = HNSWIndex(dim=D, M=3, metric="euclidean", seed=0)

        q = np.zeros(D, dtype=np.float32)  # nodo que poda (centro)

        ring_center = np.zeros(D, dtype=np.float32)
        ring_center[0] = 3.0
        e1 = np.zeros(D, dtype=np.float32)
        e1[1] = 1.0
        e2 = np.zeros(D, dtype=np.float32)
        e2[2] = 1.0
        ring = [
            (ring_center + 0.3 * np.cos(a) * e1 + 0.3 * np.sin(a) * e2)
            for a in np.linspace(0, 2 * np.pi, 9, endpoint=False)
        ]
        far_dir = np.zeros(D, dtype=np.float32)
        far_dir[3] = 1.0  # ortogonal al plano del anillo
        far = 6.0 * far_dir

        h._vectors = [v.astype(np.float32) for v in ring] + [far.astype(np.float32)]

        cands = [(h._distance(q, v), i) for i, v in enumerate(h._vectors)]
        M = 3

        # sanity del patron: cercania pura (Alg.3) descarta al lejano
        pure_closest = [i for _, i in sorted(cands)[:M]]
        assert 9 not in pure_closest, pure_closest

        # Alg.4: grado lleno (keepPrunedConnections) y el lejano sobrevive
        sel = h._select_neighbors(q, cands, M, 0)
        assert len(sel) == M, sel
        assert 9 in sel, f"Alg.4 debe conservar el edge al lejano: {sel}"
        # y el grafo sigue sin exhalar: los elegidos no son todos del anillo
        assert any(i != 9 for i in sel)

    def test_ring_probe_end_to_end(self):
        """Variante end-to-end: centro + anillo de 9 + lejano via build().

        El lejano debe quedar conectado y el grafo conexo en capa 0.
        """
        D = 32
        rng = np.random.RandomState(1)
        center = rng.randn(D) * 0.1
        base = rng.randn(D, 2).astype(np.float32)
        base, _ = np.linalg.qr(base)

        pts = [center]
        for ang in np.linspace(0, 2 * np.pi, 9, endpoint=False):
            direction = np.cos(ang) * base[:, 0] + np.sin(ang) * base[:, 1]
            pts.append(center + direction * 1.0)
        far_dir = rng.randn(D)
        far_dir = far_dir - base @ (base.T @ far_dir)  # ortogonal al plano
        far_dir /= np.linalg.norm(far_dir)
        pts.append(center + far_dir * 5.0)

        X = np.array(pts, dtype=np.float32)
        h = HNSWIndex(dim=D, M=3, ef_construction=40, ef_search=40, seed=3)
        h.build(X)

        far_nb = h._nodes[10].neighbors[0]
        assert len(far_nb) > 0, "el nodo lejano quedo aislado del grafo"

        seen = {h._entry_point}
        dq = deque([h._entry_point])
        while dq:
            cur = dq.popleft()
            for nb in h._nodes[cur].neighbors[0]:
                if nb not in seen:
                    seen.add(nb)
                    dq.append(nb)
        assert seen == set(range(len(X))), (
            f"grafo no conexo en capa 0: faltan {set(range(len(X))) - seen}"
        )


# ---------------------------------------------------- (b) corpus clusterizado

class TestClusteredRecall:
    def test_clustered_recall_at_least_095(self):
        X = _make_clustered(seed=7)
        Q = _queries(X, seed=7)
        h = HNSWIndex(dim=X.shape[1], M=8, ef_construction=60, ef_search=30, seed=7)
        h.build(X)
        rec = _recall(X, Q, h)
        assert rec >= 0.95, f"recall@10={rec:.4f} < 0.95 en corpus clusterizado"


# ------------------------------------------------------- (c) corpus uniforme

class TestUniformNoRegression:
    def test_uniform_recall_no_drop_gt_001(self):
        X = _make_uniform(seed=42)
        Q = _queries(X, seed=42)
        h = HNSWIndex(dim=X.shape[1], M=8, ef_construction=60, ef_search=30, seed=7)
        h.build(X)
        rec = _recall(X, Q, h)
        assert rec >= UNIFORM_BASELINE_RECALL - 0.01, (
            f"recall@10={rec:.4f} cae mas de 0.01 vs baseline "
            f"{UNIFORM_BASELINE_RECALL:.4f} en corpus uniforme"
        )


# -------------------------------------------------------- (d) grado contrato

class TestDegreeContract:
    @pytest.mark.parametrize("tag", ["clustered", "uniform", "real"])
    def test_degree_within_contract(self, tag):
        if tag == "real":
            path = "/tmp/m2mvs/embeddings.npy"
            if not os.path.exists(path):
                pytest.skip("dataset real no disponible")
            X = np.load(path).astype(np.float32)
        else:
            X = _make_clustered() if tag == "clustered" else _make_uniform()
        h = HNSWIndex(dim=X.shape[1], M=8, ef_construction=60, ef_search=30, seed=7)
        h.build(X)
        for node in h._nodes:
            for lev, nbrs in enumerate(node.neighbors):
                limit = h.M_max0 if lev == 0 else h.M
                assert len(nbrs) <= limit, (
                    f"{tag}: nodo {node.idx} nivel {lev} "
                    f"grado {len(nbrs)} > {limit}"
                )
