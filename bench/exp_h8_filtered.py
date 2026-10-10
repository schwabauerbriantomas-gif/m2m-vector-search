"""H8: busqueda filtrada (pre-filter sobre candidatos del indice).

Estado actual (medido, 10k x 384, selectividad 25%): el filtro se aplica
POST-filter sobre el top-k del indice => recall 0.26 y resultados vacios.
Protocolo ACORN-lite: pre-filtrar la UNION de candidatos del indice (mask
vectorizada contra los ids de metadata) y tomar top-k de los que pasan.
Sin grafos auxiliares ni efes dinamicas: el pruning de clusters de IVF ya
acota candidatos; el filtro corre sobre esa union.
"""
import numpy as np
import sys
import os
import time
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from m2m import SimpleVectorDB

rng = np.random.RandomState(42)
DIM, NQ, K = 384, 200, 10
N = 10_000
N_CLUSTERS = 50
centers = rng.randn(N_CLUSTERS, DIM).astype(np.float32)


def main():
    assign = rng.randint(0, N_CLUSTERS, size=N)
    data = centers[assign] + 0.35 * rng.randn(N, DIM).astype(np.float32)
    data /= np.linalg.norm(data, axis=1, keepdims=True)
    queries = centers[rng.randint(0, N_CLUSTERS, size=NQ)] + 0.35 * rng.randn(NQ, DIM).astype(np.float32)
    queries /= np.linalg.norm(queries, axis=1, keepdims=True)
    db = SimpleVectorDB(latent_dim=DIM, enable_lsh_fallback=False, max_splats=N, n_probe=5)
    meta = [{"cat": f"c{i % 4}"} for i in range(N)]
    db.add(ids=[str(i) for i in range(N)], vectors=data, metadata=meta)

    out = {}
    for sel, mod in [("p25", 4), ("p50", 2), ("p05", 20)]:
        mask = np.array([i % mod == 0 for i in range(N)])
        # GT filtrado: distancias +inf a los que no pasan
        d2 = 2 - 2 * (queries @ data.T)
        gt = np.argsort(d2 + np.where(mask[None, :], 0, 10**6), axis=1)[:, :K]

        t0 = time.perf_counter()
        # --- medir el pre-filter nuevo via search_batch_filtered ---
        t0 = time.perf_counter()
        got = db.search_batch_filtered(queries, k=K, filter_fn=lambda i: bool(mask[i]))
        dt = time.perf_counter() - t0
        rec = float(np.mean([len(set(got[i].tolist()) & set(gt[i].tolist())) / K for i in range(NQ)]))
        # llenura: cuantos queries devuelven k resultados
        fill = float(np.mean([len([j for j in got[i] if j >= 0]) / K for i in range(NQ)]))
        out[sel] = {"recall@10": round(rec, 4), "qps": round(NQ / dt, 1), "fill": round(fill, 3)}
        print(f"selectividad {sel}: recall={rec:.4f} QPS={NQ/dt:.0f} fill={fill:.2f}")

    with open(os.path.join(os.path.dirname(__file__), "h8_filtered.json"), "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
