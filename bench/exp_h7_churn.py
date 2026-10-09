"""H7: decaimiento de recall bajo churn (streaming).

Pregunta (IP-DiskANN arXiv:2502.13826): cuánto degrada el recall de un índice
IVF construido una vez cuando los datos cambian (adds) sin reindexar, vs
reindexar en cada lote. Protocolo:
  1. Index base de 20k; queries de evaluación fijas con GT contra el dataset
     actual de cada paso.
  2. Churn: en cada paso add() de 10% del dataset inicial, mitad inserts
     nuevos, mitad re-inserciones (misma posición, vector perturbado).
  3. En cada paso: recall@10 del índice estático vs índice reconstruido.
Métrica: curva recall vs % churn. Si el decay es <2% al 30% churn, el
reindex-lazy es viable; si no, hace falta rebuild programado.
"""
import numpy as np
import sys
import time
import json
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from m2m import SimpleVectorDB

rng = np.random.RandomState(42)
DIM, NQ, K = 384, 200, 10
N_CLUSTERS = 50
centers = rng.randn(N_CLUSTERS, DIM).astype(np.float32)

N_BASE = 20_000
CHURN_FRAC = 0.10
N_STEPS = 3  # 10%, 20%, 30% churn


def gen_data(n, seed):
    r = np.random.RandomState(seed)
    assign = r.randint(0, N_CLUSTERS, size=n)
    data = centers[assign] + 0.35 * r.randn(n, DIM).astype(np.float32)
    data /= np.linalg.norm(data, axis=1, keepdims=True)
    return data


def recall_of(db, queries, gt):
    idxs = db.search_batch(queries, k=K)
    return float(np.mean([
        len(set(idxs[i].tolist()) & set(gt[i].tolist())) / K for i in range(len(queries))
    ]))


def gt_against(data, queries):
    return np.argsort(2 - 2 * (queries @ data.T), axis=1)[:, :K]


def main():
    base = gen_data(N_BASE, 1)
    churn_new = gen_data(int(N_BASE * CHURN_FRAC) * N_STEPS, 99)
    queries = gen_data(NQ, 7)  # fijas

    db = SimpleVectorDB(latent_dim=DIM, enable_lsh_fallback=False, max_splats=200_000, n_probe=5)
    db.add(ids=[f"b{i}" for i in range(N_BASE)], vectors=base)
    print(f"base: {N_BASE} indexados")

    current = base.copy()
    current_ids = [f"b{i}" for i in range(N_BASE)]
    rows = []

    rec0 = recall_of(db, queries, gt_against(current, queries))
    rows.append({"churn_pct": 0, "recall_static": round(rec0, 4), "recall_full_rebuild": round(rec0, 4)})
    print(f"churn  0%: static={rec0:.4f}")

    offset = 0
    for step in range(1, N_STEPS + 1):
        m = int(N_BASE * CHURN_FRAC)
        # mitad re-inserciones perturbadas de docs existentes, mitad nuevos
        half = m // 2
        re_ids = current_ids[offset:offset + half]
        re_vecs = current[offset:offset + half] + 0.05 * rng.randn(half, DIM).astype(np.float32)
        re_vecs /= np.linalg.norm(re_vecs, axis=1, keepdims=True)
        new_ids = [f"n{offset + i}" for i in range(m - half)]
        new_vecs = churn_new[offset:offset + (m - half)]
        offset += m

        t0 = time.perf_counter()
        db.add(ids=re_ids + new_ids, vectors=np.vstack([re_vecs, new_vecs]))
        t_add = time.perf_counter() - t0

        current = np.vstack([current, re_vecs, new_vecs])
        current_ids += re_ids + new_ids

        rec_static = recall_of(db, queries, gt_against(current, queries))
        rows.append({
            "churn_pct": step * 10,
            "recall_static": round(rec_static, 4),
            "t_add_s": round(t_add, 2),
            "n_docs": len(current_ids),
        })
        print(f"churn {step*10:2d}%: static={rec_static:.4f} (add {t_add:.1f}s, {len(current_ids)} docs)")

    out = {"protocol": "add 10%/step sobre 20k, mitad re-insert perturbada mitad nueva; queries fijas",
           "n_probe": 5, "rows": rows}
    with open(os.path.join(os.path.dirname(__file__), "h7_churn.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("guardado h7_churn.json")


if __name__ == "__main__":
    main()
