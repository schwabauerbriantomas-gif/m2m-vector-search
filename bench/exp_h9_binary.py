"""H9 "RaBitQ light": cuantizacion binaria + rescoring en NumPy puro.

Diseno: clusters sobre datos float (identico al pipeline m2m actual); el
dataset se guarda TAMBIEN binarizado (1 bit/dim) y la fase candidata corre
sobre los bits (Hamming, 32x menos bytes tocados); el rescoring float32
contra los originales de los candidatos recupera el recall (Qdrant: 0.98
recall@100 con 4x oversampling).
"""
import numpy as np
import time
import json
import os

rng = np.random.RandomState(42)
DIM, NQ, K = 384, 200, 10
N = 50_000
N_CLUSTERS = 100
centers = rng.randn(N_CLUSTERS, DIM).astype(np.float32)

LUT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint16)


def pack_bits(x: np.ndarray) -> np.ndarray:
    return np.packbits((x > 0).astype(np.uint8), axis=1)


def make_data(n, seed):
    r = np.random.RandomState(seed)
    assign = r.randint(0, N_CLUSTERS, size=n)
    data = centers[assign] + 0.35 * r.randn(n, DIM).astype(np.float32)
    data /= np.linalg.norm(data, axis=1, keepdims=True)
    q = centers[r.randint(0, N_CLUSTERS, size=NQ)] + 0.35 * r.randn(NQ, DIM).astype(np.float32)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    gt = np.argsort(2 - 2 * (q @ data.T), axis=1)[:, :K]
    return data, q, gt


def hamming(qb, db_, chunk=25):
    """[B, d8] x [n, d8] -> [B, n] por chunks (memoria acotada)."""
    n = db_.shape[0]
    out = np.empty((qb.shape[0], n), dtype=np.uint16)
    for i in range(0, qb.shape[0], chunk):
        xor = qb[i:i + chunk, None, :] ^ db_[None, :, :]
        out[i:i + chunk] = LUT[xor].sum(axis=2, dtype=np.uint16)
    return out


def bench_binary(data, queries, gt, n_probe=5, oversample_list=(1, 2, 4, 8)):
    from sklearn.cluster import KMeans
    t0 = time.perf_counter()
    km = KMeans(n_clusters=N_CLUSTERS, n_init=4, random_state=42).fit(data)
    t_build = time.perf_counter() - t0
    assignments = km.labels_  # [n]

    data_bits = pack_bits(data)
    cbits = pack_bits(km.cluster_centers_)
    q_bits = pack_bits(queries)

    results = []
    cd = hamming(q_bits, cbits)  # [B, n_clusters]
    probe = np.argpartition(cd, n_probe - 1, axis=1)[:, :n_probe]
    cand_mask = np.isin(assignments[np.newaxis, :], probe)  # [B, n]

    for os_factor in oversample_list:
        cand_k = min(K * os_factor, N)
        t0 = time.perf_counter()
        ham = hamming(q_bits, data_bits).astype(np.int32)
        ham = np.where(cand_mask, ham, 10**9)
        part = np.argpartition(ham, cand_k - 1, axis=1)[:, :cand_k]
        # rescoring float32 contra originales
        resc = 2 - 2 * np.einsum("bd,bkd->bk", queries, data[part])
        order = np.argsort(resc, axis=1)
        top = np.take_along_axis(part, order, axis=1)[:, :K]
        dt = time.perf_counter() - t0
        rec = np.mean([len(set(top[i].tolist()) & set(gt[i].tolist())) / K for i in range(NQ)])
        results.append({
            "mode": f"binary_os{os_factor}",
            "recall@10": round(float(rec), 4),
            "qps": round(NQ / dt, 1),
        })
    return results, t_build, data_bits.nbytes


def bench_float(data, queries, gt, n_probe=5):
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=N_CLUSTERS, n_init=4, random_state=42).fit(data)
    assignments = km.labels_
    m_sq = np.einsum("ij,ij->i", data, data)
    t0 = time.perf_counter()
    diff = queries[:, None, :] - km.cluster_centers_[None, :, :]
    cdist = np.sqrt(np.einsum("bkc,bkc->bk", diff, diff))
    probe = np.argpartition(cdist, n_probe - 1, axis=1)[:, :n_probe]
    cand_mask = np.isin(assignments[np.newaxis, :], probe)
    d2 = m_sq[None, :] - 2.0 * (queries @ data.T)
    d2 += np.einsum("ij,ij->i", queries, queries)[:, None]
    d2 = np.maximum(np.where(cand_mask, d2, 10**9), 0)
    part = np.argpartition(d2, K - 1, axis=1)[:, :K]
    dt = time.perf_counter() - t0
    rec = np.mean([len(set(part[i].tolist()) & set(gt[i].tolist())) / K for i in range(NQ)])
    return [{"mode": "float32_ivf", "recall@10": round(float(rec), 4), "qps": round(NQ / dt, 1)}]


if __name__ == "__main__":
    print(f"Generando {N} x {DIM} ...")
    data, queries, gt = make_data(N, 2)
    out = {"n": N, "dim": DIM, "nq": NQ, "n_clusters": N_CLUSTERS, "results": []}

    print("float32 IVF (control)...")
    out["results"] += bench_float(data, queries, gt)

    print("binary + rescoring...")
    res, t_build, bits_bytes = bench_binary(data, queries, gt)
    out["results"] += res
    out["build_kmeans_s"] = round(t_build, 2)
    out["mem_float32_mb"] = round(data.nbytes / 1e6, 2)
    out["mem_binary_mb"] = round(bits_bytes / 1e6, 2)
    out["compression"] = round(data.nbytes / bits_bytes, 1)

    with open(os.path.join(os.path.dirname(__file__), "h9_binary.json"), "w") as f:
        json.dump(out, f, indent=2)
    for r in out["results"]:
        print(f"  {r['mode']:16s} recall@10={r['recall@10']:.4f}  QPS={r['qps']}")
    print(f"build kmeans: {out['build_kmeans_s']}s | memoria: float32={out['mem_float32_mb']}MB binario={out['mem_binary_mb']}MB ({out['compression']}x)")
