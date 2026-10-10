"""H10: cuantizacion scalar 8-bit (SQ8) + rescoring — la version correcta de H9.

H9 (1-bit) refutada: Hamming sin BLAS pierde el orden (recall 0.098 fase 1).
SQ8 mantiene el orden relativo mucho mejor (256 niveles vs 2) y la fase 1
puede correr como GEMM int8 (BLAS) o float32 sobre bytes-as-float (4x menos
ancho de banda). Protocolo: misma estructura que exp_h9_binary.py.
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


def make_data(n, seed):
    r = np.random.RandomState(seed)
    assign = r.randint(0, N_CLUSTERS, size=n)
    data = centers[assign] + 0.35 * r.randn(n, DIM).astype(np.float32)
    data /= np.linalg.norm(data, axis=1, keepdims=True)
    q = centers[r.randint(0, N_CLUSTERS, size=NQ)] + 0.35 * r.randn(NQ, DIM).astype(np.float32)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    gt = np.argsort(2 - 2 * (q @ data.T), axis=1)[:, :K]
    return data, q, gt


def quantize_sq8(x):
    """[n, d] float32 -> (uint8 [n, d], scales [n], zeros [n])"""
    mn = x.min(axis=1)
    mx = x.max(axis=1)
    scale = (mx - mn) / 255.0
    scale = np.where(scale == 0, 1e-8, scale)
    q = np.clip(np.round((x - mn[:, None]) / scale[:, None]), 0, 255).astype(np.uint8)
    return q, scale.astype(np.float32), mn.astype(np.float32)


def sq8_l2(qx, qs, qz, dx, ds, dz, chunk=50):
    """Aprox L2 entre queries float y data SQ8 via dequantizacion por chunks.
    d2 ~= sum((qx - (dx*s+z))^2). Los datos se dequantizan UNA vez por chunk
    (float32), queries quedan exactas => error acotado al data term."""
    n = dx.shape[0]
    out = np.empty((qx.shape[0], n), dtype=np.float32)
    qq = np.einsum("ij,ij->i", qx, qx)
    for i in range(0, n, chunk):
        deq = dx[i:i+chunk].astype(np.float32) * ds[i:i+chunk, None] + dz[i:i+chunk, None]
        d2 = qq[:, None] - 2.0 * (qx @ deq.T) + np.einsum("ij,ij->i", deq, deq)[None, :]
        out[:, i:i+chunk] = d2
    return out


def main():
    print(f"Generando {N} x {DIM} ...")
    data, queries, gt = make_data(N, 2)
    out = {"n": N, "dim": DIM, "nq": NQ, "n_clusters": N_CLUSTERS, "results": []}

    from sklearn.cluster import KMeans
    t0 = time.perf_counter()
    km = KMeans(n_clusters=N_CLUSTERS, n_init=4, random_state=42).fit(data)
    assignments = km.labels_
    m_sq = np.einsum("ij,ij->i", data, data)

    # control float32 (mismo que h9)
    t0 = time.perf_counter()
    diff = queries[:, None, :] - km.cluster_centers_[None, :, :]
    cdist = np.sqrt(np.einsum("bkc,bkc->bk", diff, diff))
    probe = np.argpartition(cdist, 4, axis=1)[:, :5]
    cand_mask = np.isin(assignments[np.newaxis, :], probe)
    d2 = m_sq[None, :] - 2.0 * (queries @ data.T)
    d2 += np.einsum("ij,ij->i", queries, queries)[:, None]
    d2 = np.maximum(np.where(cand_mask, d2, 10**9), 0)
    part = np.argpartition(d2, K - 1, axis=1)[:, :K]
    dt = time.perf_counter() - t0
    rec = np.mean([len(set(part[i].tolist()) & set(gt[i].tolist())) / K for i in range(NQ)])
    out["results"].append({"mode": "float32_ivf", "recall@10": round(float(rec), 4), "qps": round(NQ / dt, 1)})
    print(f"float32_ivf: recall={rec:.4f} QPS={NQ/dt:.0f}")

    # SQ8: fase 1 dequantizada-por-chunk + rescoring float32
    t0 = time.perf_counter()
    data_q, ds, dz = quantize_sq8(data)
    t_q = time.perf_counter() - t0

    for oversample in (2, 4):
        cand_k = K * oversample
        t0 = time.perf_counter()
        diff = queries[:, None, :] - km.cluster_centers_[None, :, :]
        cdist = np.sqrt(np.einsum("bkc,bkc->bk", diff, diff))
        probe = np.argpartition(cdist, 4, axis=1)[:, :5]
        cand_mask = np.isin(assignments[np.newaxis, :], probe)
        d2q = sq8_l2(queries, None, None, data_q, ds, dz)
        d2q = np.where(cand_mask, d2q, 10**9)
        part = np.argpartition(d2q, cand_k - 1, axis=1)[:, :cand_k]
        resc = 2 - 2 * np.einsum("bd,bkd->bk", queries, data[part])
        order = np.argsort(resc, axis=1)
        top = np.take_along_axis(part, order, axis=1)[:, :K]
        dt = time.perf_counter() - t0
        rec = np.mean([len(set(top[i].tolist()) & set(gt[i].tolist())) / K for i in range(NQ)])
        out["results"].append({"mode": f"sq8_os{oversample}", "recall@10": round(float(rec), 4), "qps": round(NQ / dt, 1)})
        print(f"sq8_os{oversample}: recall={rec:.4f} QPS={NQ/dt:.0f}")

    out["mem_float32_mb"] = round(data.nbytes / 1e6, 2)
    out["mem_sq8_mb"] = round(data_q.nbytes / 1e6, 2)
    out["quant_time_s"] = round(t_q, 2)
    with open(os.path.join(os.path.dirname(__file__), "h10_sq8.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"memoria: {out['mem_float32_mb']}MB -> {out['mem_sq8_mb']}MB (4x)")


if __name__ == "__main__":
    main()
