"""H11: RaBitQ (arXiv:2405.12497) en NumPy — estimador unbiased vs Hamming (H9).

Núcleo del paper (Gao & Long, SIGMOD'24):
  1. Por cluster c: o := (o_r - c)/||o_r - c|| (residual unitario)
  2. Rotación aleatoria ortogonal P compartida; código binario de P·o
  3. Estimador:  <o,q> ~= <x̄_bin, q̄> / <x̄, o>   con <x̄,o> escalar guardado
     (unbiased, error O(1/sqrt(D)) — vs Hamming que no estima nada)
  4. d2(q_r, o_r) = -2||o_r-c||·||q_r-c||·<o,q> + K

Lo que H9 refutó fue el ORDEN de Hamming (recall fase-1 0.098). El estimador
RaBitQ estima el producto interno real. Fase-1 sobre SQ8 también (control).
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


def make_data(n, seed):
    r = np.random.RandomState(seed)
    assign = r.randint(0, N_CLUSTERS, size=n)
    data = centers[assign] + 0.35 * r.randn(n, DIM).astype(np.float32)
    data /= np.linalg.norm(data, axis=1, keepdims=True)
    q = centers[r.randint(0, N_CLUSTERS, size=NQ)] + 0.35 * r.randn(NQ, DIM).astype(np.float32)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    gt = np.argsort(2 - 2 * (q @ data.T), axis=1)[:, :K]
    return data, q, gt


def main():
    print(f"Generando {N} x {DIM} ...")
    data, queries, gt = make_data(N, 2)

    from sklearn.cluster import KMeans
    t0 = time.perf_counter()
    km = KMeans(n_clusters=N_CLUSTERS, n_init=4, random_state=42).fit(data)
    assignments = km.labels_
    print(f"kmeans: {time.perf_counter()-t0:.1f}s")

    # --- RaBitQ index build ---
    t0 = time.perf_counter()
    Q, _ = np.linalg.qr(rng.randn(DIM, DIM).astype(np.float32))  # P ortogonal
    P = Q.T.astype(np.float32)
    centroids = np.asarray(km.cluster_centers_, dtype=np.float32)
    cvec = centroids[assignments]                    # [n, D]
    res = data - cvec
    rnorm = np.linalg.norm(res, axis=1).astype(np.float32)
    o_unit = res / rnorm[:, None]
    rotated = o_unit @ P.T
    codes01 = (rotated > 0).astype(np.uint8)         # [n, D] en {0,1}
    packed = np.packbits(codes01, axis=1)            # [n, D/8] — 2.4MB
    xbar = np.where(codes01 > 0, 1.0, -1.0) / np.sqrt(DIM)  # código bivalued ±1/√D
    inner_bar = np.einsum("nd,nd->n", rotated, xbar).astype(np.float32)  # o_rot @ xbar (paper)
    t_build = time.perf_counter() - t0
    print(f"rabitq build: {t_build:.1f}s | packed {packed.nbytes/1e6:.1f}MB (float32: {data.nbytes/1e6:.0f}MB)")

    # queries rotadas
    q_rot = queries @ P.T  # [B, D]

    # --- fase 1 RaBitQ CORRECTA (estimador del paper):
    #   <o,q> ~ <xbar, q_rot>/<xbar_o_rot, xbar>, con q en frame rotado y
    #   q = residual del query vs el CENTROIDE DEL PUNTO (no unit).
    t0 = time.perf_counter()
    ip_hat = np.zeros((NQ, N), dtype=np.float32)
    cdist = ((queries[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
    probe_matrix = np.argpartition(cdist, 4, axis=1)[:, :5]  # [B, 5]
    for c in range(N_CLUSTERS):
        bs = np.where(probe_matrix == c)[0]
        if len(bs) == 0:
            continue
        pts = np.where(assignments == c)[0]
        q_res = queries[bs] - centroids[c]              # residual del query vs SU centroide sondeado
        q_rot_res = q_res @ P.T                          # en frame rotado
        ip_hat[np.ix_(bs, pts)] = (xbar[pts] @ q_rot_res.T).T   # <xbar, q_rot>
    ip_hat /= inner_bar[None, :]                         # ~ <o_unit, q_res> (lleva ||q_res|| dentro)
    # d2 = ||q-c||^2 + ||o_r||^2 - 2||o_r|| * <o_unit, q_res>
    #    = qn^2 + rnorm^2 - 2 rnorm * ip_hat   (ip_hat ya trae ||q_res||)
    qn_sq = (queries**2).sum(axis=1)[:, None] + (cvec**2).sum(axis=1)[None, :] - 2.0*(queries @ cvec.T)
    d2_rabitq = np.maximum(qn_sq + rnorm[None, :] ** 2 - 2.0 * rnorm[None, :] * ip_hat, 0.0)
    t_phase = time.perf_counter() - t0

    for name, d2p in [("rabitq_os1", d2_rabitq)]:
        top = np.argpartition(d2p, K - 1, axis=1)[:, :K]
        rec = np.mean([len(set(top[i].tolist()) & set(gt[i].tolist())) / K for i in range(NQ)])
        print(f"{name}: recall fase-1={rec:.4f} ({NQ/t_phase* (N/ (N)):.0f} QPS eq)")

    # fase-1 recall de SQ8 para comparar orden de estimadores
    def quantize_sq8(x):
        mn = x.min(axis=1); mx = x.max(axis=1)
        scale = np.where((mx - mn) == 0, 1e-8, (mx - mn) / 255.0)
        q = np.clip(np.round((x - mn[:, None]) / scale[:, None]), 0, 255).astype(np.uint8)
        return q, scale.astype(np.float32), mn.astype(np.float32)
    dq, ds, dz = quantize_sq8(data)
    deq = dq.astype(np.float32) * ds[:, None] + dz[:, None]
    d2_sq8 = np.einsum("nd,nd->n", deq, deq)[None, :] - 2.0 * (queries @ deq.T) \
        + np.einsum("bd,bd->b", queries, queries)[:, None]
    top = np.argpartition(d2_sq8, K - 1, axis=1)[:, :K]
    rec_sq8 = np.mean([len(set(top[i].tolist()) & set(gt[i].tolist())) / K for i in range(NQ)])
    print(f"sq8 fase-1  : recall={rec_sq8:.4f}")

    # end-to-end: rabitq os4 + rescoring float32 (con candidate union IVF)
    cand_mask = np.isin(assignments[np.newaxis, :], np.argpartition(
        ((queries[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2), 4, axis=1)[:, :5])
    for os_f in (2, 4):
        cand_k = min(K * os_f, N)
        t0 = time.perf_counter()
        d2p = np.where(cand_mask, d2_rabitq, 10**9)
        part = np.argpartition(d2p, cand_k - 1, axis=1)[:, :cand_k]
        resc = 2 - 2 * np.einsum("bd,bkd->bk", queries, data[part])
        order = np.argsort(resc, axis=1)
        top = np.take_along_axis(part, order, axis=1)[:, :K]
        dt = time.perf_counter() - t0
        rec = np.mean([len(set(top[i].tolist()) & set(gt[i].tolist())) / K for i in range(NQ)])
        print(f"rabitq_os{os_f} e2e: recall={rec:.4f} QPS={NQ/dt:.0f}")

    out = {
        "rabitq_phase1_recall": None, "sq8_phase1_recall": round(float(rec_sq8), 4),
        "build_s": round(t_build, 2), "mem_packed_mb": round(packed.nbytes / 1e6, 2),
    }
    with open(os.path.join(os.path.dirname(__file__), "h11_rabitq.json"), "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
