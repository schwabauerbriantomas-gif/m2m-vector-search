"""Tests de SQ8 espectral (fase 1 sobre X@P top-r, rescoring mmap full-D).

Cubre el punto 4 del plan de integracion (probe E):
  a) backward-compat: spectral_r=None -> bits IDENTICOS al flujo historico.
  b) recall igualado en corpus real con rescore full-D.
  c) RAM de fase 1 ~ r/D del store full-D.
  d) determinismo: dos builds con la misma data -> codigos identicos.
"""
import os

import numpy as np
import pytest

from m2m.sq8_disk import SQ8DiskStore, _knee_rank

REAL_NPY = "/tmp/m2mvs/embeddings.npy"


def _clustered(n, dim, n_clusters, seed=42):
    rng = np.random.RandomState(seed)
    centers = rng.randn(n_clusters, dim).astype(np.float32)
    data = centers[rng.randint(0, n_clusters, size=n)] + 0.35 * rng.randn(n, dim).astype(np.float32)
    data /= np.linalg.norm(data, axis=1, keepdims=True)
    return data


def _quantize_reference(x: np.ndarray):
    """Formula historica de SQ8DiskStore.add (pre-espectral), verbatim."""
    mn = x.min(axis=1)
    mx = x.max(axis=1)
    scale = (mx - mn) / 255.0
    scale = np.where(scale == 0, np.float32(1e-8), scale).astype(np.float32)
    q = np.clip(np.round((x - mn[:, None]) / scale[:, None]), 0, 255).astype(np.uint8)
    return q, scale, mn.astype(np.float32)


def _recall_at_k(out, gt, k=10):
    return float(np.mean([
        len(set(out[i][:k].tolist()) & set(gt[i].tolist())) / k
        for i in range(len(out))
    ]))


# ---------------------------------------------------------------- a) bitwise
def test_backward_compat_none_is_bitwise_identical(tmp_path):
    """spectral_r=None debe reproducir bit a bit el flujo historico."""
    data = _clustered(2000, 64, 15, seed=11)
    st = SQ8DiskStore(dim=64, dump_dir=str(tmp_path))  # default spectral_r=None
    assert st.spectral_r is None
    st.add(data)
    st.flush()
    q_ref, scale_ref, zero_ref = _quantize_reference(data)
    assert np.array_equal(st._q, q_ref)
    assert np.array_equal(st._scale, scale_ref)
    assert np.array_equal(st._zero, zero_ref)
    # mismos bits que el constructor explicito spectral_r=None
    st2 = SQ8DiskStore(dim=64, spectral_r=None)
    st2.add(data)
    assert np.array_equal(st2._q, st._q)
    # y tambien via build_from_float_npy (camino por chunks)
    xp = str(tmp_path / "X.npy")
    np.save(xp, data)
    st3 = SQ8DiskStore.build_from_float_npy(xp, dump_dir=str(tmp_path / "b"))
    assert np.array_equal(st3._q, q_ref)
    assert np.array_equal(st3._scale, scale_ref)
    assert np.array_equal(st3._zero, zero_ref)


def test_spectral_r_validation():
    for bad in (0, -3, 2.5, True, 1.0):
        with pytest.raises(ValueError):
            SQ8DiskStore(dim=8, spectral_r=bad)
    SQ8DiskStore(dim=8, spectral_r=4)  # valido
    SQ8DiskStore(dim=8, spectral_r="auto")  # valido


# ------------------------------------------------------------- b) recall
@pytest.mark.skipif(not os.path.exists(REAL_NPY), reason="corpus real no disponible")
def test_recall_real_corpus_spectral_matches_full(tmp_path):
    """recall@10 espectral (r=16, rescore full-D) iguala al modo actual.

    El corpus real es full-rank (auto->r=384), asi que la fase 1 en r=16
    necesita un K' mayor que el probe E sintetico: se busca el K' minimo
    de una escala fija que alcanza paridad (recall >= full - 0.005).
    """
    X = np.load(REAL_NPY)
    n, dim = X.shape
    xp = str(tmp_path / "X.npy")
    np.save(xp, X)
    st_full = SQ8DiskStore.build_from_float_npy(xp, dump_dir=str(tmp_path / "full"))
    st_proj = SQ8DiskStore.build_from_float_npy(
        xp, dump_dir=str(tmp_path / "proj"), spectral_r=16)
    rng = np.random.RandomState(3)
    Q = X[rng.choice(n, 100, replace=False)]
    S = 2 * Q @ X.T - (X ** 2).sum(1)[None] - (Q ** 2).sum(1)[:, None]
    gt = np.argsort(-S, axis=1)[:, :10]
    rec_full = _recall_at_k(st_full.search_batch(Q, k=10, oversample=3), gt)
    assert rec_full >= 0.99  # sanity: el modo actual ya esta en paridad
    ladder = [3, 5, 10, 20, 30, 50]  # K' = 30, 50, 100, 200, 300, 500
    recalls = [_recall_at_k(st_proj.search_batch(Q, k=10, oversample=ov), gt)
               for ov in ladder]
    # el recall mejora monotonicamente con K' (mas candidatos -> mas recall)
    assert all(b >= a - 1e-9 for a, b in zip(recalls, recalls[1:]))
    best = max(recalls)
    assert best >= rec_full - 0.005, (
        f"mejor recall espectral {best:.4f} < full {rec_full:.4f} - 0.005 "
        f"(escala K'={ladder})"
    )


def test_recall_synthetic_spectral_roundtrip(tmp_path):
    """Corpus low-rank + r=auto: el rescore full-D recupera el recall exacto."""
    rng = np.random.RandomState(7)
    D, dr, N = 64, 6, 1500
    basis = rng.randn(D, dr).astype(np.float32)
    X = np.stack([basis @ (rng.randn(dr) * 2.0 + rng.randn(dr) * 0.4)
                  for _ in range(N)]).astype(np.float32)
    X += rng.randn(N, D).astype(np.float32) * 0.5
    xp = str(tmp_path / "X.npy")
    np.save(xp, X)
    st = SQ8DiskStore.build_from_float_npy(
        xp, dump_dir=str(tmp_path / "auto"), spectral_r="auto")
    assert st._rdim <= 12, f"auto detecto r={st._rdim}, esperaba ~dr"
    Q = X[rng.choice(N, 40, replace=False)] + 0.01
    S = 2 * Q @ X.T - (X ** 2).sum(1)[None] - (Q ** 2).sum(1)[:, None]
    gt = np.argsort(-S, axis=1)[:, :10]
    out = st.search_batch(Q, k=10, oversample=3)
    assert _recall_at_k(out, gt) >= 0.95


# ----------------------------------------------------------------- c) RAM
def test_phase1_ram_reduction(tmp_path):
    """RAM fase 1 con r=16 sobre D=384 debe ser ~r/D del store full-D."""
    rng = np.random.RandomState(5)
    D, r, N = 384, 16, 3000
    basis = rng.randn(D, 20).astype(np.float32)
    X = (basis @ rng.randn(20, N)).T.astype(np.float32)
    X += rng.randn(N, D).astype(np.float32) * 0.3
    xp = str(tmp_path / "X.npy")
    np.save(xp, X)
    st_full = SQ8DiskStore.build_from_float_npy(xp, dump_dir=str(tmp_path / "full"))
    st_proj = SQ8DiskStore.build_from_float_npy(
        xp, dump_dir=str(tmp_path / "proj"), spectral_r=r)
    assert st_proj._q.shape == (N, r)
    # RAM fase 1 = N*(r+8) bytes (uint8 r-dim + scale/zero float32); la
    # relacion exacta es (r+8)/(D+8): con r=16, D=384 -> 16.33x (probe E).
    assert st_proj.ram_bytes() <= N * (r + 8) * 1.001
    assert st_proj.ram_bytes() <= st_full.ram_bytes() * (r + 8) / (D + 8) * 1.02
    assert st_full.ram_bytes() / st_proj.ram_bytes() >= 0.9 * (D + 8) / (r + 8)


def test_rescore_uses_full_dim_originals(tmp_path):
    """El mmap de rescoring es float32 [n, D] con los vectores ORIGINALES."""
    X = _clustered(500, 48, 8, seed=3)
    st = SQ8DiskStore(dim=48, dump_dir=str(tmp_path), spectral_r=8)
    st.add(X)
    st.flush()
    assert st._float_mm is not None
    assert st._float_mm.shape == (500, 48)
    assert np.allclose(np.asarray(st._float_mm), X, atol=1e-6)
    assert st._q.shape == (500, 8)  # codigos en el subespacio


# --------------------------------------------------------- d) determinismo
def test_determinism_same_seed_same_codes(tmp_path):
    data = _clustered(1200, 96, 12, seed=99)
    codes = []
    for i in range(2):
        st = SQ8DiskStore(dim=96, spectral_r=16)
        st.add(data)
        st.flush(str(tmp_path / f"d{i}"))
        codes.append((st._q.copy(), st._scale.copy(), st._zero.copy(),
                      st._P.copy() if st._P is not None else None))
    (q0, s0, z0, P0), (q1, s1, z1, P1) = codes
    assert np.array_equal(q0, q1)
    assert np.array_equal(s0, s1)
    assert np.array_equal(z0, z1)
    assert P0 is not None and P1 is not None and np.array_equal(P0, P1)


def test_reload_preserves_spectral_search(tmp_path):
    X = _clustered(600, 64, 10, seed=13)
    st = SQ8DiskStore(dim=64, dump_dir=str(tmp_path), spectral_r=16)
    st.add(X)
    st.flush()
    out_a = st.search_batch(X[:20], k=10, oversample=3)
    st2 = SQ8DiskStore.load(str(tmp_path))
    assert st2._rdim == 16 and st2.dim == 64
    out_b = st2.search_batch(X[:20], k=10, oversample=3)
    assert np.array_equal(out_a, out_b)


def test_online_add_after_flush_spectral(tmp_path):
    """add() post-flush extiende codigos + float32 sin corromper lo previo."""
    X = _clustered(400, 64, 8, seed=21)
    st = SQ8DiskStore(dim=64, dump_dir=str(tmp_path), spectral_r=16)
    st.add(X)
    st.flush()
    extra = X[:100] + 0.05
    st.add(extra)
    st.flush()
    assert len(st) == 500
    mm = np.asarray(st._float_mm)
    assert mm.shape == (500, 64)
    assert np.allclose(mm[:400], X, atol=1e-6)
    assert np.allclose(mm[400:], extra, atol=1e-6)
    # recall contra el corpus extendido
    X2 = np.vstack([X, extra])
    rng = np.random.RandomState(4)
    Q = X2[rng.choice(len(X2), 30, replace=False)]
    S = 2 * Q @ X2.T - (X2 ** 2).sum(1)[None] - (Q ** 2).sum(1)[:, None]
    gt = np.argsort(-S, axis=1)[:, :10]
    out = st.search_batch(Q, k=10, oversample=10)
    assert _recall_at_k(out, gt) >= 0.9


# ------------------------------------------------------------ knee/auto
def test_knee_rank_sanity():
    # low-rank + ruido: rodilla en 3
    assert _knee_rank(np.array([10.0, 8.0, 6.0, 0.01, 0.01, 0.01])) == 3
    # espectro plano: sin rodilla -> dimension completa
    assert _knee_rank(np.ones(50)) == 50
    # un solo componente
    assert _knee_rank(np.array([5.0])) == 1


def test_auto_r_on_real_corpus(tmp_path):
    if not os.path.exists(REAL_NPY):
        pytest.skip("corpus real no disponible")
    X = np.load(REAL_NPY)
    xp = str(tmp_path / "X.npy")
    np.save(xp, X)
    st = SQ8DiskStore.build_from_float_npy(
        xp, dump_dir=str(tmp_path / "auto"), spectral_r="auto")
    assert 8 <= st._rdim <= 384
    # P ortonormal (columnas de eigh)
    assert st._P is not None
    np.testing.assert_allclose(st._P.T @ st._P, np.eye(st._rdim), atol=1e-4)
