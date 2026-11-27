"""Regresiones del audit r2 para SQ8DiskStore (pack P12a).

r2-01 recall@10 colapsa con el oversample por defecto de la fase espectral,
r2-02 _knee_rank inventa una rodilla sobre espectros Marchenko-Pastur planos,
r2-05 add() acepta NaN/Inf y los codifica con scale NaN/inf,
r2-06 search_batch materializa la matriz de distancias [len(Q), n] entera.

GT por fuerza bruta en float64. Datos sinteticos clustered con queries
IN-distribution (misma mezcla que el corpus): no hay datasets reales en esta
maquina. El stream del RNG es el de la sonda del auditor (work/r2/p7_followup.py).
"""

import json
import tracemalloc

import numpy as np
import pytest

import m2m.sq8_disk as sq8mod
from m2m.sq8_disk import SQ8DiskStore, _knee_rank

N, D, K = 20000, 384, 10


def _clustered_corpus(n=N, dim=D, n_clusters=50, n_queries=200):
    rng = np.random.default_rng(0)
    centers = rng.standard_normal((n_clusters, dim)).astype(np.float32) * 3.0
    labels = rng.integers(0, n_clusters, n)
    X = (centers[labels] + 0.3 * rng.standard_normal((n, dim))).astype(np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    ql = rng.integers(0, n_clusters, n_queries)
    Q = (centers[ql] + 0.3 * rng.standard_normal((n_queries, dim))).astype(np.float32)
    Q /= np.linalg.norm(Q, axis=1, keepdims=True)
    return X, Q


def _gt(q, x, k=K):
    q64, x64 = q.astype(np.float64), x.astype(np.float64)
    g = np.maximum((q64**2).sum(1)[:, None] + (x64**2).sum(1)[None, :] - 2 * q64 @ x64.T, 0)
    return np.argsort(g, axis=1)[:, :k]


def _recall(out, gt, k=K):
    return float(np.mean([len(set(o.tolist()) & set(g.tolist())) / k for o, g in zip(out, gt)]))


def _build(spec, data, dump_dir, dim=None):
    st = SQ8DiskStore(dim=dim or data.shape[1], dump_dir=dump_dir, spectral_r=spec)
    st.add(data)
    st.flush(dump_dir)
    return st


def _sample_spectrum(n, dim, seed=1):
    a = np.random.default_rng(seed).standard_normal((n, dim))
    return np.linalg.eigvalsh(np.cov(a, rowvar=False))[::-1]


@pytest.fixture(scope="module")
def clustered_case(tmp_path_factory):
    X, Q = _clustered_corpus()
    root = tmp_path_factory.mktemp("p12a")
    stores = {spec: _build(spec, X, str(root / f"cl_{spec}")) for spec in (None, 16, "auto")}
    return X, Q, _gt(Q, X), stores


# ----------------------------------------------------------------- r2-02
@pytest.mark.parametrize("n,dim", [(20000, 384), (3000, 128), (2000, 256), (500, 64)])
def test_knee_rank_mp_noise_spectrum_returns_full_dim(n, dim):
    """Espectro Marchenko-Pastur puro (ruido de muestreo, sin senal): no hay
    rodilla -> dimension completa, conforme al docstring."""
    ev = _sample_spectrum(n, dim)
    assert _knee_rank(ev, n_samples=n) == dim


def test_knee_rank_smooth_decay_has_no_knee():
    ev = np.linspace(2.0, 1.0, 384)  # decaimiento lineal: convexo, sin codo
    assert _knee_rank(ev) == 384
    assert _knee_rank(ev, n_samples=20000) == 384
    assert _knee_rank(np.ones(50)) == 50  # plano exacto: contrato previo
    assert _knee_rank(np.array([5.0])) == 1
    assert _knee_rank(np.array([10.0, 8.0, 6.0, 0.01, 0.01, 0.01])) == 3


@pytest.mark.parametrize("snr", [10.0, 3.0])
def test_knee_rank_high_snr_detects_true_rank(snr):
    rng = np.random.default_rng(2)
    n, dim, true_rank = 4000, 64, 8
    basis = rng.standard_normal((dim, true_rank))
    a = rng.standard_normal((n, true_rank)) @ basis.T * snr + rng.standard_normal((n, dim))
    ev = np.linalg.eigvalsh(np.cov(a, rowvar=False))[::-1]
    r = _knee_rank(ev, n_samples=n)
    assert abs(r - true_rank) <= 0.1 * true_rank


def test_auto_on_isotropic_corpus_keeps_full_dim(tmp_path):
    """Corpus isotropico + spectral_r='auto': sin rodilla la fase espectral no
    trunca (r = dim) y el recall con defaults es el del modo completo."""
    rng = np.random.default_rng(1)
    n, dim = 3000, 128
    U = rng.standard_normal((n, dim)).astype(np.float32)
    st = _build("auto", U, str(tmp_path))
    assert st._rdim == dim
    with open(str(tmp_path / "proj_meta.json")) as f:
        meta = json.load(f)
    assert meta["r_eff"] == dim
    assert meta["knee"]["applied"] is False
    Q = rng.standard_normal((100, dim)).astype(np.float32)
    rec = _recall(st.search_batch(Q, k=K), _gt(Q, U))
    assert rec >= 0.98, f"recall@10={rec:.4f}"


# ----------------------------------------------------------------- r2-01
@pytest.mark.parametrize("spec", [16, "auto"])
def test_spectral_recall_with_default_oversample_in_distribution(clustered_case, spec):
    """N=20k, D=384, queries in-distribution, rescore exacto, defaults."""
    X, Q, gt, stores = clustered_case
    st = stores[spec]
    assert st._rdim < D  # la fase 1 SI trunca (r explicito / rodilla real)
    rec = _recall(st.search_batch(Q, k=K), gt)
    assert (
        rec >= 0.98
    ), f"recall@10={rec:.4f} con spectral_r={spec!r} (r={st._rdim}) y oversample por defecto"


def test_full_mode_recall_default_unchanged(clustered_case):
    X, Q, gt, stores = clustered_case
    assert stores[None]._rdim == D
    assert _recall(stores[None].search_batch(Q, k=K), gt) >= 0.99


def test_explicit_oversample_still_honoured_and_validated(clustered_case):
    X, Q, gt, stores = clustered_case
    st = stores["auto"]
    assert st.search_batch(Q, k=K, oversample=1).shape == (len(Q), K)
    # el valor explicito NO se sobreescribe: con oversample 2 el recall de
    # fase 1 en r=49 sigue colapsando (conducta documentada, audit r2-01)
    assert _recall(st.search_batch(Q, k=K, oversample=2), gt) < 0.5
    for bad in (0, -1):
        with pytest.raises(ValueError):
            st.search_batch(Q, k=K, oversample=bad)


# ----------------------------------------------------------------- r2-05
@pytest.mark.parametrize("spec", [None, 16, "auto"])
@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_add_rejects_non_finite_before_writing(tmp_path, spec, bad):
    rng = np.random.default_rng(5)
    dim = 16
    st = SQ8DiskStore(dim=dim, dump_dir=str(tmp_path), spectral_r=spec)
    st.add(rng.standard_normal((50, dim)).astype(np.float32))
    st.flush()  # dump activo: el add rechazado no debe invalidarlo ni abrir spill
    before = (
        len(st),
        st._f32_lines,
        st._f32_path,
        st._float_mm is not None,
        None if st._q is None else st._q.copy(),
        None if st._scale is None else st._scale.copy(),
        None if st._stats is None else st._stats[0],
    )
    rows = rng.standard_normal((3, dim)).astype(np.float32)
    rows[1, 2] = bad
    with pytest.raises(ValueError, match="NaN or Inf"):
        st.add(rows)
    assert (len(st), st._f32_lines, st._f32_path, st._float_mm is not None) == before[:4]
    if before[4] is None:
        assert st._q is None
    else:
        np.testing.assert_array_equal(st._q, before[4])
        np.testing.assert_array_equal(st._scale, before[5])
    assert (None if st._stats is None else st._stats[0]) == before[6]
    assert np.all(np.isfinite(st._scale))
    assert np.all(np.isfinite(np.asarray(st._float_mm)))
    with pytest.raises(ValueError, match=r"row\(s\) \[1\]"):  # nombra la fila ofensiva
        st.add(rows)


@pytest.mark.parametrize("spec", [None, "auto"])
def test_build_from_float_npy_rejects_non_finite(tmp_path, spec):
    X = np.random.default_rng(6).standard_normal((500, 32)).astype(np.float32)
    X[123, 7] = np.nan
    xp = str(tmp_path / "X.npy")
    np.save(xp, X)
    dump = tmp_path / "dump"
    with pytest.raises(ValueError, match="NaN or Inf"):
        SQ8DiskStore.build_from_float_npy(xp, str(dump), chunk=100, spectral_r=spec)
    assert not (dump / "scale.npy").exists()
    assert not (dump / ".f32_spill.bin").exists()


# ----------------------------------------------------------------- r2-06
def test_search_batch_query_chunks_identical_to_single_chunk(clustered_case, monkeypatch):
    X, Q, gt, stores = clustered_case
    st = stores["auto"]
    single = st.search_batch(Q, k=K, query_chunk=len(Q))
    monkeypatch.setattr(sq8mod, "DIST_BUDGET_BYTES", 2 << 20)  # ~26 queries por chunk
    chunked = st.search_batch(Q, k=K)
    assert chunked.shape == single.shape
    np.testing.assert_array_equal(chunked, single)
    assert _recall(chunked, gt) >= 0.98


def test_search_batch_memory_bounded(clustered_case):
    """B=1000, n=20k, D=384: el pico no escala con len(Q) x n."""
    X, Q, gt, stores = clustered_case
    Qm = np.random.default_rng(7).standard_normal((1000, D)).astype(np.float32)
    tracemalloc.start()
    try:
        out = stores[None].search_batch(Qm, k=K)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    budget = 64 << 20
    # antes: d2 [1000, 20000] = 76 MiB + 2 temporales del mismo tamano + dequant ~ 260 MiB
    assert peak < 3.0 * budget, f"pico tracemalloc {peak / 2**20:.1f} MiB"
    assert out.shape == (1000, K) and (out >= 0).all()


def test_search_batch_query_chunk_validation(clustered_case):
    X, Q, gt, stores = clustered_case
    with pytest.raises(ValueError):
        stores[None].search_batch(Q, k=K, query_chunk=0)
