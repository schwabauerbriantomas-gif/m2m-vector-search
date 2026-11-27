"""P17 — `SQ8DiskStore.load()` sobre un dump creado con `build_from_float_npy()`.

Antes: el camino por chunks deja los float32 originales mmapeados DESDE el .npy
fuente y no escribe `float32.npy` en el dump; `load(dump)` hacía `np.load` de ese
archivo sin comprobar y levantaba `FileNotFoundError` críptico.
Cierra el hallazgo [audit P17-2].
"""

import numpy as np
import pytest

from m2m.sq8_disk import SQ8DiskStore


def _data(n, dim, seed=5):
    rng = np.random.RandomState(seed)
    return (rng.randn(n, dim) * 3.0).astype(np.float32)


@pytest.mark.parametrize("spectral_r", [None, 4])
def test_load_dump_de_build_from_float_npy(tmp_path, spectral_r):
    dim, n = 24, 3000
    data = _data(n, dim)
    src = tmp_path / "fuenteoriginal.npy"
    np.save(src, data)
    dump = tmp_path / "dump"
    st = SQ8DiskStore.build_from_float_npy(str(src), str(dump), chunk=700, spectral_r=spectral_r)
    # el camino por chunks no duplica los originales dentro del dump
    assert not (dump / "float32.npy").exists()
    q = data[:20] + 0.01
    esperado = st.search_batch(q, k=10)
    st2 = SQ8DiskStore.load(str(dump))
    assert len(st2) == n
    assert np.array_equal(st2.search_batch(q, k=10), esperado)


def test_load_fuente_borrada_error_claro(tmp_path):
    dim, n = 16, 500
    src = tmp_path / "fuente.npy"
    np.save(src, _data(n, dim))
    dump = tmp_path / "dump"
    SQ8DiskStore.build_from_float_npy(str(src), str(dump))
    src.unlink()
    with pytest.raises(FileNotFoundError) as ei:
        SQ8DiskStore.load(str(dump))
    msg = str(ei.value)
    assert "float_path" in msg, msg  # nombra el remedio
    assert str(src) in msg, msg  # nombra el archivo faltante


def test_load_float_path_override(tmp_path):
    dim, n = 16, 500
    data = _data(n, dim)
    src = tmp_path / "fuente.npy"
    np.save(src, data)
    dump = tmp_path / "dump"
    st = SQ8DiskStore.build_from_float_npy(str(src), str(dump))
    moved = tmp_path / "movida.npy"
    np.save(moved, data)
    src.unlink()
    st2 = SQ8DiskStore.load(str(dump), float_path=str(moved))
    assert len(st2) == n
    q = data[:7]
    assert np.array_equal(st2.search_batch(q, k=5), st.search_batch(q, k=5))
    with pytest.raises(FileNotFoundError, match="float_path"):
        SQ8DiskStore.load(str(dump), float_path=str(tmp_path / "inexistente.npy"))


def test_build_from_float_npy_add_flush_load(tmp_path):
    """add()+flush() materializa float32.npy en el dump: load() lo usa y el
    registro de la fuente (que quedó viejo) no interfiere."""
    dim, n = 16, 500
    src = tmp_path / "fuente.npy"
    np.save(src, _data(n, dim))
    dump = tmp_path / "dump"
    st = SQ8DiskStore.build_from_float_npy(str(src), str(dump))
    extra = _data(50, dim, seed=9)
    st.add(extra)
    st.flush()
    assert (dump / "float32.npy").exists()
    st2 = SQ8DiskStore.load(str(dump))
    assert len(st2) == n + 50
    q = extra[:7]
    assert np.array_equal(st2.search_batch(q, k=5), st.search_batch(q, k=5))
