"""SQ8DiskStore: store vectorial comprimido para datasets que no entran en RAM.

Protocolo validado en bench/exp_h10_sq8.py (H10): cuantizacion scalar 8-bit
por fila (uint8 + scale + zero) conserva el orden relativo suficiente para
fase 1 (recall 0.97) y el rescoring float32 sobre los candidatos recupera
recall 1.0. Aqui los float32 viven en DISCO (mmap) y solo las paginas de los
candidatos se tocan al rescorear; la fase 1 corre sobre los arrays uint8 en
RAM (4x menos RAM que float32).

Diseño: modulo aislado, no toca el store float32 de SimpleVectorDB (opt-in).

H13b: el float32 del rescoring son los ORIGINALES entregados a add(), no
dequantizados desde SQ8. Un dump dequantizado deja el rescoring sin informacion
nueva: el recall queda clavado en el techo SQ8 (~0.97) sin importar el
oversample (regresion atrapada por test_flush_preserves_original_float32 y
test_recall_improves_with_oversample). El spill de originales a disco es
incremental (append binario por add, pico de RAM acotado al chunk).

Fase 1 espectral (probe E, opt-in via ``spectral_r``): en vez de cuantizar X
completo se cuantiza la proyeccion X @ P donde P son las top-r componentes
principales (r explicito o "auto" por knee del autoespectro). La RAM de fase
1 baja ~D/r (probe E: 23.5MB -> 1.4MB = 16.3x con r=16, D=384, N=60k) y el
rescoring mmap full-D queda EXACTAMENTE igual (mismos float32 originales).
``spectral_r=None`` (default) reproduce bit a bit el flujo historico.
"""
from __future__ import annotations

import json
import os
from typing import List, Optional, Union

import numpy as np

SpectralSpec = Union[None, int, str]  # None | int >= 1 | "auto"


def _knee_rank(evals_desc: np.ndarray) -> int:
    """Rank (nro de componentes) en la rodilla del autoespectro descendente.

    Metodo de la maxima distancia a la cuerda sobre la curva de varianza
    explicada acumulada (Kneedle). Determinista. Si el espectro es plano
    (sin rodilla) devuelve la dimension completa.
    """
    ev = np.clip(np.asarray(evals_desc, dtype=np.float64), 0.0, None)
    n = len(ev)
    if n < 2:
        return max(1, n)
    cum = np.cumsum(ev)
    total = cum[-1]
    if total <= 0.0:
        return n
    y = cum / total  # varianza explicada acumulada, y[-1] == 1
    xn = np.arange(n) / (n - 1)
    chord = y[0] + (y[-1] - y[0]) * xn
    d = y - chord  # > 0 en la zona convexa; max en la rodilla
    i = int(np.argmax(d))
    if d[i] <= 1e-12:  # espectro plano: no hay rodilla -> sin truncar
        return n
    return i + 1


class SQ8DiskStore:
    """Almacen de vectores SQ8 en RAM + float32 ORIGINALES en disco (mmap) para rescoring.

    Parametros espectrales (opt-in, fase 1):
      - ``spectral_r=None``: comportamiento historico exacto (cuantiza X),
        con el spill de originales H13b.
      - ``spectral_r=int >= 1``: cuantiza (X - mean) @ P con P = top-r
        componentes principales; r se recorta a ``dim`` si hace falta.
      - ``spectral_r="auto"``: r se autodetecta con el knee del autoespectro.
    El rescoring SIEMPRE es full-D sobre los float32 originales en disco.
    """

    def __init__(self, dim: int, dump_dir: Optional[str] = None,
                 spectral_r: SpectralSpec = None):
        if dim < 1:
            raise ValueError(f"dim must be >= 1 (got {dim})")
        if spectral_r is not None and spectral_r != "auto":
            if isinstance(spectral_r, bool) or not isinstance(spectral_r, (int, np.integer)) \
                    or int(spectral_r) < 1:
                raise ValueError(
                    f"spectral_r must be None, 'auto' or int >= 1 (got {spectral_r!r})"
                )
        self.dim = int(dim)
        self.dump_dir = dump_dir
        self.spectral_r: SpectralSpec = spectral_r
        self._q: Optional[np.ndarray] = None       # uint8 [n, rdim] en RAM
        self._scale: Optional[np.ndarray] = None   # float32 [n]
        self._zero: Optional[np.ndarray] = None    # float32 [n]
        self._n = 0
        self._float_path: Optional[str] = None
        self._float_mm: Optional[np.ndarray] = None  # mmap float32 [n, dim]
        # spill incremental de los float32 ORIGINALES (H13b): append binario
        # crudo (float32 little-endian, row-major). Se consolida en flush().
        self._f32_path: Optional[str] = None
        self._f32_file = None
        self._f32_lines = 0
        self._f32_mem: List[np.ndarray] = []  # solo si no hay dump_dir aun
        self._stale_path: Optional[str] = None  # dump previo con originales (merge en flush)
        # --- estado espectral (fase 1 proyectada) ---
        self._P: Optional[np.ndarray] = None       # float32 [dim, r] (top-r PCs)
        self._mean: Optional[np.ndarray] = None    # float32 [dim]
        self._rdim = int(dim)                      # dimension de los codigos
        self.eigenvalues: Optional[np.ndarray] = None  # autoespectro descendente
        self._stats: Optional[tuple] = None        # (n, sum_x, sum_xx) float64
        self._raw_chunks: list = []                # buffer transient de filas nuevas
        self._float_rows_written = 0               # filas ya persistidas en float32.npy

    # --- cuantizacion (formula historica, compartida por ambos modos) -----
    @staticmethod
    def _quantize_rows(x: np.ndarray):
        mn = x.min(axis=1)
        mx = x.max(axis=1)
        scale = (mx - mn) / 255.0
        scale = np.where(scale == 0, np.float32(1e-8), scale).astype(np.float32)
        q = np.clip(np.round((x - mn[:, None]) / scale[:, None]), 0, 255).astype(np.uint8)
        return q, scale, mn.astype(np.float32)

    def _append_codes(self, q: np.ndarray, scale: np.ndarray, zero: np.ndarray) -> None:
        if self._q is None or self._scale is None or self._zero is None:
            self._q = q
            self._scale = scale
            self._zero = zero
        else:
            self._q = np.vstack([self._q, q])
            self._scale = np.concatenate([self._scale, scale])
            self._zero = np.concatenate([self._zero, zero])
        self._n = len(self._q)

    def _spill_float(self, x: np.ndarray) -> None:
        """Append de los float32 originales: a disco si hay dump_dir, a RAM si no."""
        if self.dump_dir is None:
            self._f32_mem.append(x.copy())
            self._f32_lines += len(x)
            return
        if self._f32_file is None:
            os.makedirs(self.dump_dir, exist_ok=True)
            self._f32_path = os.path.join(self.dump_dir, ".f32_spill.bin")
            self._f32_file = open(self._f32_path, "ab")
        self._f32_file.write(np.ascontiguousarray(x, dtype=np.float32).tobytes(order="C"))
        self._f32_lines += len(x)

    def _close_spill(self) -> None:
        if self._f32_file is not None:
            self._f32_file.close()
            self._f32_file = None

    # --- ingestión -------------------------------------------------------
    def add(self, vectors: np.ndarray) -> None:
        x = np.asarray(vectors, dtype=np.float32)
        if x.ndim == 1:
            x = x[np.newaxis, :]
        if x.shape[1] != self.dim:
            raise ValueError(f"dim mismatch: expected {self.dim}, got {x.shape[1]}")
        if self.spectral_r is not None:
            self._add_spectral(x)
            return
        if self._q is not None and self._float_mm is not None:
            # dump activo + adds nuevos (H13b): invalidar para que el proximo
            # flush() lo re-materialice con TODOS los originales. El dump
            # previo queda como _stale_path: contiene originales reutilizables
            # en el merge.
            self._stale_path = self._float_path
            self._float_mm = None
            self._float_path = None
        q, scale, zero = self._quantize_rows(x)
        self._spill_float(x)
        self._append_codes(q, scale, zero)

    def _add_spectral(self, x: np.ndarray) -> None:
        if self._P is None:
            # pre-finalize: acumular estadisticas PCA (D x D, RAM acotada) y
            # bufferizar las filas crudas para cuantizarlas al primer flush().
            n0, sx, sxx = self._stats if self._stats is not None else (
                0, np.zeros(self.dim, dtype=np.float64),
                np.zeros((self.dim, self.dim), dtype=np.float64))
            x64 = x.astype(np.float64)
            sx = sx + x64.sum(axis=0)
            sxx = sxx + x64.T @ x64
            self._stats = (n0 + len(x), sx, sxx)
            self._raw_chunks.append(x.copy())
            self._n = n0 + len(x)
            return
        # post-finalize: proyectar on the fly con la P ya aprendida; las filas
        # crudas quedan pendientes para que el proximo flush() extienda el
        # float32.npy (rescoring full-D siempre contra los originales).
        xp = (x - self._mean[None, :]) @ self._P
        q, scale, zero = self._quantize_rows(xp)
        self._append_codes(q, scale, zero)
        self._raw_chunks.append(x.copy())

    def _finalize_pca(self) -> None:
        """Computa P (top-r PCs) desde las estadisticas acumuladas."""
        if self._stats is None:
            raise AssertionError("no data added")
        n, sx, sxx = self._stats
        if n < 1:
            raise AssertionError("no data added")
        mean = sx / n
        cov = sxx / n - np.outer(mean, mean)
        cov = (cov + cov.T) / 2.0  # simetria numerica
        evals, evecs = np.linalg.eigh(cov)  # ascendente
        evals_desc = evals[::-1]
        if self.spectral_r == "auto":
            r = _knee_rank(evals_desc)
        else:
            r = int(self.spectral_r)  # type: ignore[arg-type]
        r = max(1, min(r, self.dim))
        P = evecs[:, ::-1][:, :r].copy()
        # canon de signo: determinismo ante la ambiguedad +-v de eigh
        for j in range(P.shape[1]):
            col = P[:, j]
            if col[int(np.argmax(np.abs(col)))] < 0.0:
                P[:, j] = -col
        self._P = P.astype(np.float32)
        self._mean = mean.astype(np.float32)
        self._rdim = r
        self.eigenvalues = evals_desc
        self._stats = None

    # --- persistencia ----------------------------------------------------
    def flush(self, dump_dir: Optional[str] = None) -> str:
        """Escribe float32 ORIGINALES a disco + meta; desde aca el rescoring usa mmap.

        H13b: si hubo adds desde el ultimo dump, lo re-materializa con los
        originales del spill (escritura a .tmp + rename atomico). Si no hubo
        cambios y el dump existe, no lo toca.
        """
        d = dump_dir or self.dump_dir
        if not d:
            raise ValueError("dump_dir required")
        os.makedirs(d, exist_ok=True)
        if self.spectral_r is not None:
            return self._flush_spectral(d)
        q = self._q
        scale = self._scale
        zero = self._zero
        assert q is not None and scale is not None and zero is not None
        if self._f32_mem:
            # spill en RAM (store sin dump_dir): volcarlo a disco ahora
            if self._f32_file is None:
                self._f32_path = os.path.join(d, ".f32_spill.bin")
                self._f32_file = open(self._f32_path, "ab")
            for blk in self._f32_mem:
                self._f32_file.write(
                    np.ascontiguousarray(blk, dtype=np.float32).tobytes(order="C")
                )
            self._f32_mem = []
        self._close_spill()
        spill = self._f32_path if self._f32_path is not None else os.path.join(d, ".f32_spill.bin")
        need_rebuild = (
            self._float_mm is None
            or self._float_path is None
            or not os.path.exists(self._float_path)
        )
        if need_rebuild:
            self._float_path = os.path.join(d, "float32.npy")
            tmp_path = self._float_path + ".tmp"
            spill_ok = (
                self._f32_lines == self._n
                and os.path.exists(spill)
                and os.path.getsize(spill) == self._n * self.dim * 4
            )
            stale_ok = (
                self._stale_path is not None
                and os.path.exists(self._stale_path)
            )
            mm = np.lib.format.open_memmap(
                tmp_path, mode="w+", dtype=np.float32, shape=(self._n, self.dim)
            )
            chunk = 100_000
            if spill_ok:
                # spill completo y en orden: copia por chunks, pico de RAM acotado
                src = np.fromfile(spill, dtype=np.float32)
                for i in range(0, self._n, chunk):
                    mm[i : i + chunk] = src[i * self.dim : (i + chunk) * self.dim].reshape(
                        chunk if i + chunk <= self._n else self._n - i, self.dim
                    )
                mm.flush()
                del src
            elif stale_ok:
                # adds sobre un dump cargado (load()/build_from_float_npy()):
                # merge del dump previo (originales, filas [0, n_old)) con el
                # spill de los adds nuevos (filas [n_old, n)). No degrada a
                # dequantizado: el rescoring conserva los originales (H13b).
                stale = self._stale_path
                assert stale is not None
                st_mm = np.load(stale, mmap_mode="r")
                n_old = 0
                if st_mm.ndim == 2 and st_mm.shape[1] == self.dim and st_mm.shape[0] <= self._n:
                    n_old = int(st_mm.shape[0])
                    mm[:n_old] = st_mm[:]
                del st_mm
                for i in range(n_old, self._n, chunk):
                    j = min(i + chunk, self._n)
                    # el spill solo tiene las filas nuevas: offset relativo a n_old
                    src = np.fromfile(
                        spill, dtype=np.float32,
                        offset=(i - n_old) * self.dim * 4,
                        count=(j - i) * self.dim,
                    )
                    mm[i:j] = src.reshape(j - i, self.dim)
                mm.flush()
            else:
                # reconstruccion de emergencia desde SQ8 (estados sin spill
                # completo, p.ej. dumps previos a H13b): NO preserva
                # originales, el rescoring queda con techo SQ8 (ver H13b).
                for i in range(0, self._n, chunk):
                    mm[i : i + chunk] = (
                        q[i : i + chunk].astype(np.float32) * scale[i : i + chunk, None]
                        + zero[i : i + chunk, None]
                    )
                mm.flush()
            del mm
            os.replace(tmp_path, self._float_path)
            # el spill ya fue consumido (o es invalido): borrarlo para que el
            # proximo add-after-flush arranque uno fresco con SOLO las filas nuevas
            if os.path.exists(spill):
                os.remove(spill)
            self._f32_path = None
            self._f32_lines = 0
        np.save(os.path.join(d, "sq8.npy"), q)
        np.save(os.path.join(d, "scale.npy"), scale)
        np.save(os.path.join(d, "zero.npy"), zero)
        self._float_mm = np.load(self._float_path, mmap_mode="r")
        return d

    def _persist_spectral(self, d: str) -> None:
        assert self._q is not None and self._scale is not None and self._zero is not None
        np.save(os.path.join(d, "sq8.npy"), self._q)
        np.save(os.path.join(d, "scale.npy"), self._scale)
        np.save(os.path.join(d, "zero.npy"), self._zero)
        assert self._P is not None and self._mean is not None
        np.save(os.path.join(d, "proj_P.npy"), self._P)
        np.save(os.path.join(d, "proj_mean.npy"), self._mean)
        meta = {
            "spectral_r": self.spectral_r if not isinstance(self.spectral_r, np.integer)
            else int(self.spectral_r),
            "r_eff": int(self._rdim),
            "dim": int(self.dim),
            "n": int(self._n),
            "eigenvalues_top": [float(v) for v in np.maximum(
                self.eigenvalues[: min(64, len(self.eigenvalues))], 0.0)]
            if self.eigenvalues is not None else [],
        }
        with open(os.path.join(d, "proj_meta.json"), "w") as f:
            json.dump(meta, f)

    def _flush_spectral(self, d: str) -> str:
        if self._P is None:
            # primer flush: aprender P y cuantizar la proyeccion de TODO el
            # corpus crudo bufferizado; float32.npy = originales (rescore exacto).
            self._finalize_pca()
            raw = np.vstack(self._raw_chunks) if self._raw_chunks else None
            assert raw is not None and self._P is not None and self._mean is not None
            q, scale, zero = self._quantize_rows(
                (raw - self._mean[None, :]) @ self._P)
            self._q, self._scale, self._zero = q, scale, zero
            self._n = len(q)
            self._float_path = os.path.join(d, "float32.npy")
            mm = np.lib.format.open_memmap(
                self._float_path, mode="w+", dtype=np.float32, shape=(self._n, self.dim)
            )
            chunk = 100_000
            for i in range(0, self._n, chunk):
                mm[i : i + chunk] = raw[i : i + chunk]
            mm.flush()
            del mm
            self._raw_chunks = []
            self._float_rows_written = self._n
        elif self._raw_chunks:
            # flush posterior con adds online: consolidar float32.npy en el
            # dump_dir con las filas crudas pendientes (los codigos ya fueron
            # proyectados en add). Nunca se muta el .npy fuente externo.
            new = np.vstack(self._raw_chunks)
            old_n = self._float_rows_written
            assert self._float_path is not None
            src = self._float_mm
            # escribir a un temp y reemplazar: src y target NO pueden ser el
            # mismo archivo (el truncado del open_memmap clobberia el mmap).
            target = os.path.join(d, "float32.npy")
            tmp_path = target + ".new"
            mm = np.lib.format.open_memmap(
                tmp_path, mode="w+", dtype=np.float32, shape=(self._n, self.dim)
            )
            chunk = 100_000
            for i in range(0, old_n, chunk):
                j = min(i + chunk, old_n)
                mm[i:j] = src[i:j]
            mm[old_n : self._n] = new[: self._n - old_n]
            mm.flush()
            del mm, src
            os.replace(tmp_path, target)
            self._float_path = target
            self._raw_chunks = []
            self._float_rows_written = self._n
        self._persist_spectral(d)
        assert self._float_path is not None
        self._float_mm = np.load(self._float_path, mmap_mode="r")
        return d

    @classmethod
    def build_from_float_npy(
        cls, float_npy_path: str, dump_dir: str, chunk: int = 100_000,
        spectral_r: SpectralSpec = None,
    ) -> "SQ8DiskStore":
        """Cuantiza por chunks desde un .npy en disco (mmap) y deja el store
        listo: SQ8 en RAM, rescoring contra el float32 ORIGINAL (sin
        duplicarlo). Pico de RAM acotado al chunk (tambien en modo espectral:
        pass 1 acumula el covariante D x D, pass 2 cuantiza X @ P por chunks).

        Nota H13b: adds posteriores invalidan el dump y el proximo flush()
        lo re-materializa desde el spill de originales del propio store (el
        archivo float_npy_path NO se modifica ni se relee: si se movio, el
        flush re-construye desde el spill interno completo).
        """
        mm = np.load(float_npy_path, mmap_mode="r")
        st = cls(dim=int(mm.shape[1]), dump_dir=dump_dir, spectral_r=spectral_r)
        os.makedirs(dump_dir, exist_ok=True)
        if spectral_r is None:
            for i in range(0, len(mm), chunk):
                st.add(np.asarray(mm[i : i + chunk]))
            q = st._q
            scale = st._scale
            zero = st._zero
            assert q is not None and scale is not None and zero is not None
            np.save(os.path.join(dump_dir, "sq8.npy"), q)
            np.save(os.path.join(dump_dir, "scale.npy"), scale)
            np.save(os.path.join(dump_dir, "zero.npy"), zero)
            fp = str(float_npy_path)
            st._float_path = fp
            st._float_mm = np.load(fp, mmap_mode="r")
            # el spill del build se descarta: el float32 vivo es el .npy
            # fuente (H13b); un add posterior abre un spill fresco.
            st._close_spill()
            if st._f32_path is not None and os.path.exists(st._f32_path):
                os.remove(st._f32_path)
            st._f32_path = None
            st._f32_lines = 0
            return st
        # modo espectral: pass 1 stats, eigh (D x D), pass 2 cuantiza X @ P
        sx = np.zeros(st.dim, dtype=np.float64)
        sxx = np.zeros((st.dim, st.dim), dtype=np.float64)
        ntot = 0
        for i in range(0, len(mm), chunk):
            x64 = np.asarray(mm[i : i + chunk]).astype(np.float64)
            sx = sx + x64.sum(axis=0)
            sxx = sxx + x64.T @ x64
            ntot += len(x64)
        st._stats = (ntot, sx, sxx)
        st._finalize_pca()
        assert st._P is not None and st._mean is not None
        q_parts, scale_parts, zero_parts = [], [], []
        for i in range(0, len(mm), chunk):
            xp = (np.asarray(mm[i : i + chunk]) - st._mean[None, :]) @ st._P
            q, s, z = st._quantize_rows(xp)
            q_parts.append(q)
            scale_parts.append(s)
            zero_parts.append(z)
        st._q = np.vstack(q_parts)
        st._scale = np.concatenate(scale_parts)
        st._zero = np.concatenate(zero_parts)
        st._n = len(st._q)
        st._float_rows_written = st._n  # el float original ya esta en disco
        st._persist_spectral(dump_dir)
        st._float_path = str(float_npy_path)
        st._float_mm = np.load(st._float_path, mmap_mode="r")
        return st

    @classmethod
    def load(cls, dump_dir: str) -> "SQ8DiskStore":
        st = cls(dim=1, dump_dir=dump_dir)
        q = np.load(os.path.join(dump_dir, "sq8.npy"))
        assert q is not None
        st._q = q
        st._scale = np.load(os.path.join(dump_dir, "scale.npy"))
        st._zero = np.load(os.path.join(dump_dir, "zero.npy"))
        st._n = len(q)
        p_path = os.path.join(dump_dir, "proj_P.npy")
        if os.path.exists(p_path):
            st._P = np.load(p_path)
            st._mean = np.load(os.path.join(dump_dir, "proj_mean.npy"))
            assert st._P is not None and st._mean is not None
            st._rdim = int(st._P.shape[1])
            st.dim = int(st._P.shape[0])
            st.spectral_r = st._rdim
            st._float_rows_written = st._n
            meta_path = os.path.join(dump_dir, "proj_meta.json")
            if os.path.exists(meta_path):
                with open(meta_path) as f:
                    meta = json.load(f)
                st.spectral_r = meta.get("spectral_r", st._rdim)
                if isinstance(st.spectral_r, int):
                    st.spectral_r = int(st.spectral_r)
        else:
            st.dim = int(q.shape[1])
            st._rdim = st.dim
        st._float_path = os.path.join(dump_dir, "float32.npy")
        st._float_mm = np.load(st._float_path, mmap_mode="r")
        return st

    # --- busqueda --------------------------------------------------------
    def search_batch(
        self,
        queries: np.ndarray,
        k: int = 10,
        oversample: int = 2,
    ) -> np.ndarray:
        """Fase 1 sobre SQ8 en RAM + rescoring float32 via mmap (solo candidatos).

        En modo espectral las queries (full-D) se proyectan con la misma P de
        indexacion antes de la fase 1; el rescoring de la fase 2 es full-D
        sobre los float32 originales (identico al modo historico).
        """
        Q = np.asarray(queries, dtype=np.float32)
        if Q.ndim == 1:
            Q = Q[np.newaxis, :]
        if Q.shape[1] != self.dim:
            raise ValueError(f"dim mismatch: expected {self.dim}, got {Q.shape[1]}")
        if k < 1:
            raise ValueError(f"k must be >= 1 (got {k})")
        assert self._q is not None and self._scale is not None and self._zero is not None
        if self._P is not None:
            assert self._mean is not None
            # fase 1 en el subespacio r-dim; la fase 2 rescorea full-D con la
            # Q ORIGINAL (mmap float32 = vectores originales sin proyectar).
            Qp = (Q - self._mean[None, :]) @ self._P
        else:
            Qp = Q
        n = self._n
        k1 = min(k * oversample, n)

        # fase 1: dequantize por chunks (linea H10), d2 aproximada
        qq = np.einsum("ij,ij->i", Qp, Qp)
        d2 = np.empty((len(Q), n), dtype=np.float32)
        chunk = 100_000
        for i in range(0, n, chunk):
            deq = (
                self._q[i : i + chunk].astype(np.float32) * self._scale[i : i + chunk, None]
                + self._zero[i : i + chunk, None]
            )
            d2[:, i : i + chunk] = (
                qq[:, None] - 2.0 * (Qp @ deq.T) + np.einsum("ij,ij->i", deq, deq)[None, :]
            )
        np.maximum(d2, 0.0, out=d2)

        # fase 2: rescoring float32 via mmap gather (solo k1 candidatos/fila)
        # d2 completa: ||x||^2 + ||q||^2 - 2<x,q> (valida SIN normalizar)
        assert self._float_mm is not None, "flush()/load() required before search"
        qq0 = np.einsum("ij,ij->i", Q, Q)  # norma de la query ORIGINAL (full-D)
        out = np.full((len(Q), k), -1, dtype=np.int64)
        for b in range(len(Q)):
            part = np.argpartition(d2[b], k1 - 1)[:k1]
            cand = np.sort(part)  # gather ordenado = lectura de paginas secuencial
            x = np.asarray(self._float_mm[cand])
            exact = (
                np.einsum("ij,ij->i", x, x)
                - 2.0 * (x @ Q[b])
                + qq0[b]
            )
            best = cand[np.argsort(exact)[:k]]  # menor distancia primero
            out[b, : len(best)] = best
        return out

    def __len__(self) -> int:
        return self._n

    def ram_bytes(self) -> int:
        """Bytes de arrays en RAM (uint8 + scales), sin contar el mmap ni el spill."""
        total = 0
        q = self._q
        scale = self._scale
        zero = self._zero
        if q is not None and scale is not None and zero is not None:
            total += q.nbytes + scale.nbytes + zero.nbytes
        return total
