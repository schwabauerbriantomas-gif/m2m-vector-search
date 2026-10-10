"""SQ8DiskStore: store vectorial comprimido para datasets que no entran en RAM.

Protocolo validado en bench/exp_h10_sq8.py (H10): cuantizacion scalar 8-bit
por fila (uint8 + scale + zero) conserva el orden relativo suficiente para
fase 1 (recall 0.97) y el rescoring float32 sobre los candidatos recupera
recall 1.0. Aqui los float32 viven en DISCO (mmap) y solo las paginas de los
candidatos se tocan al rescorear; la fase 1 corre sobre los arrays uint8 en
RAM (4x menos RAM que float32).

Diseño: modulo aislado, no toca el store float32 de SimpleVectorDB (opt-in).
"""
from __future__ import annotations

import os
from typing import Optional

import numpy as np


class SQ8DiskStore:
    """Almacen de vectores SQ8 en RAM + float32 en disco (mmap) para rescoring."""

    def __init__(self, dim: int, dump_dir: Optional[str] = None):
        if dim < 1:
            raise ValueError(f"dim must be >= 1 (got {dim})")
        self.dim = int(dim)
        self.dump_dir = dump_dir
        self._q: Optional[np.ndarray] = None       # uint8 [n, dim] en RAM
        self._scale: Optional[np.ndarray] = None   # float32 [n]
        self._zero: Optional[np.ndarray] = None    # float32 [n]
        self._n = 0
        self._float_path: Optional[str] = None
        self._float_mm: Optional[np.ndarray] = None  # mmap float32 [n, dim]

    # --- ingestión -------------------------------------------------------
    def add(self, vectors: np.ndarray) -> None:
        x = np.asarray(vectors, dtype=np.float32)
        if x.ndim == 1:
            x = x[np.newaxis, :]
        if x.shape[1] != self.dim:
            raise ValueError(f"dim mismatch: expected {self.dim}, got {x.shape[1]}")
        mn = x.min(axis=1)
        mx = x.max(axis=1)
        scale = (mx - mn) / 255.0
        scale = np.where(scale == 0, np.float32(1e-8), scale).astype(np.float32)
        q = np.clip(np.round((x - mn[:, None]) / scale[:, None]), 0, 255).astype(np.uint8)
        if self._q is None or self._scale is None or self._zero is None:
            self._q = q
            self._scale = scale
            self._zero = mn.astype(np.float32)
        else:
            self._q = np.vstack([self._q, q])
            self._scale = np.concatenate([self._scale, scale])
            self._zero = np.concatenate([self._zero, mn.astype(np.float32)])
        self._n = len(self._q)

    # --- persistencia ----------------------------------------------------
    def flush(self, dump_dir: Optional[str] = None) -> str:
        """Escribe float32 a disco + meta; desde aca el rescoring usa mmap."""
        d = dump_dir or self.dump_dir
        if not d:
            raise ValueError("dump_dir required")
        os.makedirs(d, exist_ok=True)
        q = self._q
        scale = self._scale
        zero = self._zero
        assert q is not None and scale is not None and zero is not None
        if self._float_mm is not None or self._float_path is None:
            # todavia no tenemos un dump: materializar float32 POR CHUNKS via
            # memmap (pico de RAM acotado al chunk, no a n x dim)
            self._float_path = os.path.join(d, "float32.npy")
            mm = np.lib.format.open_memmap(
                self._float_path, mode="w+", dtype=np.float32, shape=(self._n, self.dim)
            )
            chunk = 100_000
            for i in range(0, self._n, chunk):
                mm[i : i + chunk] = (
                    q[i : i + chunk].astype(np.float32) * scale[i : i + chunk, None]
                    + zero[i : i + chunk, None]
                )
            mm.flush()
            del mm
        np.save(os.path.join(d, "sq8.npy"), q)
        np.save(os.path.join(d, "scale.npy"), scale)
        np.save(os.path.join(d, "zero.npy"), zero)
        self._float_mm = np.load(self._float_path, mmap_mode="r")
        return d

    @classmethod
    def build_from_float_npy(
        cls, float_npy_path: str, dump_dir: str, chunk: int = 100_000
    ) -> "SQ8DiskStore":
        """Cuantiza por chunks desde un .npy en disco (mmap) y deja el store
        listo: SQ8 en RAM, rescoring contra el float32 ORIGINAL (sin
        duplicarlo). Pico de RAM acotado al chunk."""
        mm = np.load(float_npy_path, mmap_mode="r")
        st = cls(dim=int(mm.shape[1]), dump_dir=dump_dir)
        for i in range(0, len(mm), chunk):
            st.add(np.asarray(mm[i : i + chunk]))
        os.makedirs(dump_dir, exist_ok=True)
        q = st._q
        scale = st._scale
        zero = st._zero
        assert q is not None and scale is not None and zero is not None
        np.save(os.path.join(dump_dir, "sq8.npy"), q)
        np.save(os.path.join(dump_dir, "scale.npy"), scale)
        np.save(os.path.join(dump_dir, "zero.npy"), zero)
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
        st.dim = int(q.shape[1])
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
        """Fase 1 sobre SQ8 en RAM + rescoring float32 via mmap (solo candidatos)."""
        Q = np.asarray(queries, dtype=np.float32)
        if Q.ndim == 1:
            Q = Q[np.newaxis, :]
        if Q.shape[1] != self.dim:
            raise ValueError(f"dim mismatch: expected {self.dim}, got {Q.shape[1]}")
        if k < 1:
            raise ValueError(f"k must be >= 1 (got {k})")
        assert self._q is not None and self._scale is not None and self._zero is not None
        n = self._n
        k1 = min(k * oversample, n)

        # fase 1: dequantize por chunks (linea H10), d2 aproximada
        qq = np.einsum("ij,ij->i", Q, Q)
        d2 = np.empty((len(Q), n), dtype=np.float32)
        chunk = 100_000
        for i in range(0, n, chunk):
            deq = (
                self._q[i : i + chunk].astype(np.float32) * self._scale[i : i + chunk, None]
                + self._zero[i : i + chunk, None]
            )
            d2[:, i : i + chunk] = (
                qq[:, None] - 2.0 * (Q @ deq.T) + np.einsum("ij,ij->i", deq, deq)[None, :]
            )
        np.maximum(d2, 0.0, out=d2)

        # fase 2: rescoring float32 via mmap gather (solo k1 candidatos/fila)
        # d2 completa: ||x||^2 + ||q||^2 - 2<x,q> (valida SIN normalizar)
        assert self._float_mm is not None, "flush()/load() required before search"
        out = np.full((len(Q), k), -1, dtype=np.int64)
        for b in range(len(Q)):
            part = np.argpartition(d2[b], k1 - 1)[:k1]
            cand = np.sort(part)  # gather ordenado = lectura de paginas secuencial
            x = np.asarray(self._float_mm[cand])
            exact = (
                np.einsum("ij,ij->i", x, x)
                - 2.0 * (x @ Q[b])
                + qq[b]
            )
            best = cand[np.argsort(exact)[:k]]  # menor distancia primero
            out[b, : len(best)] = best
        return out

    def __len__(self) -> int:
        return self._n

    def ram_bytes(self) -> int:
        """Bytes de arrays en RAM (uint8 + scales), sin contar el mmap."""
        total = 0
        q = self._q
        scale = self._scale
        zero = self._zero
        if q is not None and scale is not None and zero is not None:
            total += q.nbytes + scale.nbytes + zero.nbytes
        return total
