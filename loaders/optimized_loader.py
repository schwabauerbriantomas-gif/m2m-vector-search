"""Loader para datasets transformados."""

import numpy as np
import struct
from dataclasses import dataclass
from typing import List


@dataclass
class LoadedSplat:
    mu: np.ndarray
    alpha: float
    kappa: float
    n_vectors: int
    indices: np.ndarray


def load_m2m_dataset(path: str) -> List[LoadedSplat]:
    """Carga dataset en formato M2M optimizado."""
    with open(path, 'rb') as f:
        n_splats, dim, n_orig, n_levels = struct.unpack('IIII', f.read(16))
        
        splats = []
        for _ in range(n_splats):
            mu = np.frombuffer(f.read(dim * 4), dtype=np.float32).copy()
            alpha, kappa, n_vecs = struct.unpack('ffI', f.read(12))
            indices = np.frombuffer(f.read(n_vecs * 4), dtype=np.int32).copy()
            
            splats.append(LoadedSplat(mu, alpha, kappa, n_vecs, indices))
        
        print(f"✅ Cargados {n_splats} splats ({n_orig} vectores originales)")
        return splats
