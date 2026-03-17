"""Optimized dataset loader for M2M binary format.

Reads splats saved by M2MDatasetTransformer.save_for_m2m().

Binary layout:
  Header: 4 uint32 (n_splats, dim, n_vectors, hierarchy_levels)
  Per splat:
    mu: dim * float32 (NOT float64 — the transformer stores float32 centroids)
    alpha: float32
    kappa: float32
    n_vectors: uint32
    indices: n_vectors * int32
  Metadata: separate JSON file at <path>_meta.json
"""

import struct
import os
import json
import numpy as np
from typing import Dict, Any


def load_m2m_dataset(path: str) -> Dict[str, Any]:
    """Load an M2M binary dataset produced by save_for_m2m().

    Args:
        path: Path to the .bin file.

    Returns:
        Dict with keys:
            - "splats": list of dicts {mu, alpha, kappa, indices, n_vectors}
            - "metadata": dict from the companion JSON file (if any)
    """
    with open(path, "rb") as f:
        # --- Header ---
        header = f.read(struct.calcsize("IIII"))
        if len(header) < struct.calcsize("IIII"):
            raise ValueError(f"File too small for header: {len(header)} bytes")

        n_splats, dim, n_vectors_total, hierarchy_levels = struct.unpack(
            "IIII", header
        )

        # mu is stored as float32 (the transformer uses MiniBatchKMeans which
        # produces float32 centroids)
        mu_bytes_per_splat = dim * np.float32().itemsize  # dim * 4
        afi_bytes = struct.calcsize("ffI")  # 12 bytes

        # --- Splats ---
        splats = []
        for _ in range(n_splats):
            # Read mu (float32 array of shape (dim,))
            mu_raw = f.read(mu_bytes_per_splat)
            if len(mu_raw) < mu_bytes_per_splat:
                raise ValueError(
                    f"Unexpected EOF reading mu at splat {len(splats)}: "
                    f"expected {mu_bytes_per_splat} bytes, got {len(mu_raw)}"
                )
            mu = np.frombuffer(mu_raw, dtype=np.float32).copy().astype(np.float64)

            # Read alpha (float32), kappa (float32), n_vectors (uint32)
            afi_raw = f.read(afi_bytes)
            if len(afi_raw) < afi_bytes:
                raise ValueError(
                    f"Unexpected EOF reading afi at splat {len(splats)}"
                )
            alpha, kappa, n_vec = struct.unpack("ffI", afi_raw)

            # Read indices (int32 array of shape (n_vec,))
            indices = np.array([], dtype=np.int32)
            if n_vec > 0:
                idx_bytes = int(n_vec) * np.int32().itemsize
                idx_raw = f.read(idx_bytes)
                if len(idx_raw) < idx_bytes:
                    raise ValueError(
                        f"Unexpected EOF reading indices at splat {len(splats)}"
                    )
                indices = np.frombuffer(idx_raw, dtype=np.int32).copy()

            splats.append(
                {
                    "mu": mu,
                    "alpha": float(alpha),
                    "kappa": float(kappa),
                    "n_vectors": int(n_vec),
                    "indices": indices,
                }
            )

    # --- Companion metadata JSON (if it exists) ---
    metadata: Dict[str, Any] = {}
    meta_path = path.replace(".bin", "_meta.json")
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as mf:
                metadata = json.load(mf)
        except (json.JSONDecodeError, UnicodeDecodeError, ValueError):
            pass

    return {"splats": splats, "metadata": metadata}
