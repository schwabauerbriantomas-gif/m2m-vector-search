from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import logging
logger = logging.getLogger(__name__)



class M2MEngine:
    """
    Hardware-accelerated engine for M2M workloads.

    Soporta backends: CUDA, Vulkan, CPU.
    Auto-detect basado en config.device y disponibilidad.
    """

    def __init__(self, config=None):
        self.config = config
        self.device = getattr(config, "compute_device", "cpu") if config else "cpu"

        self.use_cuda = False
        self.use_vulkan = False
        self.gpu_router = None

        if config:
            # 1. Try CUDA first
            if getattr(config, "enable_cuda", False):
                try:
                    from .gpu_vector_index import CUDAVectorIndex, _has_cuda

                    if _has_cuda():
                        dim = getattr(config, "latent_dim", 640)
                        dummy = np.zeros((1, dim), dtype="float32")
                        self.gpu_router = CUDAVectorIndex(dummy, max_batch_size=1)
                        self.use_cuda = True
                        logger.info("Initialized CUDA GPU Router.")
                except Exception as e:
                    logger.warning("CUDA init failed: %s. Trying Vulkan.", e)

            # 2. Try Vulkan
            if not self.use_cuda and getattr(config, "enable_vulkan", False):
                try:
                    from .gpu_vector_index import GPUVectorIndex

                    dim = getattr(config, "latent_dim", 640)
                    dummy = np.zeros((1, dim), dtype="float32")
                    self.gpu_router = GPUVectorIndex(dummy, max_batch_size=1)
                    self.use_vulkan = True
                    logger.info("Initialized Vulkan GPU Router.")
                except Exception as e:
                    logger.warning("Vulkan init failed: %s. Falling back to CPU.", e)

        self.compute_device = "cuda" if self.use_cuda else ("vulkan" if self.use_vulkan else "cpu")

    def compute_expert_distances(
        self,
        query: np.ndarray,
        expert_embeddings: np.ndarray,
        expert_indices: np.ndarray,
        coarse_ids: np.ndarray,
        fine_ids: np.ndarray,
    ) -> List[Tuple[int, float, int, int]]:
        if len(expert_embeddings) == 0:
            return []

        # GPU path (CUDA or Vulkan)
        if self.gpu_router is not None:
            distances_cpu = self.gpu_router.compute_distances(query, expert_embeddings)
        else:
            # CPU fallback
            q_arr = np.array(query, dtype=np.float32)[np.newaxis, :]
            e_arr = np.array(expert_embeddings, dtype=np.float32)
            distances_cpu = np.linalg.norm(e_arr - q_arr, axis=1)

        results: List[Tuple[int, float, int, int]] = []
        for i in range(len(expert_indices)):
            results.append(
                (
                    int(expert_indices[i]),
                    float(distances_cpu[i]),
                    int(coarse_ids[i]),
                    int(fine_ids[i]),
                )
            )

        return results
