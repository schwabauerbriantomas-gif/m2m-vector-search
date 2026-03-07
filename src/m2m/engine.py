import numpy as np
from typing import List, Tuple


class M2MEngine:
    """
    Hardware-accelerated engine for M2M workloads.
    Handles parallel compute tasks like MoE router distances using available accelerators (CUDA/Vulkan).
    """

    def __init__(self, config=None):
        self.config = config
        self.device = (
            config.compute_device if hasattr(config, "compute_device") else "cpu"
        )

        self.use_vulkan = False
        self.vulkan_router = None

        if config and hasattr(config, "enable_vulkan") and config.enable_vulkan:
            try:
                from gpu_vector_index import GPUVectorIndex

                dim = config.latent_dim if hasattr(config, "latent_dim") else 640
                # Use a small dummy index (1 vector); compute_distances() uploads
                # dynamic expert sets per call — the index buffer is reused.
                dummy = __import__("numpy").zeros((1, dim), dtype="float32")
                self.vulkan_router = GPUVectorIndex(dummy, max_batch_size=1)
                self.use_vulkan = True
                print("[INFO] Initialized True Vulkan Compute Shader MoE Router.")
            except Exception as e:
                print(
                    f"[WARNING] Vulkan MoE initialization failed: {e}. Falling back to NumPy proxy."
                )
                self.compute_device = self.device
        else:
            self.compute_device = self.device

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

        if self.use_vulkan:
            distances_cpu = self.vulkan_router.compute_distances(
                query, expert_embeddings
            )
        else:
            q_arr = np.array(query, dtype=np.float32)[np.newaxis, :]
            e_arr = np.array(expert_embeddings, dtype=np.float32)
            distances = np.linalg.norm(e_arr - q_arr, axis=1)
            distances_cpu = distances

        # Zip results
        results = []
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
