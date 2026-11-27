from __future__ import annotations

import logging
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .config import M2MConfig

logger = logging.getLogger(__name__)

# audit r3a-09: el UserWarning de "backend pedido no disponible" se emite una
# sola vez por backend y por proceso; el logger.warning sale en cada construcción.
_WARNED_FALLBACKS: set = set()


def _reset_backend_warning_guard() -> None:
    """Vuelve a permitir el UserWarning de fallback (para tests)."""
    _WARNED_FALLBACKS.clear()


def _cuda_unavailable_reason() -> Optional[str]:
    """None si torch + CUDA son usables; si no, el motivo (para logs y backend_info)."""
    try:
        import torch
    except ImportError:
        return "torch no está instalado (pip install torch)"
    try:
        if not torch.cuda.is_available():
            return "torch.cuda.is_available() es False (sin GPU/driver NVIDIA)"
    except Exception as e:  # pragma: no cover - instalación de torch rota
        return f"torch.cuda.is_available() falló: {e}"
    return None


def _warn_backend_fallback(requested: str, effective: str, reason: str) -> None:
    logger.warning(
        "Backend %r pedido pero NO disponible: %s. Corriendo en %r.",
        requested,
        reason,
        effective,
    )
    if requested in _WARNED_FALLBACKS:
        return
    _WARNED_FALLBACKS.add(requested)
    warnings.warn(
        f"device={requested!r} was requested but is unavailable: {reason}. "
        f"Running on {effective!r} instead.",
        UserWarning,
        stacklevel=3,
    )


class M2MEngine:
    """
    Hardware-accelerated engine for M2M workloads: the backend SELECTOR.

    Es el único lugar donde se decide qué backend corre: CUDA si
    ``config.enable_cuda`` y torch+CUDA están disponibles; si no, Vulkan si
    ``config.enable_vulkan`` y su índice llega a inicializarse; si no, CPU.
    El resultado se consulta en :attr:`backend` y :meth:`get_backend_info`
    (audit r3a-09): ``device='cuda'`` sin torch NO corre en silencio, emite un
    ``UserWarning`` (una vez) + ``logger.warning`` con el motivo.

    ``self.device`` refleja ``config.compute_device`` (lo pedido); el backend
    EFECTIVO es ``self.backend``.

    No confundir con ``m2m.M2MEngine`` (``m2m/__init__.py``): ese es el
    envoltorio de ``M2MMemory`` que instancia ``SimpleVectorDB.engine``; esta
    clase se crea como ``HRM2Engine.router``. No hay soporte multi-GPU: un
    proceso usa un solo backend en un solo dispositivo.

    Soporta backends: CUDA, Vulkan, CPU.
    Auto-detect basado en config.device y disponibilidad.
    """

    def __init__(self, config: Optional[M2MConfig] = None) -> None:
        self.config = config
        self.device = getattr(config, "compute_device", "cpu") if config else "cpu"

        self.use_cuda = False
        self.use_vulkan = False
        self.gpu_router = None

        # audit r3a-09: estado real del selector.
        #   requested: backend que pide la config ('cpu' si no pidió GPU)
        #   reasons:   motivo por el que cada backend pedido NO se usa
        reasons: Dict[str, str] = {}
        want_cuda = bool(config and getattr(config, "enable_cuda", False))
        want_vulkan = bool(config and getattr(config, "enable_vulkan", False))
        requested = "cuda" if want_cuda else ("vulkan" if want_vulkan else "cpu")

        if config:
            # 1. Try CUDA first
            if want_cuda:
                reason = _cuda_unavailable_reason()
                if reason is None:
                    try:
                        from .gpu_vector_index import CUDAVectorIndex

                        dim = getattr(config, "latent_dim", 640)
                        dummy = np.zeros((1, dim), dtype="float32")
                        self.gpu_router = CUDAVectorIndex(dummy, max_batch_size=1)
                        self.use_cuda = True
                        logger.info("Initialized CUDA GPU Router.")
                    except Exception as e:
                        reasons["cuda"] = f"CUDAVectorIndex falló al inicializar: {e}"
                        logger.warning(
                            "CUDA init failed: %s. %s",
                            e,
                            "Trying Vulkan." if want_vulkan else "Falling back to CPU.",
                        )
                else:
                    reasons["cuda"] = reason
                    logger.warning(
                        "CUDA unavailable: %s. %s",
                        reason,
                        "Trying Vulkan." if want_vulkan else "Falling back to CPU.",
                    )

            # 2. Try Vulkan
            if not self.use_cuda and want_vulkan:
                try:
                    from .gpu_vector_index import GPUVectorIndex

                    dim = getattr(config, "latent_dim", 640)
                    dummy = np.zeros((1, dim), dtype="float32")
                    self.gpu_router = GPUVectorIndex(dummy, max_batch_size=1)
                    self.use_vulkan = True
                    logger.info("Initialized Vulkan GPU Router.")
                except Exception as e:
                    reasons["vulkan"] = f"GPUVectorIndex falló al inicializar: {e}"
                    logger.warning("Vulkan init failed: %s. Falling back to CPU.", e)

        self.compute_device = "cuda" if self.use_cuda else ("vulkan" if self.use_vulkan else "cpu")
        # Backend EFECTIVO (audit r3a-09): alias explícito y consultable de compute_device.
        self.backend = self.compute_device
        self.backend_info: Dict[str, Any] = {
            "requested": requested,
            "effective": self.backend,
            "reasons": reasons,
        }
        if self.backend != requested:
            _warn_backend_fallback(
                requested, self.backend, reasons.get(requested, "no se pudo inicializar")
            )

    def get_backend_info(self) -> Dict[str, Any]:
        """Backend pedido, backend EFECTIVO y el motivo de cada caída (copia)."""
        info = dict(self.backend_info)
        info["reasons"] = dict(info["reasons"])
        return info

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
