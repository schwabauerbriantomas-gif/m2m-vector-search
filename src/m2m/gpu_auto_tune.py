"""
GPU Auto-Tuning y Optimización para M2M.

Características:
- Auto-detección de hardware GPU
- Benchmarking automático de rendimiento
- Configuración óptima de workgroups
- Memory pool management
- Dynamic batching
"""

import os
import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np


@dataclass
class GPUProfile:
    """Perfil de GPU detectado."""

    vendor: str
    device_name: str
    vram_mb: int
    compute_units: int
    max_workgroup_size: int
    optimal_batch_size: int
    memory_bandwidth_gbps: float
    supports_fp16: bool
    supports_subgroups: bool


class GPUAutoTuner:
    """
    Auto-tuning para GPU acceleration.

    Detecta hardware, benchmarks rendimiento y configura
    parámetros óptimos automáticamente.
    """

    def __init__(self):
        self.profile: Optional[GPUProfile] = None
        self.benchmarks: Dict[str, float] = {}

    def detect_gpu(self) -> Optional[GPUProfile]:
        """
        Detecta GPU disponible y sus características.

        Returns:
            GPUProfile o None si no hay GPU disponible
        """
        try:
            import vulkan as vk

            # Crear instancia Vulkan
            app_info = vk.VkApplicationInfo(
                sType=vk.VK_STRUCTURE_TYPE_APPLICATION_INFO,
                pApplicationName="M2M GPU Tuner",
                applicationVersion=1,
                pEngineName="M2M",
                engineVersion=1,
                apiVersion=vk.VK_API_VERSION_1_2,
            )

            create_info = vk.VkInstanceCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, pApplicationInfo=app_info
            )

            instance = vk.vkCreateInstance(create_info, None)

            # Enumerar dispositivos físicos
            devices = vk.vkEnumeratePhysicalDevices(instance)

            if not devices:
                vk.vkDestroyInstance(instance, None)
                return None

            # Usar primer dispositivo
            physical_device = devices[0]
            props = vk.vkGetPhysicalDeviceProperties(physical_device)
            memory_props = vk.vkGetPhysicalDeviceMemoryProperties(physical_device)

            # Calcular VRAM total
            vram_mb = 0
            for i in range(memory_props.memoryHeapCount):
                if memory_props.memoryHeaps[i].flags & vk.VK_MEMORY_HEAP_DEVICE_LOCAL_BIT:
                    vram_mb += memory_props.memoryHeaps[i].size // (1024 * 1024)

            # Detectar vendor
            vendor_map = {0x1002: "AMD", 0x10DE: "NVIDIA", 0x8086: "Intel", 0x13B5: "ARM"}
            vendor = vendor_map.get(props.vendorID, "Unknown")

            # Detectar características
            features = vk.vkGetPhysicalDeviceFeatures(physical_device)

            # Obtener límites de compute
            limits = props.limits

            self.profile = GPUProfile(
                vendor=vendor,
                device_name=props.deviceName,
                vram_mb=vram_mb,
                compute_units=16,  # Estimación
                max_workgroup_size=limits.maxComputeWorkGroupSize[0],
                optimal_batch_size=self._estimate_optimal_batch(vram_mb),
                memory_bandwidth_gbps=100.0,  # Estimación
                supports_fp16=bool(features.shaderFloat16),
                supports_subgroups=True,  # Vulkan 1.1+
            )

            vk.vkDestroyInstance(instance, None)

            return self.profile

        except ImportError:
            print("[WARNING] Vulkan SDK no disponible")
            return None
        except Exception as e:
            print(f"[ERROR] GPU detection failed: {e}")
            return None

    def _estimate_optimal_batch(self, vram_mb: int) -> int:
        """
        Estima batch size óptimo basado en VRAM disponible.

        Args:
            vram_mb: VRAM en MB

        Returns:
            Batch size óptimo
        """
        # Reservar 50% para índice, usar 50% para queries
        usable_mb = vram_mb * 0.5

        # Asumiendo 640D × 4 bytes = 2.56 KB por vector
        # Result buffer: 100 vectors × 4 bytes = 0.4 KB
        # Total por query: ~3 KB

        batch_size = int(usable_mb * 1024 / 3)  # KB / 3KB

        # Clampear a rangos razonables
        return max(10, min(1000, batch_size))

    def benchmark_throughput(
        self, index_vectors: np.ndarray, n_queries: int = 1000
    ) -> Dict[str, float]:
        """
        Benchmark de throughput de GPU.

        Args:
            index_vectors: Vectores de índice [N, D]
            n_queries: Número de queries para benchmark

        Returns:
            Métricas de rendimiento
        """
        if self.profile is None:
            return {}

        try:
            from .gpu_vector_index import GPUVectorIndex

            # Crear índice GPU
            gpu_index = GPUVectorIndex(
                index_vectors, max_batch_size=self.profile.optimal_batch_size
            )

            # Generar queries aleatorias
            queries = np.random.randn(n_queries, index_vectors.shape[1]).astype(np.float32)

            # Warmup
            _ = gpu_index.batch_search(queries[:10], k=10)

            # Benchmark
            start = time.time()

            batch_size = self.profile.optimal_batch_size
            for i in range(0, n_queries, batch_size):
                batch = queries[i : i + batch_size]
                _ = gpu_index.batch_search(batch, k=10)

            elapsed = time.time() - start

            qps = n_queries / elapsed
            latency_ms = (elapsed / n_queries) * 1000

            self.benchmarks = {
                "qps": qps,
                "latency_ms": latency_ms,
                "batch_size": batch_size,
                "n_vectors": len(index_vectors),
                "n_queries": n_queries,
            }

            return self.benchmarks

        except Exception as e:
            print(f"[ERROR] GPU benchmark failed: {e}")
            return {}

    def get_optimal_config(self) -> Dict:
        """
        Retorna configuración óptima basada en el perfil detectado.

        Returns:
            Diccionario con parámetros óptimos
        """
        if self.profile is None:
            return {"batch_size": 100, "workgroup_size": 256, "enable_vulkan": False, "fp16": False}

        return {
            "batch_size": self.profile.optimal_batch_size,
            "workgroup_size": min(256, self.profile.max_workgroup_size),
            "enable_vulkan": True,
            "fp16": self.profile.supports_fp16,
            "vram_mb": self.profile.vram_mb,
            "vendor": self.profile.vendor,
            "device": self.profile.device_name,
        }


class GPUMemoryPool:
    """
    Pool de memoria GPU para reutilización de buffers.

    Evita allocation/deallocation frecuente manteniendo
    buffers pre-asignados.
    """

    def __init__(self, max_buffers: int = 10):
        self.max_buffers = max_buffers
        self._pool: Dict[str, list] = {
            "small": [],  # < 1 MB
            "medium": [],  # 1-10 MB
            "large": [],  # > 10 MB
        }

    def get_buffer(self, size_bytes: int) -> Optional[Tuple]:
        """
        Obtiene buffer del pool o crea uno nuevo.

        Args:
            size_bytes: Tamaño requerido en bytes

        Returns:
            (buffer, memory) tuple o None
        """
        # Categorizar tamaño
        mb = size_bytes / (1024 * 1024)
        if mb < 1:
            category = "small"
        elif mb < 10:
            category = "medium"
        else:
            category = "large"

        # Buscar en pool
        if self._pool[category]:
            return self._pool[category].pop()

        # Crear nuevo buffer (implementación simplificada)
        return None

    def return_buffer(self, buffer_mem: Tuple, size_bytes: int):
        """
        Retorna buffer al pool para reutilización.

        Args:
            buffer_mem: (buffer, memory) tuple
            size_bytes: Tamaño del buffer
        """
        mb = size_bytes / (1024 * 1024)
        if mb < 1:
            category = "small"
        elif mb < 10:
            category = "medium"
        else:
            category = "large"

        if len(self._pool[category]) < self.max_buffers:
            self._pool[category].append(buffer_mem)

    def clear(self):
        """Limpia todo el pool."""
        for category in self._pool:
            self._pool[category].clear()


# Instancia global del auto-tuner
_auto_tuner: Optional[GPUAutoTuner] = None
_memory_pool: Optional[GPUMemoryPool] = None


def get_gpu_tuner() -> GPUAutoTuner:
    """Obtiene instancia global del auto-tuner."""
    global _auto_tuner
    if _auto_tuner is None:
        _auto_tuner = GPUAutoTuner()
    return _auto_tuner


def get_memory_pool() -> GPUMemoryPool:
    """Obtiene instancia global del memory pool."""
    global _memory_pool
    if _memory_pool is None:
        _memory_pool = GPUMemoryPool()
    return _memory_pool
