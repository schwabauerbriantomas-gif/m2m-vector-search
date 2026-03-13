"""
M2M API Integrada con todas las características avanzadas.

Integra:
- GPU Auto-Tuning
- Query Optimization (Cache + Prefetch)
- Auto-Scaling
- Distributed Cluster Mode
"""

import time
from typing import Any, Dict, List, Optional

import numpy as np

from . import AdvancedVectorDB, M2MConfig, SimpleVectorDB
from .auto_scaling import AutoScaler, HorizontalScaler, NodeMetrics
from .gpu_auto_tune import GPUAutoTuner, get_gpu_tuner
from .query_optimizer import QueryOptimizer


class M2MOptimized:
    """
    M2M Vector Database con todas las optimizaciones habilitadas.

    Features:
    - GPU acceleration con auto-tuning
    - Query caching con prefetching
    - Auto-scaling horizontal
    - Métricas en tiempo real

    Uso:
        db = M2MOptimized(
            latent_dim=768,
            enable_gpu=True,
            enable_cache=True,
            enable_autoscale=True
        )

        # Añadir documentos
        db.add(ids, vectors, metadata)

        # Búsqueda optimizada (cached + GPU si disponible)
        results = db.search(query, k=10)

        # Ver métricas
        stats = db.get_optimization_stats()
    """

    def __init__(
        self,
        latent_dim: int = 640,
        device: Optional[str] = None,
        enable_gpu: bool = True,
        enable_cache: bool = True,
        cache_entries: int = 1000,
        cache_memory_mb: int = 100,
        enable_autoscale: bool = False,
        min_nodes: int = 1,
        max_nodes: int = 10,
    ):
        """
        Args:
            latent_dim: Dimensión de vectores
            device: Dispositivo ('cpu', 'cuda', 'vulkan', None=auto)
            enable_gpu: Habilitar GPU acceleration
            enable_cache: Habilitar query cache
            cache_entries: Máximo entradas en cache
            cache_memory_mb: Máxima memoria para cache
            enable_autoscale: Habilitar auto-scaling
            min_nodes: Mínimo nodos en cluster
            max_nodes: Máximo nodos en cluster
        """
        # Detectar GPU y configurar
        self.enable_gpu = enable_gpu
        self.gpu_config = {}

        if enable_gpu:
            tuner = get_gpu_tuner()
            profile = tuner.detect_gpu()

            if profile:
                self.gpu_config = tuner.get_optimal_config()
                device = device or "vulkan"
                print(f"[M2M] GPU detected: {profile.device_name}")
                print(f"[M2M] VRAM: {profile.vram_mb}MB")
                print(f"[M2M] Optimal batch size: {profile.optimal_batch_size}")
            else:
                print("[M2M] No GPU detected, using CPU")
                device = "cpu"

        # Crear base de datos
        self.db = AdvancedVectorDB(
            device=device, latent_dim=latent_dim, enable_soc=True, enable_energy_features=True
        )

        # Query optimizer
        self.enable_cache = enable_cache
        self.optimizer = None

        if enable_cache:
            self.optimizer = QueryOptimizer(
                cache_entries=cache_entries, cache_memory_mb=cache_memory_mb, enable_prefetch=True
            )

        # Auto-scaler
        self.enable_autoscale = enable_autoscale
        self.scaler = None
        self.cluster = None

        if enable_autoscale:
            self.scaler = AutoScaler(min_nodes=min_nodes, max_nodes=max_nodes)

            self.cluster = HorizontalScaler(
                node_template={"cpu": 4, "memory": "8GB", "latent_dim": latent_dim}
            )

            # Registrar callbacks
            self.scaler.register_callbacks(
                scale_up=self.cluster.scale_up, scale_down=self.cluster.scale_down
            )

            # Nodo inicial
            self.cluster.scale_up()

        # Métricas
        self._metrics = {
            "total_queries": 0,
            "total_adds": 0,
            "gpu_queries": 0,
            "cache_hits": 0,
            "cache_misses": 0,
        }

    def add(
        self,
        ids: List[str],
        vectors: np.ndarray,
        metadata: Optional[List[Dict]] = None,
        documents: Optional[List[str]] = None,
    ) -> int:
        """
        Añade documentos con optimizaciones.

        Args:
            ids: Lista de IDs
            vectors: Vectores [N, D]
            metadata: Metadata opcional
            documents: Textos originales opcionales

        Returns:
            Número de documentos añadidos
        """
        self._metrics["total_adds"] += 1

        # Usar batch size óptimo si GPU disponible
        if self.enable_gpu and self.gpu_config:
            batch_size = self.gpu_config.get("batch_size", 100)

            # Procesar en batches para mejor rendimiento
            n = len(ids)
            total_added = 0

            for i in range(0, n, batch_size):
                batch_ids = ids[i : i + batch_size]
                batch_vectors = vectors[i : i + batch_size]
                batch_meta = metadata[i : i + batch_size] if metadata else None
                batch_docs = documents[i : i + batch_size] if documents else None

                n_added = self.db.add(
                    ids=batch_ids, vectors=batch_vectors, metadata=batch_meta, documents=batch_docs
                )
                total_added += n_added

            return total_added
        else:
            # Sin GPU, añadir normalmente
            return self.db.add(ids=ids, vectors=vectors, metadata=metadata, documents=documents)

    def search(
        self, query: np.ndarray, k: int = 10, filter: Optional[Dict] = None, use_cache: bool = True
    ) -> Any:
        """
        Búsqueda optimizada con cache y GPU.

        Args:
            query: Vector de consulta [D]
            k: Número de resultados
            filter: Filtros de metadata
            use_cache: Usar cache (default True)

        Returns:
            Resultados de búsqueda
        """
        self._metrics["total_queries"] += 1

        # Función de búsqueda real
        def search_fn(q, k_val):
            return self.db.search(q, k=k_val, filter=filter, include_metadata=True)

        # Usar optimizer si está habilitado
        if self.enable_cache and use_cache and self.optimizer:
            result = self.optimizer.execute_with_cache(query, k, search_fn, filter)

            # Actualizar métricas
            stats = self.optimizer.get_stats()
            self._metrics["cache_hits"] = stats["cache"]["hits"]
            self._metrics["cache_misses"] = stats["cache"]["misses"]

            return result
        else:
            # Búsqueda directa
            return search_fn(query, k)

    def search_with_energy(self, query: np.ndarray, k: int = 10) -> Any:
        """
        Búsqueda con información energética.

        Args:
            query: Vector de consulta
            k: Número de resultados

        Returns:
            SearchResult con energías
        """
        self._metrics["total_queries"] += 1
        return self.db.search_with_energy(query, k=k)

    def update(self, id: str, **kwargs) -> Any:
        """Actualiza un documento."""
        return self.db.update(id, **kwargs)

    def delete(self, id: str, **kwargs) -> Any:
        """Elimina un documento."""
        return self.db.delete(id, **kwargs)

    def suggest_exploration(self, n: int = 3) -> List:
        """Sugerencias de exploración."""
        return self.db.suggest_exploration(n=n)

    def consolidate(self, threshold: Optional[float] = None) -> int:
        """Consolida memoria usando SOC."""
        return self.db.consolidate(threshold=threshold)

    def update_cluster_metrics(
        self, cpu_percent: float, memory_percent: float, qps: float, latency_ms: float
    ):
        """
        Actualiza métricas del cluster para auto-scaling.

        Args:
            cpu_percent: Uso de CPU %
            memory_percent: Uso de memoria %
            qps: Queries por segundo
            latency_ms: Latencia en ms
        """
        if not self.enable_autoscale or not self.scaler:
            return

        metrics = NodeMetrics(
            node_id=self.cluster.get_active_nodes()[0] if self.cluster else "node1",
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            qps=qps,
            memory_percent=memory_percent,
            latency_ms=latency_ms,
            active_queries=0,
            uptime_seconds=time.time(),
        )

        self.scaler.update_metrics(metrics)

        # Evaluar escalado
        decision = self.scaler.evaluate_scaling()
        if decision:
            self.scaler.execute_scaling(decision)

    def get_optimization_stats(self) -> Dict[str, Any]:
        """
        Retorna estadísticas completas de optimización.

        Returns:
            Diccionario con métricas de GPU, cache y scaling
        """
        stats = {
            "database": self.db.get_stats(),
            "optimization": self._metrics.copy(),
            "gpu": self.gpu_config.copy() if self.gpu_config else {},
        }

        if self.enable_cache and self.optimizer:
            stats["cache"] = self.optimizer.get_stats()

        if self.enable_autoscale and self.scaler:
            stats["autoscale"] = self.scaler.get_stats()

        return stats

    def clear_cache(self):
        """Limpia el cache de queries."""
        if self.optimizer:
            self.optimizer.clear()

    def get_prefetch_suggestions(self, n: int = 3) -> List[str]:
        """Obtiene sugerencias de prefetch."""
        if self.optimizer and self.optimizer.enable_prefetch:
            return self.optimizer.get_prefetch_suggestions(n)
        return []


# Factory function
def create_optimized_m2m(config: Optional[Dict] = None) -> M2MOptimized:
    """
    Factory para crear M2M optimizado.

    Args:
        config: Configuración opcional

    Returns:
        M2MOptimized instance
    """
    config = config or {}

    return M2MOptimized(
        latent_dim=config.get("latent_dim", 640),
        device=config.get("device"),
        enable_gpu=config.get("enable_gpu", True),
        enable_cache=config.get("enable_cache", True),
        cache_entries=config.get("cache_entries", 1000),
        cache_memory_mb=config.get("cache_memory_mb", 100),
        enable_autoscale=config.get("enable_autoscale", False),
        min_nodes=config.get("min_nodes", 1),
        max_nodes=config.get("max_nodes", 10),
    )
