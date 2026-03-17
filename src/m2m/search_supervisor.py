"""
Search Supervisor — Patrón Supervisor para orquestación de búsqueda multi-backend.

Coordina múltiples backends (CPU, CUDA, Vulkan) y decide dinámicamente
cuál usar basándose en tipo de consulta, tamaño del dataset, hardware
disponible y latencia requerida. Fallback automático si un backend falla.

Basado en MASFactory's SupervisorAgent pattern.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np


class BackendType(str, Enum):
    """Tipos de backend disponibles."""
    CPU = "cpu"
    CUDA = "cuda"
    VULKAN = "vulkan"


class QueryComplexity(str, Enum):
    """Niveles de complejidad de consulta."""
    SIMPLE = "simple"          # k pequeño, dataset pequeño
    MODERATE = "moderate"      # k medio, dataset medio
    COMPLEX = "complex"        # k grande, dataset grande, o batch


@dataclass
class BackendInfo:
    """Información sobre un backend registrado."""
    name: BackendType
    available: bool = False
    search_fn: Optional[Callable] = None
    index_fn: Optional[Callable] = None
    avg_latency_ms: float = 0.0
    total_queries: int = 0
    total_errors: int = 0
    max_dimension: int = 0
    memory_usage_mb: float = 0.0


@dataclass
class SupervisorDecision:
    """Decisión del supervisor sobre qué backend usar."""
    backend: BackendType
    reason: str
    fallback_chain: List[BackendType] = field(default_factory=list)


@dataclass
class SupervisorStats:
    """Estadísticas del supervisor."""
    total_decisions: int = 0
    decisions_by_backend: Dict[str, int] = field(default_factory=lambda: {"cpu": 0, "cuda": 0, "vulkan": 0})
    fallbacks_triggered: int = 0
    total_errors: int = 0
    avg_decision_time_ms: float = 0.0


class SearchSupervisor:
    """
    Supervisor de búsqueda multi-backend.

    Orquesta backends CPU/CUDA/Vulkan usando el patrón Supervisor de MASFactory.
    Toma decisiones basadas en complejidad de query, hardware disponible y
    latencia requerida. Incluye fallback automático.

    Example:
        >>> supervisor = SearchSupervisor()
        >>> supervisor.register_backend(BackendType.CPU, search_fn=my_cpu_search)
        >>> result = supervisor.search(query_vector, k=10)
    """

    # Umbrales de complejidad
    _SIMPLE_K_THRESHOLD = 20
    _SIMPLE_DATASET_THRESHOLD = 10_000
    _MODERATE_K_THRESHOLD = 100
    _MODERATE_DATASET_THRESHOLD = 100_000

    def __init__(
        self,
        default_backend: BackendType = BackendType.CPU,
        latency_budget_ms: Optional[float] = None,
        enable_auto_fallback: bool = True,
    ):
        """
        Args:
            default_backend: Backend por defecto si no se puede decidir.
            latency_budget_ms: Latencia máxima aceptable (None = sin límite).
            enable_auto_fallback: Activar fallback automático a otro backend.
        """
        self.default_backend = default_backend
        self.latency_budget_ms = latency_budget_ms
        self.enable_auto_fallback = enable_auto_fallback

        self._backends: Dict[BackendType, BackendInfo] = {
            bt: BackendInfo(name=bt) for bt in BackendType
        }
        self._stats = SupervisorStats()
        self._latency_history: Dict[BackendType, List[float]] = {
            bt: [] for bt in BackendType
        }

    def register_backend(
        self,
        backend_type: BackendType,
        search_fn: Callable[[np.ndarray, int], Any],
        index_fn: Optional[Callable] = None,
        max_dimension: int = 0,
    ) -> None:
        """
        Registra un backend de búsqueda.

        Args:
            backend_type: Tipo de backend (CPU, CUDA, Vulkan).
            search_fn: Función de búsqueda, firma: (query, k) -> results.
            index_fn: Función de indexación opcional.
            max_dimension: Dimensión máxima soportada (0 = sin límite).
        """
        info = self._backends[backend_type]
        info.search_fn = search_fn
        info.index_fn = index_fn
        info.available = True
        info.max_dimension = max_dimension

    def unregister_backend(self, backend_type: BackendType) -> None:
        """Desregistra un backend."""
        info = self._backends[backend_type]
        info.search_fn = None
        info.index_fn = None
        info.available = False

    def classify_complexity(
        self,
        k: int,
        dataset_size: int,
        batch_size: int = 1,
        query_dim: int = 0,
    ) -> QueryComplexity:
        """
        Clasifica la complejidad de una consulta.

        Args:
            k: Número de vecinos solicitados.
            dataset_size: Tamaño del dataset indexado.
            batch_size: Número de queries en batch.
            query_dim: Dimensión de la query.

        Returns:
            Nivel de complejidad.
        """
        # Batch queries son siempre complejas
        if batch_size > 1:
            return QueryComplexity.COMPLEX

        if k <= self._SIMPLE_K_THRESHOLD and dataset_size <= self._SIMPLE_DATASET_THRESHOLD:
            return QueryComplexity.SIMPLE
        elif k <= self._MODERATE_K_THRESHOLD and dataset_size <= self._MODERATE_DATASET_THRESHOLD:
            return QueryComplexity.MODERATE
        return QueryComplexity.COMPLEX

    def decide_backend(
        self,
        k: int = 10,
        dataset_size: int = 0,
        batch_size: int = 1,
        query_dim: int = 0,
    ) -> SupervisorDecision:
        """
        Decide qué backend usar para una consulta.

        Args:
            k: Número de vecinos.
            dataset_size: Tamaño del dataset.
            batch_size: Tamaño del batch.
            query_dim: Dimensión de la query.

        Returns:
            SupervisorDecision con backend seleccionado y fallback chain.
        """
        start = time.time()
        complexity = self.classify_complexity(k, dataset_size, batch_size, query_dim)

        # Construir lista de backends disponibles ordenados por preferencia
        available = [
            bt for bt, info in self._backends.items()
            if info.available and info.search_fn is not None
        ]

        if not available:
            self._stats.total_errors += 1
            raise RuntimeError("No hay backends disponibles")

        # Estrategia de selección basada en complejidad
        preference_order = self._get_preference_order(complexity, available, query_dim)

        # Verificar latencia budget
        if self.latency_budget_ms is not None:
            preference_order = self._filter_by_latency(preference_order)

        # Seleccionar el mejor
        primary = preference_order[0] if preference_order else available[0]
        fallback_chain = preference_order[1:] if len(preference_order) > 1 else []

        self._stats.total_decisions += 1
        self._stats.decisions_by_backend[primary.value] = (
            self._stats.decisions_by_backend.get(primary.value, 0) + 1
        )
        self._stats.avg_decision_time_ms = (
            self._stats.avg_decision_time_ms * (self._stats.total_decisions - 1)
            + (time.time() - start) * 1000
        ) / self._stats.total_decisions

        return SupervisorDecision(
            backend=primary,
            reason=f"complexity={complexity.value}, available={len(available)}",
            fallback_chain=fallback_chain,
        )

    def _get_preference_order(
        self,
        complexity: QueryComplexity,
        available: List[BackendType],
        query_dim: int,
    ) -> List[BackendType]:
        """
        Determina el orden de preferencia de backends según complejidad.

        SIMPLE → CPU (rápido, sin overhead de GPU)
        MODERATE → Vulkan > CUDA > CPU
        COMPLEX → CUDA > Vulkan > CPU
        """
        # Filtrar por dimensión si hay límite
        valid = [
            bt for bt in available
            if self._backends[bt].max_dimension == 0 or query_dim <= self._backends[bt].max_dimension
        ]

        if complexity == QueryComplexity.SIMPLE:
            order = [BackendType.CPU, BackendType.VULKAN, BackendType.CUDA]
        elif complexity == QueryComplexity.MODERATE:
            order = [BackendType.VULKAN, BackendType.CUDA, BackendType.CPU]
        else:
            order = [BackendType.CUDA, BackendType.VULKAN, BackendType.CPU]

        return [bt for bt in order if bt in valid]

    def _filter_by_latency(self, order: List[BackendType]) -> List[BackendType]:
        """Filtra backends que cumplen el budget de latencia."""
        if self.latency_budget_ms is None:
            return order

        filtered = []
        for bt in order:
            history = self._latency_history.get(bt, [])
            if not history:
                filtered.append(bt)  # Sin datos, dar oportunidad
                continue
            avg = sum(history) / len(history)
            if avg <= self.latency_budget_ms:
                filtered.append(bt)

        return filtered if filtered else order[:1]

    def search(
        self,
        query: np.ndarray,
        k: int = 10,
        dataset_size: int = 0,
        **kwargs,
    ) -> Any:
        """
        Ejecuta búsqueda usando el backend seleccionado por el supervisor.

        Incluye fallback automático si el backend primario falla.

        Args:
            query: Vector de consulta.
            k: Número de resultados.
            dataset_size: Tamaño del dataset (para decisión).
            **kwargs: Argumentos adicionales para la función de búsqueda.

        Returns:
            Resultados de búsqueda del backend exitoso.

        Raises:
            RuntimeError: Si todos los backends fallan.
        """
        query_dim = query.shape[-1] if hasattr(query, 'shape') else 0
        decision = self.decide_backend(k=k, dataset_size=dataset_size, query_dim=query_dim)

        # Intentar backends en orden (primario + fallback chain)
        backends_to_try = [decision.backend] + decision.fallback_chain

        for backend_type in backends_to_try:
            info = self._backends[backend_type]
            if not info.available or info.search_fn is None:
                continue

            start = time.time()
            try:
                result = info.search_fn(query, k, **kwargs)
                elapsed = (time.time() - start) * 1000

                # Actualizar métricas
                info.total_queries += 1
                info.avg_latency_ms = (
                    info.avg_latency_ms * (info.total_queries - 1) + elapsed
                ) / info.total_queries
                self._latency_history[backend_type].append(elapsed)
                # Mantener solo las últimas 100 latencias
                if len(self._latency_history[backend_type]) > 100:
                    self._latency_history[backend_type] = self._latency_history[backend_type][-100:]

                return result

            except Exception as e:
                info.total_errors += 1
                if self.enable_auto_fallback:
                    self._stats.fallbacks_triggered += 1
                    import logging
                    logging.getLogger("m2m.supervisor").warning(
                        f"Backend {backend_type.value} falló: {e}. Intentando fallback..."
                    )
                    continue
                raise

        self._stats.total_errors += 1
        raise RuntimeError("Todos los backends fallaron")

    def search_batch(
        self,
        queries: np.ndarray,
        k: int = 10,
        dataset_size: int = 0,
        **kwargs,
    ) -> List[Any]:
        """
        Ejecuta búsqueda batch.

        Args:
            queries: Array de queries [B, D].
            k: Número de resultados por query.
            dataset_size: Tamaño del dataset.
            **kwargs: Argumentos adicionales.

        Returns:
            Lista de resultados, uno por query.
        """
        results = []
        for i in range(len(queries)):
            result = self.search(queries[i], k=k, dataset_size=dataset_size, **kwargs)
            results.append(result)
        return results

    def health_check(self) -> Dict[str, Dict[str, Any]]:
        """
        Verifica el estado de todos los backends.

        Returns:
            Dict con estado de cada backend.
        """
        status = {}
        for bt, info in self._backends.items():
            status[bt.value] = {
                "available": info.available,
                "total_queries": info.total_queries,
                "total_errors": info.total_errors,
                "avg_latency_ms": round(info.avg_latency_ms, 2),
                "error_rate": (
                    info.total_errors / info.total_queries if info.total_queries > 0 else 0.0
                ),
            }
        return status

    def get_stats(self) -> SupervisorStats:
        """Retorna estadísticas del supervisor."""
        return self._stats
