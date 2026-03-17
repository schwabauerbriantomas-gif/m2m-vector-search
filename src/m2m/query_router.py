"""
Query Router — Patrón Router para clasificación y enrutamiento de queries.

Clasifica queries y las dirige a la estrategia de búsqueda más apropiada:
- Exact search → k-NN directo (brute force)
- Approximate search → HRM2 clustering
- Range search → brute force filtrado
- Batch search → paralelo

Incluye auto-learning: registra latencias y ajusta rutas dinámicamente.

Basado en MASFactory's RouterAgent pattern.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np


class SearchStrategy(str, Enum):
    """Estrategias de búsqueda disponibles."""
    EXACT = "exact"                  # k-NN brute force
    APPROXIMATE_HRM2 = "hrm2"       # HRM2 hierarchical clustering
    RANGE = "range"                  # Brute force con filtro de distancia
    BATCH_PARALLEL = "batch"         # Batch paralelo
    LSH = "lsh"                      # Locality-Sensitive Hashing


@dataclass
class QueryProfile:
    """Perfil de una query para clasificación."""
    k: int = 10
    batch_size: int = 1
    query_dim: int = 0
    has_range_filter: bool = False
    range_radius: float = 0.0
    dataset_size: int = 0
    priority: int = 0  # 0=normal, 1=low-latency, 2=high-recall


@dataclass
class RouteDecision:
    """Decisión de enrutamiento."""
    strategy: SearchStrategy
    confidence: float
    reason: str
    estimated_latency_ms: float = 0.0


@dataclass
class RouterStats:
    """Estadísticas del router."""
    total_routes: int = 0
    routes_by_strategy: Dict[str, int] = field(default_factory=dict)
    avg_latency_by_strategy: Dict[str, float] = field(default_factory=dict)
    total_latency_samples: Dict[str, int] = field(default_factory=dict)
    auto_adjustments: int = 0


class QueryRouter:
    """
    Router de queries para M2M.

    Clasifica queries y selecciona la estrategia de búsqueda óptima.
    Aprende de latencias pasadas para ajustar rutas dinámicamente.

    Example:
        >>> router = QueryRouter()
        >>> router.register_strategy(SearchStrategy.EXACT, exact_search_fn)
        >>> decision = router.route(QueryProfile(k=10, dataset_size=1000))
    """

    # Umbrales de decisión
    _HRM2_MIN_DATASET = 1000       # Mínimo dataset para HRM2
    _HRM2_MIN_K = 5                # Mínimo k para HRM2
    _BATCH_MIN_SIZE = 4            # Mínimo batch para paralelo
    _EXACT_MAX_DATASET = 5000      # Máximo dataset para exact razonable

    def __init__(
        self,
        default_strategy: SearchStrategy = SearchStrategy.EXACT,
        enable_auto_learning: bool = True,
        learning_window: int = 100,
    ):
        """
        Args:
            default_strategy: Estrategia por defecto.
            enable_auto_learning: Activar aprendizaje automático de latencias.
            learning_window: Tamaño de ventana para promediar latencias.
        """
        self.default_strategy = default_strategy
        self.enable_auto_learning = enable_auto_learning
        self.learning_window = learning_window

        self._strategies: Dict[SearchStrategy, Callable] = {}
        self._stats = RouterStats()
        self._latency_history: Dict[SearchStrategy, List[float]] = {
            s: [] for s in SearchStrategy
        }
        self._route_history: List[Tuple[QueryProfile, RouteDecision]] = []

    def register_strategy(
        self,
        strategy: SearchStrategy,
        search_fn: Callable,
    ) -> None:
        """
        Registra una estrategia de búsqueda.

        Args:
            strategy: Tipo de estrategia.
            search_fn: Función de búsqueda.
        """
        self._strategies[strategy] = search_fn

    def unregister_strategy(self, strategy: SearchStrategy) -> None:
        """Desregistra una estrategia."""
        self._strategies.pop(strategy, None)

    def classify(self, profile: QueryProfile) -> RouteDecision:
        """
        Clasifica una query y selecciona la mejor estrategia.

        Args:
            profile: Perfil de la query.

        Returns:
            RouteDecision con estrategia seleccionada.
        """
        # 1. Range search tiene prioridad si hay filtro de rango
        if profile.has_range_filter:
            return RouteDecision(
                strategy=SearchStrategy.RANGE,
                confidence=0.95,
                reason="Range filter activo",
                estimated_latency_ms=self._estimate_latency(SearchStrategy.RANGE, profile),
            )

        # 2. Batch search para queries múltiples
        if profile.batch_size >= self._BATCH_MIN_SIZE:
            # Para batch grande, HRM2 es mejor; para batch chico, parallel exact
            if profile.dataset_size >= self._HRM2_MIN_DATASET:
                strategy = SearchStrategy.APPROXIMATE_HRM2
                reason = f"Batch de {profile.batch_size} con dataset grande"
            else:
                strategy = SearchStrategy.BATCH_PARALLEL
                reason = f"Batch de {profile.batch_size} queries"
            return RouteDecision(
                strategy=strategy,
                confidence=0.85,
                reason=reason,
                estimated_latency_ms=self._estimate_latency(strategy, profile),
            )

        # 3. Ajustar por auto-learning si disponible
        adjusted_strategy = self._auto_adjust(profile)

        # 4. Decisión basada en k y dataset size
        if adjusted_strategy is not None:
            return adjusted_strategy

        if profile.dataset_size < self._HRM2_MIN_DATASET or profile.k < self._HRM2_MIN_K:
            # Dataset chico o k pequeño → exact
            strategy = SearchStrategy.EXACT
            reason = f"Dataset={profile.dataset_size}, k={profile.k} → exact"
            confidence = 0.9
        elif profile.dataset_size <= self._EXACT_MAX_DATASET and profile.k <= 20:
            # Dataset medio y k pequeño → exact sigue siendo bueno
            strategy = SearchStrategy.EXACT
            reason = f"Dataset={profile.dataset_size}, k={profile.k} → exact viable"
            confidence = 0.75
        else:
            # Dataset grande → HRM2
            strategy = SearchStrategy.APPROXIMATE_HRM2
            reason = f"Dataset={profile.dataset_size}, k={profile.k} → HRM2"
            confidence = 0.85

        return RouteDecision(
            strategy=strategy,
            confidence=confidence,
            reason=reason,
            estimated_latency_ms=self._estimate_latency(strategy, profile),
        )

    def route(self, profile: QueryProfile) -> RouteDecision:
        """
        Alias de classify para compatibilidad con patrón Router.

        Returns:
            RouteDecision.
        """
        return self.classify(profile)

    def execute(
        self,
        profile: QueryProfile,
        query: np.ndarray,
        **kwargs,
    ) -> Any:
        """
        Clasifica y ejecuta la búsqueda.

        Args:
            profile: Perfil de la query.
            query: Vector de consulta.
            **kwargs: Argumentos adicionales.

        Returns:
            Resultados de búsqueda.

        Raises:
            RuntimeError: Si la estrategia seleccionada no está registrada.
        """
        decision = self.classify(profile)
        fn = self._strategies.get(decision.strategy)

        if fn is None:
            # Fallback a estrategia por defecto
            fn = self._strategies.get(self.default_strategy)
            if fn is None:
                raise RuntimeError(
                    f"Estrategia '{decision.strategy.value}' no registrada y "
                    f"no hay default disponible"
                )
            decision.strategy = self.default_strategy

        start = time.time()
        try:
            result = fn(query, profile.k, **kwargs)
        except Exception:
            # Fallback a exact si falla
            exact_fn = self._strategies.get(SearchStrategy.EXACT)
            if exact_fn is not None and decision.strategy != SearchStrategy.EXACT:
                result = exact_fn(query, profile.k, **kwargs)
                decision.strategy = SearchStrategy.EXACT
            else:
                raise

        elapsed = (time.time() - start) * 1000

        # Registrar latencia para auto-learning
        self._record_latency(decision.strategy, elapsed)

        # Actualizar stats
        self._stats.total_routes += 1
        self._stats.routes_by_strategy[decision.strategy.value] = (
            self._stats.routes_by_strategy.get(decision.strategy.value, 0) + 1
        )

        # Guardar historial
        self._route_history.append((profile, decision))
        if len(self._route_history) > 1000:
            self._route_history = self._route_history[-500:]

        return result

    def _auto_adjust(self, profile: QueryProfile) -> Optional[RouteDecision]:
        """
        Ajusta la estrategia basándose en latencias históricas.

        Returns:
            RouteDecision ajustada o None si no hay suficiente data.
        """
        if not self.enable_auto_learning:
            return None

        # Necesitamos al menos 10 muestras de cada estrategia relevante
        relevant = [SearchStrategy.EXACT, SearchStrategy.APPROXIMATE_HRM2]
        latencies = {}
        for s in relevant:
            history = self._latency_history[s]
            if len(history) >= 10:
                latencies[s] = sum(history) / len(history)

        if len(latencies) < 2:
            return None

        # Seleccionar la más rápida
        best = min(latencies, key=latencies.get)
        self._stats.auto_adjustments += 1

        return RouteDecision(
            strategy=best,
            confidence=0.7,
            reason=f"Auto-learning: {best.value} es más rápido ({latencies[best]:.1f}ms)",
            estimated_latency_ms=latencies[best],
        )

    def _estimate_latency(self, strategy: SearchStrategy, profile: QueryProfile) -> float:
        """Estima latencia basándose en historial."""
        history = self._latency_history[strategy]
        if not history:
            return 0.0
        return sum(history) / len(history)

    def _record_latency(self, strategy: SearchStrategy, latency_ms: float) -> None:
        """Registra latencia para auto-learning."""
        history = self._latency_history[strategy]
        history.append(latency_ms)
        if len(history) > self.learning_window:
            self._latency_history[strategy] = history[-self.learning_window:]

        # Actualizar promedio en stats
        self._stats.total_latency_samples[strategy.value] = len(history)
        self._stats.avg_latency_by_strategy[strategy.value] = sum(history) / len(history)

    def get_stats(self) -> Dict[str, Any]:
        """Retorna estadísticas del router."""
        return {
            "total_routes": self._stats.total_routes,
            "routes_by_strategy": dict(self._stats.routes_by_strategy),
            "avg_latency_by_strategy": {
                k: round(v, 2) for k, v in self._stats.avg_latency_by_strategy.items()
            },
            "auto_adjustments": self._stats.auto_adjustments,
            "registered_strategies": [s.value for s in self._strategies.keys()],
            "auto_learning_enabled": self.enable_auto_learning,
        }
