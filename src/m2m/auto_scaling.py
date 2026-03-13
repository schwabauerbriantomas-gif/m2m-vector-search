"""
Auto-Scaling para M2M Cluster.

Características:
- Escalado automático basado en métricas
- Horizontal scaling (añadir/remover nodos)
- Vertical scaling (ajustar recursos por nodo)
- Load-based scaling
- Predictive scaling
- Cost optimization
"""

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

import numpy as np


class ScalingDirection(Enum):
    """Dirección de escalado."""

    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    NONE = "none"


class ScalingTrigger(Enum):
    """Trigger de escalado."""

    CPU_THRESHOLD = "cpu_threshold"
    MEMORY_THRESHOLD = "memory_threshold"
    LATENCY_THRESHOLD = "latency_threshold"
    QPS_THRESHOLD = "qps_threshold"
    PREDICTIVE = "predictive"
    MANUAL = "manual"


@dataclass
class NodeMetrics:
    """Métricas de un nodo del cluster."""

    node_id: str
    cpu_percent: float
    memory_percent: float
    qps: float
    latency_ms: float
    active_queries: int
    uptime_seconds: float
    timestamp: float = field(default_factory=time.time)


@dataclass
class ScalingDecision:
    """Decisión de escalado."""

    action: ScalingDirection
    trigger: ScalingTrigger
    current_nodes: int
    target_nodes: int
    reason: str
    metrics: Dict[str, float]
    timestamp: float = field(default_factory=time.time)


class MetricsCollector:
    """
    Recolector de métricas del cluster.

    Agrega métricas de todos los nodos y calcula
    estadísticas globales.
    """

    def __init__(self, history_size: int = 100):
        """
        Args:
            history_size: Número de muestras a mantener
        """
        self.history_size = history_size
        self._metrics_history: Dict[str, List[NodeMetrics]] = {}

    def record(self, metrics: NodeMetrics):
        """
        Registra métricas de un nodo.

        Args:
            metrics: Métricas del nodo
        """
        node_id = metrics.node_id

        if node_id not in self._metrics_history:
            self._metrics_history[node_id] = []

        self._metrics_history[node_id].append(metrics)

        # Mantener tamaño
        if len(self._metrics_history[node_id]) > self.history_size:
            self._metrics_history[node_id] = self._metrics_history[node_id][-self.history_size :]

    def get_cluster_stats(self) -> Dict[str, float]:
        """
        Calcula estadísticas agregadas del cluster.

        Returns:
            Diccionario con métricas agregadas
        """
        if not self._metrics_history:
            return {
                "nodes": 0,
                "avg_cpu": 0.0,
                "avg_memory": 0.0,
                "total_qps": 0.0,
                "avg_latency_ms": 0.0,
                "total_active_queries": 0,
            }

        # Agregar métricas actuales de todos los nodos
        cpus = []
        memories = []
        qps_list = []
        latencies = []
        active_queries = []

        for node_id, history in self._metrics_history.items():
            if history:
                latest = history[-1]
                cpus.append(latest.cpu_percent)
                memories.append(latest.memory_percent)
                qps_list.append(latest.qps)
                latencies.append(latest.latency_ms)
                active_queries.append(latest.active_queries)

        return {
            "nodes": len(self._metrics_history),
            "avg_cpu": float(np.mean(cpus)) if cpus else 0.0,
            "avg_memory": float(np.mean(memories)) if memories else 0.0,
            "total_qps": float(sum(qps_list)),
            "avg_latency_ms": float(np.mean(latencies)) if latencies else 0.0,
            "total_active_queries": sum(active_queries),
        }

    def get_trend(self, metric: str, window: int = 10) -> str:
        """
        Detecta tendencia de una métrica.

        Args:
            metric: Nombre de la métrica
            window: Ventana de análisis

        Returns:
            'increasing', 'decreasing', 'stable'
        """
        if not self._metrics_history:
            return "stable"

        # Recopilar valores históricos
        all_values = []
        for history in self._metrics_history.values():
            for m in history[-window:]:
                if hasattr(m, metric):
                    all_values.append(getattr(m, metric))

        if len(all_values) < 3:
            return "stable"

        # Calcular pendiente simple
        first_half = np.mean(all_values[: len(all_values) // 2])
        second_half = np.mean(all_values[len(all_values) // 2 :])

        diff = second_half - first_half
        threshold = first_half * 0.1  # 10% de cambio

        if diff > threshold:
            return "increasing"
        elif diff < -threshold:
            return "decreasing"
        else:
            return "stable"


class AutoScaler:
    """
    Auto-scaler para M2M Cluster.

    Monitoriza métricas y toma decisiones de escalado
    automáticamente.
    """

    def __init__(
        self,
        min_nodes: int = 1,
        max_nodes: int = 10,
        scale_up_threshold: float = 80.0,
        scale_down_threshold: float = 30.0,
        cooldown_seconds: float = 60.0,
        enable_predictive: bool = False,
    ):
        """
        Args:
            min_nodes: Mínimo número de nodos
            max_nodes: Máximo número de nodos
            scale_up_threshold: Umbral de CPU para escalar up (%)
            scale_down_threshold: Umbral de CPU para escalar down (%)
            cooldown_seconds: Tiempo entre escalados
            enable_predictive: Habilitar scaling predictivo
        """
        self.min_nodes = min_nodes
        self.max_nodes = max_nodes
        self.scale_up_threshold = scale_up_threshold
        self.scale_down_threshold = scale_down_threshold
        self.cooldown_seconds = cooldown_seconds
        self.enable_predictive = enable_predictive

        self.metrics_collector = MetricsCollector()

        # Estado
        self.current_nodes = min_nodes
        self._last_scale_time = 0.0
        self._scaling_history: List[ScalingDecision] = []

        # Callbacks
        self._scale_up_callback: Optional[Callable] = None
        self._scale_down_callback: Optional[Callable] = None

    def register_callbacks(
        self, scale_up: Optional[Callable] = None, scale_down: Optional[Callable] = None
    ):
        """
        Registra callbacks para escalado.

        Args:
            scale_up: Función para añadir nodo
            scale_down: Función para remover nodo
        """
        self._scale_up_callback = scale_up
        self._scale_down_callback = scale_down

    def update_metrics(self, metrics: NodeMetrics):
        """
        Actualiza métricas de un nodo.

        Args:
            metrics: Métricas del nodo
        """
        self.metrics_collector.record(metrics)

    def evaluate_scaling(self) -> Optional[ScalingDecision]:
        """
        Evalúa si es necesario escalar.

        Returns:
            ScalingDecision si se necesita escalar, None si no
        """
        # Verificar cooldown
        if time.time() - self._last_scale_time < self.cooldown_seconds:
            return None

        stats = self.metrics_collector.get_cluster_stats()

        # Verificar tendencias si predictive enabled
        if self.enable_predictive:
            trend = self.metrics_collector.get_trend("cpu_percent")
            if trend == "increasing" and stats["avg_cpu"] > self.scale_up_threshold * 0.8:
                return self._create_decision(
                    ScalingDirection.SCALE_UP,
                    ScalingTrigger.PREDICTIVE,
                    stats,
                    "Predictive scaling: CPU trend increasing",
                )

        # Verificar umbrales
        if stats["avg_cpu"] > self.scale_up_threshold:
            return self._create_decision(
                ScalingDirection.SCALE_UP,
                ScalingTrigger.CPU_THRESHOLD,
                stats,
                f"CPU {stats['avg_cpu']:.1f}% > threshold {self.scale_up_threshold}%",
            )

        if stats["avg_memory"] > self.scale_up_threshold:
            return self._create_decision(
                ScalingDirection.SCALE_UP,
                ScalingTrigger.MEMORY_THRESHOLD,
                stats,
                f"Memory {stats['avg_memory']:.1f}% > threshold {self.scale_up_threshold}%",
            )

        if stats["avg_latency_ms"] > 20.0:  # 20ms threshold
            return self._create_decision(
                ScalingDirection.SCALE_UP,
                ScalingTrigger.LATENCY_THRESHOLD,
                stats,
                f"Latency {stats['avg_latency_ms']:.1f}ms > 20ms threshold",
            )

        # Verificar scale down
        if (
            stats["avg_cpu"] < self.scale_down_threshold
            and stats["avg_memory"] < self.scale_down_threshold
            and self.current_nodes > self.min_nodes
        ):
            return self._create_decision(
                ScalingDirection.SCALE_DOWN,
                ScalingTrigger.CPU_THRESHOLD,
                stats,
                f"CPU {stats['avg_cpu']:.1f}% < threshold {self.scale_down_threshold}%",
            )

        return None

    def _create_decision(
        self,
        direction: ScalingDirection,
        trigger: ScalingTrigger,
        stats: Dict[str, float],
        reason: str,
    ) -> ScalingDecision:
        """
        Crea decisión de escalado.

        Args:
            direction: Dirección de escalado
            trigger: Trigger que causó la decisión
            stats: Métricas actuales
            reason: Razón de la decisión

        Returns:
            ScalingDecision
        """
        target_nodes = self.current_nodes

        if direction == ScalingDirection.SCALE_UP:
            target_nodes = min(self.current_nodes + 1, self.max_nodes)
        elif direction == ScalingDirection.SCALE_DOWN:
            target_nodes = max(self.current_nodes - 1, self.min_nodes)

        return ScalingDecision(
            action=direction,
            trigger=trigger,
            current_nodes=self.current_nodes,
            target_nodes=target_nodes,
            reason=reason,
            metrics=stats,
        )

    def execute_scaling(self, decision: ScalingDecision) -> bool:
        """
        Ejecuta decisión de escalado.

        Args:
            decision: Decisión a ejecutar

        Returns:
            True si se ejecutó correctamente
        """
        if decision.action == ScalingDirection.NONE:
            return False

        # Verificar que el target sea diferente
        if decision.target_nodes == self.current_nodes:
            return False

        # Ejecutar callback apropiado
        success = False

        if decision.action == ScalingDirection.SCALE_UP and self._scale_up_callback:
            success = self._scale_up_callback()
        elif decision.action == ScalingDirection.SCALE_DOWN and self._scale_down_callback:
            success = self._scale_down_callback()

        if success:
            self.current_nodes = decision.target_nodes
            self._last_scale_time = time.time()
            self._scaling_history.append(decision)

        return success

    def get_scaling_history(self, n: int = 10) -> List[ScalingDecision]:
        """
        Obtiene historial de escalados.

        Args:
            n: Número de decisiones a retornar

        Returns:
            Lista de decisiones
        """
        return self._scaling_history[-n:]

    def get_stats(self) -> Dict[str, Any]:
        """
        Obtiene estadísticas del auto-scaler.

        Returns:
            Diccionario con estadísticas
        """
        cluster_stats = self.metrics_collector.get_cluster_stats()

        return {
            "current_nodes": self.current_nodes,
            "min_nodes": self.min_nodes,
            "max_nodes": self.max_nodes,
            "cluster": cluster_stats,
            "last_scale_time": self._last_scale_time,
            "total_scaling_events": len(self._scaling_history),
            "predictive_enabled": self.enable_predictive,
        }


class HorizontalScaler:
    """
    Escalador horizontal para añadir/remover nodos.

    Implementa la lógica real de escalado (provisioning).
    """

    def __init__(self, node_template: Dict[str, Any]):
        """
        Args:
            node_template: Template de configuración para nuevos nodos
        """
        self.node_template = node_template
        self._active_nodes: Dict[str, Any] = {}

    def scale_up(self) -> bool:
        """
        Añade un nuevo nodo al cluster.

        Returns:
            True si se añadió correctamente
        """
        import uuid

        node_id = f"node-{uuid.uuid4().hex[:8]}"

        # Simular creación de nodo
        # En producción: usar Kubernetes/Docker/Cloud API
        self._active_nodes[node_id] = {
            "id": node_id,
            "status": "starting",
            "config": self.node_template,
            "created_at": time.time(),
        }

        # Simular arranque
        time.sleep(0.1)  # En producción: esperar health check
        self._active_nodes[node_id]["status"] = "ready"

        print(f"[AutoScale] Added node {node_id}")
        return True

    def scale_down(self) -> bool:
        """
        Remueve un nodo del cluster.

        Returns:
            True si se removió correctamente
        """
        if len(self._active_nodes) <= 1:
            print("[AutoScale] Cannot remove last node")
            return False

        # Remover nodo menos utilizado
        # En producción: drenar conexiones primero
        node_id = list(self._active_nodes.keys())[-1]

        self._active_nodes[node_id]["status"] = "draining"
        time.sleep(0.1)  # Simular drenado
        del self._active_nodes[node_id]

        print(f"[AutoScale] Removed node {node_id}")
        return True

    def get_active_nodes(self) -> List[str]:
        """Retorna lista de nodos activos."""
        return list(self._active_nodes.keys())
