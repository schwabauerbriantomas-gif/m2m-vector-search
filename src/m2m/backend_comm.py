"""
Backend Communication Protocol — Protocolo de comunicación mejorado entre backends.

Mensajes estructurados, health checks, métricas de rendimiento y
dead letter queue para queries fallidas.

Basado en MASFactory's Communication Protocol (MessageBus, Message).
"""

from __future__ import annotations

import json
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Deque, Dict, List, Optional


class BackendMsgType(str, Enum):
    """Tipos de mensajes entre backends."""
    SEARCH_REQUEST = "search_request"
    SEARCH_RESULT = "search_result"
    INDEX_REQUEST = "index_request"
    INDEX_RESULT = "index_result"
    HEALTH_CHECK = "health_check"
    HEALTH_RESPONSE = "health_response"
    METRICS_REPORT = "metrics_report"
    ERROR = "error"
    SHUTDOWN = "shutdown"


@dataclass
class BackendMessage:
    """
    Mensaje estructurado entre backends M2M.

    Incluye metadata para routing, prioridad, y tracking.
    """
    sender: str
    receiver: str
    msg_type: BackendMsgType
    content: Dict[str, Any]
    message_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    timestamp: float = field(default_factory=time.time)
    priority: int = 0          # -1=low, 0=normal, 1=high, 2=critical
    ttl_seconds: float = 60.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    parent_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serializa el mensaje a dict."""
        return {
            "id": self.message_id,
            "type": self.msg_type.value,
            "from": self.sender,
            "to": self.receiver,
            "content": self.content,
            "timestamp": self.timestamp,
            "priority": self.priority,
            "metadata": self.metadata,
            "parent_id": self.parent_id,
        }

    def to_json(self) -> str:
        """Serializa el mensaje a JSON."""
        return json.dumps(self.to_dict(), ensure_ascii=False, default=str)

    @classmethod
    def from_json(cls, json_str: str) -> "BackendMessage":
        """Deserializa un mensaje desde JSON."""
        data = json.loads(json_str)
        return cls(
            sender=data["from"],
            receiver=data["to"],
            msg_type=BackendMsgType(data["type"]),
            content=data["content"],
            message_id=data.get("id", uuid.uuid4().hex[:12]),
            timestamp=data.get("timestamp", time.time()),
            priority=data.get("priority", 0),
            metadata=data.get("metadata", {}),
            parent_id=data.get("parent_id"),
        )

    def is_expired(self) -> bool:
        """Verifica si el mensaje expiró."""
        if self.ttl_seconds <= 0:
            return False
        return (time.time() - self.timestamp) > self.ttl_seconds

    def __repr__(self) -> str:
        return (
            f"BackendMessage({self.msg_type.value}, "
            f"{self.sender}->{self.receiver}, "
            f"priority={self.priority})"
        )


@dataclass
class BackendHealth:
    """Estado de salud de un backend."""
    name: str
    is_healthy: bool = True
    last_heartbeat: float = field(default_factory=time.time)
    total_queries: int = 0
    total_errors: int = 0
    avg_latency_ms: float = 0.0
    memory_usage_mb: float = 0.0
    index_size: int = 0
    error_message: str = ""


@dataclass
class BackendMetrics:
    """Métricas de rendimiento de un backend."""
    name: str
    queries_per_second: float = 0.0
    avg_latency_ms: float = 0.0
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    error_rate: float = 0.0
    total_queries: int = 0
    total_errors: int = 0
    uptime_seconds: float = 0.0


class BackendComm:
    """
    Protocolo de comunicación entre backends M2M.

    Provee:
    - Message bus con prioridades
    - Health checks automáticos
    - Métricas de rendimiento por backend
    - Dead letter queue para mensajes fallidos
    - Request/response tracking

    Example:
        >>> comm = BackendComm()
        >>> comm.register_backend("cpu", health_check_fn=my_health)
        >>> comm.send_search_request("cpu", query_id="q1", k=10)
        >>> health = comm.get_health("cpu")
    """

    def __init__(
        self,
        max_message_history: int = 10000,
        dead_letter_max: int = 1000,
        health_check_interval: float = 30.0,
    ):
        """
        Args:
            max_message_history: Máximo mensajes en historial.
            dead_letter_max: Máximo mensajes en dead letter queue.
            health_check_interval: Intervalo entre health checks (segundos).
        """
        self.max_message_history = max_message_history
        self.dead_letter_max = dead_letter_max
        self.health_check_interval = health_check_interval

        # Message bus
        self._messages: Deque[BackendMessage] = deque(maxlen=max_message_history)
        self._dead_letter: Deque[BackendMessage] = deque(maxlen=dead_letter_max)

        # Backend registry
        self._backends: Dict[str, Dict[str, Any]] = {}
        self._health_checks: Dict[str, Callable[[], BackendHealth]] = {}
        self._health_status: Dict[str, BackendHealth] = {}

        # Metrics tracking
        self._latency_samples: Dict[str, Deque[float]] = {}
        self._last_health_check: float = 0.0

        # Pending requests (request_id -> message)
        self._pending: Dict[str, BackendMessage] = {}

    def register_backend(
        self,
        name: str,
        health_check_fn: Optional[Callable[[], BackendHealth]] = None,
    ) -> None:
        """
        Registra un backend en el protocolo de comunicación.

        Args:
            name: Nombre del backend.
            health_check_fn: Función de health check opcional.
        """
        self._backends[name] = {"registered_at": time.time()}
        if health_check_fn:
            self._health_checks[name] = health_check_fn
        self._latency_samples[name] = deque(maxlen=1000)

    def unregister_backend(self, name: str) -> None:
        """Desregistra un backend."""
        self._backends.pop(name, None)
        self._health_checks.pop(name, None)
        self._health_status.pop(name, None)
        self._latency_samples.pop(name, None)

    def send(
        self,
        sender: str,
        receiver: str,
        msg_type: BackendMsgType,
        content: Dict[str, Any],
        priority: int = 0,
        parent_id: Optional[str] = None,
    ) -> str:
        """
        Envía un mensaje al bus.

        Args:
            sender: Nombre del emisor.
            receiver: Nombre del receptor (o "all" para broadcast).
            msg_type: Tipo de mensaje.
            content: Contenido del mensaje.
            priority: Prioridad (-1 a 2).
            parent_id: ID del mensaje padre para encadenar.

        Returns:
            ID del mensaje enviado.
        """
        msg = BackendMessage(
            sender=sender,
            receiver=receiver,
            msg_type=msg_type,
            content=content,
            priority=priority,
            parent_id=parent_id,
        )

        self._messages.append(msg)
        return msg.message_id

    def send_search_request(
        self,
        receiver: str,
        query_id: str,
        k: int = 10,
        backend: str = "unknown",
        **kwargs,
    ) -> str:
        """
        Envía una solicitud de búsqueda.

        Args:
            receiver: Backend destino.
            query_id: ID único de la query.
            k: Número de resultados.
            backend: Backend emisor.
            **kwargs: Parámetros adicionales.

        Returns:
            ID del mensaje.
        """
        return self.send(
            sender=backend,
            receiver=receiver,
            msg_type=BackendMsgType.SEARCH_REQUEST,
            content={"query_id": query_id, "k": k, **kwargs},
            priority=1,
        )

    def send_search_result(
        self,
        sender: str,
        request_id: str,
        query_id: str,
        result_count: int,
        latency_ms: float,
        success: bool = True,
        error: str = "",
    ) -> str:
        """
        Envía un resultado de búsqueda.

        Args:
            sender: Backend que ejecutó la búsqueda.
            request_id: ID del mensaje de request original.
            query_id: ID de la query.
            result_count: Número de resultados.
            latency_ms: Latencia en ms.
            success: Si la búsqueda fue exitosa.
            error: Mensaje de error si falló.

        Returns:
            ID del mensaje.
        """
        # Remover de pending
        self._pending.pop(request_id, None)

        return self.send(
            sender=sender,
            receiver="supervisor",
            msg_type=BackendMsgType.SEARCH_RESULT,
            content={
                "query_id": query_id,
                "result_count": result_count,
                "latency_ms": latency_ms,
                "success": success,
                "error": error,
            },
            parent_id=request_id,
            priority=1,
        )

    def receive(
        self,
        receiver: str,
        msg_type: Optional[BackendMsgType] = None,
        last_n: Optional[int] = None,
        min_priority: int = 0,
    ) -> List[BackendMessage]:
        """
        Recibe mensajes para un receptor.

        Args:
            receiver: Nombre del receptor.
            msg_type: Filtrar por tipo (None = todos).
            last_n: Limitar a los últimos N.
            min_priority: Prioridad mínima.

        Returns:
            Lista de mensajes.
        """
        filtered = [
            m for m in self._messages
            if (m.receiver == receiver or m.receiver == "all")
            and (msg_type is None or m.msg_type == msg_type)
            and m.priority >= min_priority
            and not m.is_expired()
        ]

        if last_n is not None:
            filtered = filtered[-last_n:]

        return filtered

    def report_error(
        self,
        sender: str,
        error: str,
        query_id: str = "",
        severity: str = "error",
    ) -> str:
        """
        Reporta un error al bus.

        Args:
            sender: Backend que reporta el error.
            error: Mensaje de error.
            query_id: ID de la query asociada.
            severity: Severidad (error, warning, critical).

        Returns:
            ID del mensaje de error.
        """
        return self.send(
            sender=sender,
            receiver="supervisor",
            msg_type=BackendMsgType.ERROR,
            content={"error": error, "query_id": query_id, "severity": severity},
            priority=2 if severity == "critical" else 1,
        )

    def record_latency(self, backend: str, latency_ms: float) -> None:
        """Registra latencia para métricas de un backend."""
        if backend in self._latency_samples:
            self._latency_samples[backend].append(latency_ms)

    def get_health(self, backend: str) -> Optional[BackendHealth]:
        """
        Obtiene el estado de salud de un backend.

        Ejecuta el health check si el intervalo ha pasado.

        Args:
            backend: Nombre del backend.

        Returns:
            BackendHealth o None si no está registrado.
        """
        if backend not in self._backends:
            return None

        # Ejecutar health check si es tiempo
        now = time.time()
        if (
            backend in self._health_checks
            and (now - self._last_health_check) >= self.health_check_interval
        ):
            try:
                health = self._health_checks[backend]()
                self._health_status[backend] = health
                self._last_health_check = now
            except Exception:
                # Si el health check falla, marcar como unhealthy
                self._health_status[backend] = BackendHealth(
                    name=backend,
                    is_healthy=False,
                    error_message="Health check failed",
                )

        return self._health_status.get(backend, BackendHealth(name=backend))

    def get_all_health(self) -> Dict[str, Dict[str, Any]]:
        """Obtiene health de todos los backends registrados."""
        result = {}
        for name in self._backends:
            health = self.get_health(name)
            if health:
                result[name] = {
                    "healthy": health.is_healthy,
                    "last_heartbeat": health.last_heartbeat,
                    "total_queries": health.total_queries,
                    "total_errors": health.total_errors,
                    "avg_latency_ms": round(health.avg_latency_ms, 2),
                    "error_message": health.error_message,
                }
        return result

    def get_metrics(self, backend: str) -> BackendMetrics:
        """
        Calcula métricas de rendimiento de un backend.

        Args:
            backend: Nombre del backend.

        Returns:
            BackendMetrics con percentiles y tasas.
        """
        samples = list(self._latency_samples.get(backend, []))
        health = self.get_health(backend)

        metrics = BackendMetrics(name=backend)
        metrics.total_queries = health.total_queries if health else 0
        metrics.total_errors = health.total_errors if health else 0
        metrics.error_rate = (
            metrics.total_errors / metrics.total_queries if metrics.total_queries > 0 else 0.0
        )

        if samples:
            sorted_samples = sorted(samples)
            n = len(sorted_samples)
            metrics.avg_latency_ms = sum(sorted_samples) / n
            metrics.p50_latency_ms = sorted_samples[int(n * 0.50)]
            metrics.p95_latency_ms = sorted_samples[int(n * 0.95)] if n > 1 else sorted_samples[-1]
            metrics.p99_latency_ms = sorted_samples[int(n * 0.99)] if n > 10 else sorted_samples[-1]

            # QPS estimado
            if len(samples) > 1 and metrics.avg_latency_ms > 0:
                metrics.queries_per_second = 1000.0 / metrics.avg_latency_ms

        return metrics

    def get_dead_letter_queue(self) -> List[Dict[str, Any]]:
        """Retorna los mensajes en la dead letter queue."""
        return [m.to_dict() for m in self._dead_letter]

    def requeue_dead_letters(self) -> int:
        """Re-encola mensajes de la dead letter queue."""
        count = len(self._dead_letter)
        while self._dead_letter:
            msg = self._dead_letter.popleft()
            msg.timestamp = time.time()  # Reset TTL
            self._messages.append(msg)
        return count

    def get_stats(self) -> Dict[str, Any]:
        """Retorna estadísticas del protocolo de comunicación."""
        type_counts: Dict[str, int] = {}
        priority_counts: Dict[int, int] = {}
        for m in self._messages:
            type_counts[m.msg_type.value] = type_counts.get(m.msg_type.value, 0) + 1
            priority_counts[m.priority] = priority_counts.get(m.priority, 0) + 1

        return {
            "total_messages": len(self._messages),
            "by_type": type_counts,
            "by_priority": priority_counts,
            "dead_letter_count": len(self._dead_letter),
            "registered_backends": list(self._backends.keys()),
            "pending_requests": len(self._pending),
        }

    def clear(self) -> None:
        """Limpia todo el estado del protocolo."""
        self._messages.clear()
        self._dead_letter.clear()
        self._pending.clear()
