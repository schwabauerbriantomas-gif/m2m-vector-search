"""
Quality Reflector — Patrón Reflection para evaluación de calidad de resultados.

Evalúa calidad de búsqueda (precision@k, recall@k), compara resultados
entre backends, detecta anomalías y sugiere re-indexación si la calidad decae.

Basado en MASFactory's ReflectionAgent pattern.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np


class QualityLevel(str, Enum):
    """Niveles de calidad."""
    EXCELLENT = "excellent"    # score >= 0.9
    GOOD = "good"              # score >= 0.7
    ACCEPTABLE = "acceptable"  # score >= 0.5
    POOR = "poor"              # score >= 0.3
    CRITICAL = "critical"      # score < 0.3


@dataclass
class QualityReport:
    """Reporte de calidad de una búsqueda."""
    precision_at_k: float
    recall_at_k: float
    quality_level: QualityLevel
    anomalies: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    backend_used: str = ""
    timestamp: float = field(default_factory=time.time)


@dataclass
class ReflectorStats:
    """Estadísticas del reflector."""
    total_evaluations: int = 0
    avg_precision: float = 0.0
    avg_recall: float = 0.0
    anomalies_detected: int = 0
    reindex_suggestions: int = 0
    quality_history: List[QualityReport] = field(default_factory=list)


class QualityReflector:
    """
    Reflector de calidad para búsqueda M2M.

    Evalúa resultados de búsqueda comparándolos con ground truth o entre
    backends. Detecta anomalías y sugiere re-indexación cuando la calidad
    decae por debajo de umbrales configurables.

    Example:
        >>> reflector = QualityReflector()
        >>> report = reflector.evaluate(
        ...     result_ids=[1, 3, 5],
        ...     ground_truth=[1, 2, 3, 4, 5],
        ...     k=5,
        ...     backend="cpu",
        ... )
    """

    # Umbrales de calidad
    _PRECISION_WARNING = 0.5
    _PRECISION_CRITICAL = 0.3
    _RECALL_WARNING = 0.6
    _RECALL_CRITICAL = 0.4
    _ANOMALY_SCORE_THRESHOLD = 0.3
    _MAX_HISTORY = 200
    _DEGRADATION_WINDOW = 10  # Evaluaciones para detectar degradación

    def __init__(
        self,
        precision_warning: float = 0.5,
        precision_critical: float = 0.3,
        recall_warning: float = 0.6,
        recall_critical: float = 0.4,
        enable_cross_backend: bool = True,
    ):
        """
        Args:
            precision_warning: Umbral de warning para precision@k.
            precision_critical: Umbral crítico para precision@k.
            recall_warning: Umbral de warning para recall@k.
            recall_critical: Umbral crítico para recall@k.
            enable_cross_backend: Habilitar comparación entre backends.
        """
        self.precision_warning = precision_warning
        self.precision_critical = precision_critical
        self.recall_warning = recall_warning
        self.recall_critical = recall_critical
        self.enable_cross_backend = enable_cross_backend

        self._stats = ReflectorStats()
        self._ground_truth_cache: Dict[str, Set[Any]] = {}
        self._backend_results: Dict[str, List[Tuple[List[Any], float]]] = {}

    def evaluate(
        self,
        result_ids: List[Any],
        ground_truth: Optional[List[Any]] = None,
        k: int = 10,
        backend: str = "unknown",
        query_vector: Optional[np.ndarray] = None,
        distances: Optional[List[float]] = None,
    ) -> QualityReport:
        """
        Evalúa la calidad de un resultado de búsqueda.

        Args:
            result_ids: IDs de los resultados devueltos.
            ground_truth: IDs relevantes (ground truth). Si None, solo se detectan anomalías.
            k: Número de resultados solicitados.
            backend: Nombre del backend usado.
            query_vector: Vector de query (para detección de anomalías).
            distances: Distancias de los resultados.

        Returns:
            QualityReport con métricas y sugerencias.
        """
        anomalies: List[str] = []
        suggestions: List[str] = []

        # Calcular métricas si hay ground truth
        precision = self._compute_precision(result_ids, ground_truth, k)
        recall = self._compute_recall(result_ids, ground_truth)

        # Detectar anomalías
        if query_vector is not None and distances is not None:
            query_anomalies = self._detect_distance_anomalies(distances)
            anomalies.extend(query_anomalies)

        # Detección de duplicados
        if len(result_ids) != len(set(result_ids)):
            anomalies.append(f"Se encontraron {len(result_ids) - len(set(result_ids))} IDs duplicados")

        # Verificar si se devolvieron menos resultados que k
        if len(result_ids) < k:
            anomalies.append(f"Solo se devolvieron {len(result_ids)}/{k} resultados")

        # Comparar con historial de calidad
        degradation = self._detect_degradation(precision, recall)
        if degradation:
            anomalies.append(degradation)
            suggestions.append("Considerar re-indexar el dataset")

        # Clasificar nivel de calidad
        quality_level = self._classify_quality(precision, recall)

        # Generar sugerencias basadas en nivel
        suggestions.extend(self._generate_suggestions(quality_level, precision, recall, backend))

        report = QualityReport(
            precision_at_k=precision,
            recall_at_k=recall,
            quality_level=quality_level,
            anomalies=anomalies,
            suggestions=suggestions,
            backend_used=backend,
        )

        # Actualizar stats
        self._update_stats(report)

        # Guardar resultado del backend para cross-comparison
        if self.enable_cross_backend:
            self._backend_results.setdefault(backend, []).append((result_ids, precision))

        return report

    def evaluate_cross_backend(
        self,
        results_map: Dict[str, List[Any]],
        ground_truth: Optional[List[Any]] = None,
        k: int = 10,
    ) -> Dict[str, QualityReport]:
        """
        Compara resultados entre múltiples backends.

        Args:
            results_map: {backend_name: [result_ids]}.
            ground_truth: IDs relevantes.
            k: Número de resultados.

        Returns:
            Dict con QualityReport por backend.
        """
        reports = {}
        for backend, result_ids in results_map.items():
            reports[backend] = self.evaluate(
                result_ids=result_ids,
                ground_truth=ground_truth,
                k=k,
                backend=backend,
            )

        # Comparación entre backends
        if len(reports) > 1:
            self._compare_backend_reports(reports)

        return reports

    def _compute_precision(
        self, result_ids: List[Any], ground_truth: Optional[List[Any]], k: int
    ) -> float:
        """Calcula precision@k."""
        if ground_truth is None or not ground_truth:
            return 1.0  # Sin ground truth, asumir buena calidad

        relevant = set(ground_truth)
        if not relevant:
            return 1.0

        hits = sum(1 for rid in result_ids if rid in relevant)
        return hits / min(len(result_ids), k)

    def _compute_recall(
        self, result_ids: List[Any], ground_truth: Optional[List[Any]]
    ) -> float:
        """Calcula recall@k."""
        if ground_truth is None or not ground_truth:
            return 1.0

        relevant = set(ground_truth)
        if not relevant:
            return 1.0

        found = set(result_ids) & relevant
        return len(found) / len(relevant)

    def _detect_distance_anomalies(self, distances: List[float]) -> List[str]:
        """Detecta anomalías en las distancias de los resultados."""
        if not distances or len(distances) < 3:
            return []

        anomalies = []
        arr = np.array(distances)

        # Verificar que las distancias están ordenadas (deberían ser crecientes)
        if not np.all(np.diff(arr) >= -1e-6):
            anomalies.append("Las distancias no están en orden creciente")

        # Detectar saltos anómalos en distancias
        if len(arr) > 2:
            diffs = np.diff(arr)
            median_diff = np.median(diffs)
            if median_diff > 0:
                # Saltos > 5x la mediana son anomalías
                big_jumps = np.where(diffs > 5 * median_diff)[0]
                if len(big_jumps) > 0:
                    anomalies.append(
                        f"Saltos anómalos en distancias en posiciones {big_jumps.tolist()}"
                    )

        # Verificar si las distancias son demasiado uniformes (posible problema)
        if len(arr) > 5:
            std = np.std(arr)
            mean = np.mean(arr)
            cv = std / mean if mean > 0 else 0  # Coefficient of variation
            if cv < 0.01:
                anomalies.append("Distancias demasiado uniformes (posible índice degenerado)")

        return anomalies

    def _detect_degradation(self, precision: float, recall: float) -> Optional[str]:
        """Detecta degradación de calidad respecto al historial."""
        history = self._stats.quality_history
        if len(history) < self._DEGRADATION_WINDOW:
            return None

        recent = history[-self._DEGRADATION_WINDOW:]
        recent_precision = [r.precision_at_k for r in recent]
        recent_recall = [r.recall_at_k for r in recent]

        avg_prec = sum(recent_precision) / len(recent_precision)
        avg_rec = sum(recent_recall) / len(recent_recall)

        # Si la métrica actual es significativamente peor que el promedio reciente
        if precision < avg_prec * 0.7 and avg_prec > 0.5:
            return (
                f"Degradación de precision: actual={precision:.2f} vs "
                f"promedio reciente={avg_prec:.2f}"
            )
        if recall < avg_rec * 0.7 and avg_rec > 0.5:
            return (
                f"Degradación de recall: actual={recall:.2f} vs "
                f"promedio reciente={avg_rec:.2f}"
            )

        return None

    def _classify_quality(self, precision: float, recall: float) -> QualityLevel:
        """Clasifica el nivel de calidad."""
        score = (precision + recall) / 2

        if score >= 0.9:
            return QualityLevel.EXCELLENT
        elif score >= 0.7:
            return QualityLevel.GOOD
        elif score >= 0.5:
            return QualityLevel.ACCEPTABLE
        elif score >= 0.3:
            return QualityLevel.POOR
        return QualityLevel.CRITICAL

    def _generate_suggestions(
        self, level: QualityLevel, precision: float, recall: float, backend: str
    ) -> List[str]:
        """Genera sugerencias basadas en el nivel de calidad."""
        suggestions = []

        if level == QualityLevel.CRITICAL:
            suggestions.append("[!]️ Calidad CRÍTICA: re-indexar inmediatamente")
            self._stats.reindex_suggestions += 1
        elif level == QualityLevel.POOR:
            suggestions.append("[!]️ Calidad baja: considerar re-indexación")
            self._stats.reindex_suggestions += 1
        elif level == QualityLevel.ACCEPTABLE:
            if precision < self.precision_warning:
                suggestions.append("Precision por debajo del umbral de warning")
            if recall < self.recall_warning:
                suggestions.append("Recall por debajo del umbral de warning")

        return suggestions

    def _compare_backend_reports(self, reports: Dict[str, QualityReport]) -> None:
        """Compara reportes entre backends (interna, solo logging)."""
        import logging
        logger = logging.getLogger("m2m.reflector")

        best_precision = max(reports.values(), key=lambda r: r.precision_at_k)
        worst_precision = min(reports.values(), key=lambda r: r.precision_at_k)

        if best_precision.precision_at_k > 0 and worst_precision.precision_at_k > 0:
            ratio = best_precision.precision_at_k / worst_precision.precision_at_k
            if ratio > 1.5:
                logger.info(
                    f"Diferencia significativa entre backends: "
                    f"{best_precision.backend_used} ({best_precision.precision_at_k:.2f}) vs "
                    f"{worst_precision.backend_used} ({worst_precision.precision_at_k:.2f})"
                )

    def _update_stats(self, report: QualityReport) -> None:
        """Actualiza estadísticas internas."""
        self._stats.total_evaluations += 1
        self._stats.avg_precision = (
            self._stats.avg_precision * (self._stats.total_evaluations - 1) + report.precision_at_k
        ) / self._stats.total_evaluations
        self._stats.avg_recall = (
            self._stats.avg_recall * (self._stats.total_evaluations - 1) + report.recall_at_k
        ) / self._stats.total_evaluations
        self._stats.anomalies_detected += len(report.anomalies)

        self._stats.quality_history.append(report)
        if len(self._stats.quality_history) > self._MAX_HISTORY:
            self._stats.quality_history = self._stats.quality_history[-self._MAX_HISTORY:]

    def get_stats(self) -> Dict[str, Any]:
        """Retorna estadísticas del reflector."""
        return {
            "total_evaluations": self._stats.total_evaluations,
            "avg_precision": round(self._stats.avg_precision, 4),
            "avg_recall": round(self._stats.avg_recall, 4),
            "anomalies_detected": self._stats.anomalies_detected,
            "reindex_suggestions": self._stats.reindex_suggestions,
            "history_size": len(self._stats.quality_history),
        }

    def should_reindex(self) -> bool:
        """
        Determina si se debería re-indexar basándose en el historial.

        Returns:
            True si la calidad está degradándose consistentemente.
        """
        history = self._stats.quality_history
        if len(history) < self._DEGRADATION_WINDOW:
            return False

        recent = history[-self._DEGRADATION_WINDOW:]
        avg_recent = sum(r.precision_at_k for r in recent) / len(recent)

        return avg_recent < self.precision_critical
