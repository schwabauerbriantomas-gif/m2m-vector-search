"""
SOCEngine - Self-Organized Criticality para M2M.

El SOC permite que el sistema se auto-organice naturalmente.
Cuando acumula demasiada energía (supercritical), una "avalanche"
redistribuye la energía de manera natural, como arena en una pila.

Estados del sistema:
  SUBCRITICAL:   Sistema estable, sin reorganización necesaria
  CRITICAL:      Punto de transición, avalanche inminente
  SUPERCRITICAL: Sistema inestable, necesita reorganización
"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

import numpy as np

from .energy_api import EBMEnergy


class CriticalityState(Enum):
    """Estado de criticalidad del sistema."""

    SUBCRITICAL = "subcritical"
    CRITICAL = "critical"
    SUPERCRITICAL = "supercritical"


@dataclass
class CriticalityReport:
    """Reporte del estado de criticalidad."""

    state: CriticalityState
    index: float  # 0-1, mayor = más inestable
    energy_variance: float
    size_variance: float

    def needs_relaxation(self) -> bool:
        """True si el sistema necesita relajación."""
        return self.state == CriticalityState.SUPERCRITICAL

    def needs_monitoring(self) -> bool:
        """True si el sistema está en estado crítico."""
        return self.state in (CriticalityState.CRITICAL, CriticalityState.SUPERCRITICAL)


@dataclass
class AvalancheResult:
    """Resultado de una avalanche de reorganización."""

    affected_clusters: int
    energy_released: float
    duration_ms: float
    new_equilibrium: float

    def summary(self) -> str:
        return (
            f"Avalanche: {self.affected_clusters} clusters afectados, "
            f"energía liberada={self.energy_released:.4f}, "
            f"duración={self.duration_ms:.1f}ms"
        )


@dataclass
class RelaxationResult:
    """Resultado de relajación del sistema."""

    initial_energy: float
    final_energy: float
    energy_delta: float
    iterations: int

    @property
    def improved(self) -> bool:
        """True si la relajación mejoró el sistema."""
        return self.final_energy < self.initial_energy


@dataclass
class _Cluster:
    """Cluster interno para SOC."""

    center: np.ndarray
    energy: float
    size: int
    splat_indices: List[int]

    def reorganize(self):
        """Marca el cluster como reorganizado, reduciendo su energía."""
        self.energy *= 0.7  # Libera el 30%

    def receive_energy(self, amount: float):
        """Recibe energía de clusters vecinos."""
        self.energy += amount


class SOCEngine:
    """
    Motor de Self-Organized Criticality para M2M.

    Detecta cuando el sistema se vuelve inestable (supercritical)
    y desencadena avalanches para redistribuir la energía naturalmente.

    Uso:
        soc = SOCEngine(energy_api, splat_mu, splat_alpha, splat_kappa)
        report = soc.check_criticality()
        if report.needs_relaxation():
            result = soc.trigger_avalanche()
        soc.relax(iterations=10)
    """

    def __init__(
        self,
        energy_api: EBMEnergy,
        splat_mu: Optional[np.ndarray] = None,
        splat_alpha: Optional[np.ndarray] = None,
        splat_kappa: Optional[np.ndarray] = None,
        critical_threshold: float = 0.8,
    ):
        """
        Args:
            energy_api: EBMEnergy del sistema
            splat_mu: Centroides de splats [N, D]
            splat_alpha: Amplitudes [N]
            splat_kappa: Concentraciones [N]
            critical_threshold: Umbral de energía para desencadenar avalanche
        """
        self.energy_api = energy_api
        self.critical_threshold = critical_threshold
        self.avalanche_history: List[AvalancheResult] = []

        self.splat_mu = splat_mu
        self.splat_alpha = splat_alpha
        self.splat_kappa = splat_kappa

        # Clusters internos para SOC
        self._clusters: List[_Cluster] = []

    def update_splats(
        self,
        splat_mu: np.ndarray,
        splat_alpha: np.ndarray,
        splat_kappa: np.ndarray,
    ):
        """Actualiza los splats y reconstruye clusters internos."""
        self.splat_mu = np.asarray(splat_mu, dtype=np.float32)
        self.splat_alpha = np.asarray(splat_alpha, dtype=np.float32)
        self.splat_kappa = np.asarray(splat_kappa, dtype=np.float32)
        self._rebuild_clusters()

    def _rebuild_clusters(self):
        """Construye los clusters internos desde los splats."""
        if self.splat_mu is None or len(self.splat_mu) == 0:
            self._clusters = []
            return

        N = len(self.splat_mu)
        # Crear un cluster por splat (simplificado; en prod sería por coarse cluster)
        self._clusters = [
            _Cluster(
                center=self.splat_mu[i],
                energy=float(self.splat_alpha[i]),
                size=1,
                splat_indices=[i],
            )
            for i in range(N)
        ]

    def _compute_energy_variance(self) -> float:
        """Varianza de energía entre clusters."""
        if not self._clusters:
            return 0.0
        energies = [c.energy for c in self._clusters]
        return float(np.var(energies))

    def _compute_criticality_index(
        self, energy_variance: float, size_variance: float
    ) -> float:
        """
        Índice de criticalidad combinado.

        Combina:
        - Varianza de energía (alta = sistema heterogéneo)
        - Varianza de tamaños de cluster (alta = distribución desbalanceada)
        """
        # Normalizar usando sigmoide suave
        e_norm = energy_variance / (energy_variance + 1.0)
        s_norm = size_variance / (size_variance + 1.0)
        return float(0.6 * e_norm + 0.4 * s_norm)

    def check_criticality(self) -> CriticalityReport:
        """
        Verifica si el sistema está en estado crítico.

        Returns:
            CriticalityReport con estado y métricas.
        """
        energy_variance = self._compute_energy_variance()
        cluster_sizes = [len(c.splat_indices) for c in self._clusters] if self._clusters else [0]
        size_variance = float(np.var(cluster_sizes))

        index = self._compute_criticality_index(energy_variance, size_variance)

        if index < 0.3:
            state = CriticalityState.SUBCRITICAL
        elif index < 0.7:
            state = CriticalityState.CRITICAL
        else:
            state = CriticalityState.SUPERCRITICAL

        return CriticalityReport(
            state=state,
            index=index,
            energy_variance=energy_variance,
            size_variance=size_variance,
        )

    def trigger_avalanche(
        self, seed_point: Optional[np.ndarray] = None
    ) -> AvalancheResult:
        """
        Dispara una avalanche de reorganización.

        Las avalanches redistribuyen la energía del sistema de manera
        natural, similar a la arena cayendo en una pila de criticidad.

        Puede ser:
        - Automática: cuando se llama después de check_criticality()
        - Manual: disparada explícitamente por el usuario

        Args:
            seed_point: Punto semilla. Si None, usa el cluster de mayor energía.

        Returns:
            AvalancheResult con estadísticas de la avalanche.
        """
        import time

        start_time = time.time()

        if not self._clusters:
            return AvalancheResult(
                affected_clusters=0,
                energy_released=0.0,
                duration_ms=0.0,
                new_equilibrium=0.0,
            )

        # Encontrar punto semilla: cluster con mayor energía
        if seed_point is not None and self.splat_mu is not None:
            # Encontrar cluster más cercano al seed_point
            dists = [np.linalg.norm(c.center - seed_point) for c in self._clusters]
            start_cluster_idx = int(np.argmin(dists))
        else:
            energies = [c.energy for c in self._clusters]
            start_cluster_idx = int(np.argmax(energies))

        # BFS de avalanche
        affected_clusters = []
        queue = [start_cluster_idx]
        visited = set()
        total_energy_released = 0.0
        max_cascade = min(1000, len(self._clusters))

        while queue and len(affected_clusters) < max_cascade:
            idx = queue.pop(0)
            if idx in visited:
                continue
            visited.add(idx)

            cluster = self._clusters[idx]
            if cluster.energy > self.critical_threshold:
                affected_clusters.append(cluster)
                energy_released = cluster.energy * 0.3  # Libera 30%
                total_energy_released += energy_released

                # Redistribuir energía a vecinos
                neighbors = self._get_neighbor_clusters(idx)
                if neighbors:
                    share = energy_released / len(neighbors)
                    for n_idx in neighbors:
                        self._clusters[n_idx].receive_energy(share)
                        if self._clusters[n_idx].energy > self.critical_threshold:
                            queue.append(n_idx)

                cluster.reorganize()

        # Actualizar splat_alpha para reflejar la relajación
        if self.splat_alpha is not None and affected_clusters:
            for cluster in affected_clusters:
                for splat_idx in cluster.splat_indices:
                    if splat_idx < len(self.splat_alpha):
                        self.splat_alpha[splat_idx] *= 0.7
            # Actualizar energy_api
            self.energy_api.update_splats(
                self.splat_mu, self.splat_alpha, self.splat_kappa
            )

        new_eq = self._compute_equilibrium_state()
        duration_ms = (time.time() - start_time) * 1000

        result = AvalancheResult(
            affected_clusters=len(affected_clusters),
            energy_released=total_energy_released,
            duration_ms=duration_ms,
            new_equilibrium=new_eq,
        )
        self.avalanche_history.append(result)
        return result

    def _get_neighbor_clusters(self, cluster_idx: int, radius: float = 2.0) -> List[int]:
        """Obtiene índices de clusters vecinos dentro de un radio."""
        if not self._clusters or cluster_idx >= len(self._clusters):
            return []

        center = self._clusters[cluster_idx].center
        neighbors = []
        for i, c in enumerate(self._clusters):
            if i != cluster_idx:
                dist = np.linalg.norm(c.center - center)
                if dist <= radius:
                    neighbors.append(i)
        # Limitar a los 10 vecinos más cercanos
        if len(neighbors) > 10:
            dists = [np.linalg.norm(self._clusters[i].center - center) for i in neighbors]
            sorted_idx = np.argsort(dists)[:10]
            neighbors = [neighbors[i] for i in sorted_idx]
        return neighbors

    def _compute_equilibrium_state(self) -> float:
        """Calcula el estado de equilibrio actual del sistema."""
        if not self._clusters:
            return 0.0
        return float(np.mean([c.energy for c in self._clusters]))

    def relax(self, iterations: int = 10) -> RelaxationResult:
        """
        Relaja el sistema hacia un estado de menor energía.

        Proceso iterativo que ajusta los splats para minimizar
        la energía total del sistema. Útil para:
        - Optimización nocturna (mantenimiento)
        - Estabilización post-avalanche
        - Rutinas de mantenimiento periódico

        Args:
            iterations: Número de iteraciones de relajación

        Returns:
            RelaxationResult con métricas antes/después.
        """
        initial_energy = self.energy_api.free_energy()

        if self.splat_mu is None or self.splat_alpha is None:
            return RelaxationResult(
                initial_energy=initial_energy,
                final_energy=initial_energy,
                energy_delta=0.0,
                iterations=0,
            )

        for _ in range(iterations):
            if self.splat_alpha is not None and len(self.splat_alpha) > 0:
                # Ajustar alpha según uso (clusters activos aumentan, inactivos disminuyen)
                usage = np.exp(-self.splat_kappa * 0.1)  # proxy de uso
                self.splat_alpha *= (1 + usage * 0.05)

                # Normalizar alpha para que sumen a 1
                total = np.sum(self.splat_alpha)
                if total > 0:
                    self.splat_alpha /= total

            if self.splat_kappa is not None and self.splat_mu is not None:
                # Ajustar kappa según densidad local (simplificado)
                self.splat_kappa = np.clip(self.splat_kappa * 1.01, 0.1, 100.0)

        # Actualizar energy_api con splats relajados
        if self.splat_mu is not None:
            self.energy_api.update_splats(
                self.splat_mu, self.splat_alpha, self.splat_kappa
            )

        final_energy = self.energy_api.free_energy()

        return RelaxationResult(
            initial_energy=initial_energy,
            final_energy=final_energy,
            energy_delta=final_energy - initial_energy,
            iterations=iterations,
        )

    def get_statistics(self) -> dict:
        """Estadísticas del SOC engine."""
        report = self.check_criticality()
        return {
            "criticality_state": report.state.value,
            "criticality_index": report.index,
            "n_clusters": len(self._clusters),
            "n_avalanches": len(self.avalanche_history),
            "last_avalanche_clusters": (
                self.avalanche_history[-1].affected_clusters
                if self.avalanche_history
                else 0
            ),
            "equilibrium_state": self._compute_equilibrium_state(),
        }
