"""
EBMEnergy - API de Energía para M2M EBM Database.

La energía de un punto en el espacio de splats representa qué tan "natural"
es esa posición en el paisaje energético aprendido:

  E(x) = -log(Σᵢ αᵢ · exp(-κᵢ · ‖x - μᵢ‖²))

- Energía baja (0–0.3):  Alta confianza, región bien conocida
- Energía media (0.3–0.7): Confianza moderada
- Energía alta (>0.7):   Baja confianza, región incierta o inexplorada
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class EnergyResult:
    """Resultado de cálculo de energía."""

    vector: np.ndarray
    energy: float
    confidence: float  # 1 / (1 + energy)
    zone: str  # 'high_confidence', 'moderate', 'uncertain'


class EBMEnergy:
    """
    API de Energía del sistema EBM.

    Expone el paisaje energético aprendido por los Gaussian Splats,
    permitiendo a agentes/LLMs conocer la confianza en cada resultado.

    Uso:
        ebm = EBMEnergy(splats)
        e = ebm.energy(query_vector)
        grad = ebm.energy_gradient(query_vector)
        map_2d = ebm.local_energy_map(center, radius=1.0)
    """

    def __init__(
        self,
        splat_mu: Optional[np.ndarray] = None,
        splat_alpha: Optional[np.ndarray] = None,
        splat_kappa: Optional[np.ndarray] = None,
    ):
        """
        Args:
            splat_mu: Centroides de splats [N, D]
            splat_alpha: Amplitudes [N]
            splat_kappa: Concentraciones [N]
        """
        self.splat_mu = splat_mu
        self.splat_alpha = splat_alpha
        self.splat_kappa = splat_kappa

    def update_splats(
        self,
        splat_mu: np.ndarray,
        splat_alpha: np.ndarray,
        splat_kappa: np.ndarray,
    ):
        """Actualiza los splats del paisaje energético."""
        self.splat_mu = np.asarray(splat_mu, dtype=np.float32)
        self.splat_alpha = np.asarray(splat_alpha, dtype=np.float32)
        self.splat_kappa = np.asarray(splat_kappa, dtype=np.float32)

    def _has_splats(self) -> bool:
        return (
            self.splat_mu is not None
            and len(self.splat_mu) > 0
        )

    def energy(self, vector: np.ndarray) -> float:
        """
        Calcula la energía E(x) del vector en el paisaje energético.

        E(x) = -log(Σᵢ αᵢ · exp(-κᵢ · ‖x - μᵢ‖²))

        Returns:
            Valor de energía escalar. Menor = mayor confianza.
        """
        if not self._has_splats():
            return 1.0  # Sin splats, energía máxima por defecto

        x = np.asarray(vector, dtype=np.float32).flatten()
        mu = self.splat_mu
        alpha = self.splat_alpha
        kappa = self.splat_kappa

        # Vectorizado: diferencias al cuadrado
        diff = mu - x[np.newaxis, :]  # [N, D]
        dist_sq = np.sum(diff ** 2, axis=1)  # [N]

        # Contribuciones de cada splat
        contributions = alpha * np.exp(-kappa * dist_sq)  # [N]
        total = np.sum(contributions)

        if total < 1e-10:
            return 10.0  # Muy fuera de distribución

        return float(-np.log(total))

    def energy_batch(self, vectors: np.ndarray) -> np.ndarray:
        """
        Calcula energía para múltiples vectores (batch vectorizado).

        Args:
            vectors: [M, D]

        Returns:
            energies: [M]
        """
        if not self._has_splats():
            return np.ones(len(vectors), dtype=np.float32)

        X = np.asarray(vectors, dtype=np.float32)
        mu = self.splat_mu  # [N, D]
        alpha = self.splat_alpha  # [N]
        kappa = self.splat_kappa  # [N]

        # [M, N, D] -> [M, N]
        energies = []
        for x in X:
            diff = mu - x[np.newaxis, :]
            dist_sq = np.sum(diff ** 2, axis=1)
            contribs = alpha * np.exp(-kappa * dist_sq)
            total = np.sum(contribs)
            e = -np.log(total) if total > 1e-10 else 10.0
            energies.append(e)

        return np.array(energies, dtype=np.float32)

    def energy_gradient(self, vector: np.ndarray) -> np.ndarray:
        """
        Calcula el gradiente de energía ∇E(x).

        El gradiente apunta hacia la dirección de mayor ascenso energético.
        Su negativo (descenso de gradiente) lleva a regiones de menor energía.

        Útil para:
        - Encontrar documentos similares de forma iterativa
        - Navegar el espacio de manera dirigida

        Returns:
            Gradiente [D] del mismo tamaño que el vector de entrada.
        """
        if not self._has_splats():
            return np.zeros_like(vector, dtype=np.float32)

        x = np.asarray(vector, dtype=np.float32).flatten()
        mu = self.splat_mu
        alpha = self.splat_alpha
        kappa = self.splat_kappa

        diff = x[np.newaxis, :] - mu  # [N, D]
        dist_sq = np.sum(diff ** 2, axis=1)  # [N]
        exp_terms = np.exp(-kappa * dist_sq)  # [N]
        factors = 2 * kappa * alpha * exp_terms  # [N]

        # Gradiente: Σᵢ 2κᵢαᵢexp(...)(x - μᵢ)
        gradient = np.sum(factors[:, np.newaxis] * diff, axis=0)  # [D]

        # Denominator para normalizar
        total = np.sum(alpha * exp_terms)
        if total > 1e-10:
            gradient = gradient / total

        return gradient.astype(np.float32)

    def free_energy(self) -> float:
        """
        Energía libre del sistema: F = -log(Z)

        Z = Σᵢ αᵢ (función de partición)

        Útil para:
        - Comparar estados del sistema antes/después de reorganización
        - Detectar necesidad de reorganización (SOC)

        Returns:
            Energía libre escalar.
        """
        if not self._has_splats():
            return float("inf")

        Z = float(np.sum(self.splat_alpha))
        return -np.log(Z) if Z > 0 else float("inf")

    def local_energy_map(
        self,
        center: np.ndarray,
        radius: float = 1.0,
        resolution: int = 20,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Genera mapa de energía 2D alrededor de un punto central.

        Útil para visualización y debugging del paisaje energético.

        Args:
            center: Vector central [D]
            radius: Radio de exploración
            resolution: Resolución del mapa (resolution x resolution)

        Returns:
            Tuple (X_grid, Y_grid, energy_map) para plt.contourf()
        """
        center = np.asarray(center, dtype=np.float32)
        D = len(center)

        if D >= 2:
            # Proyección en los dos primeros ejes
            basis_0 = np.zeros(D, dtype=np.float32)
            basis_0[0] = 1.0
            basis_1 = np.zeros(D, dtype=np.float32)
            basis_1[1] = 1.0
        else:
            basis_0 = np.array([1.0], dtype=np.float32)
            basis_1 = np.array([1.0], dtype=np.float32)

        x_vals = np.linspace(-radius, radius, resolution)
        y_vals = np.linspace(-radius, radius, resolution)
        X, Y = np.meshgrid(x_vals, y_vals)

        energy_map = np.zeros((resolution, resolution), dtype=np.float32)
        for i in range(resolution):
            for j in range(resolution):
                point = center.copy()
                if D >= 2:
                    point[0] = center[0] + X[i, j]
                    point[1] = center[1] + Y[i, j]
                energy_map[i, j] = self.energy(point)

        return X, Y, energy_map

    def classify_energy(self, energy_value: float) -> str:
        """Clasifica un valor de energía en zona de confianza."""
        if energy_value < 0.3:
            return "high_confidence"
        elif energy_value < 0.7:
            return "moderate"
        else:
            return "uncertain"

    def get_result(self, vector: np.ndarray) -> EnergyResult:
        """Retorna resultado completo de energía para un vector."""
        e = self.energy(vector)
        confidence = 1.0 / (1.0 + e)
        zone = self.classify_energy(e)
        return EnergyResult(
            vector=np.asarray(vector, dtype=np.float32),
            energy=e,
            confidence=confidence,
            zone=zone,
        )
