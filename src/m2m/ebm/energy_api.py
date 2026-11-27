"""
EBMEnergy - API de Energía para M2M EBM Database.

La energía de un punto en el espacio de splats representa qué tan "natural"
es esa posición en el paisaje energético aprendido:

  E(x) = -log(Σᵢ αᵢ · exp(-κᵢ · ‖x̂ - μᵢ‖²))

``x`` se normaliza a la esfera (la MISMA normalización que ``add_splats`` aplica a μ)
y el resultado tiene un tope superior ``OOD_ENERGY`` (punto fuera de distribución).

- Energía baja (0–0.3):  Alta confianza, región bien conocida
- Energía media (0.3–0.7): Confianza moderada
- Energía alta (>0.7):   Baja confianza, región incierta o inexplorada

``splat_energy`` es la implementación CANÓNICA [audit r2-15]: el resto del paquete
(``m2m.energy.EnergyFunction.E_splats`` y los ``DocResult`` de ``SimpleVectorDB.search``)
delega en ella, así a la misma entrada le corresponde el mismo valor en todas las APIs
(antes: 10.0 acá vs 69.08 = -log(1e-30) en E_splats para el mismo punto OOD, y 1.0 vs
10.0 sin splats). ``confidence_from_energy`` mapea E → confianza en (0, 1)
[audit r2-15, r3a-02].
"""

from dataclasses import dataclass
import math
from typing import Optional, Tuple

import numpy as np

# Energía que se devuelve para un punto fuera de distribución (densidad < e^-10 bajo
# los splats considerados). ÚNICO valor de tope y de "sin splats" del paquete.
OOD_ENERGY = 10.0

# Mismo epsilon que m2m.normalize_sphere (add_splats normaliza μ con esta fórmula).
_EPS = 1e-8


def _normalize_sphere(x: np.ndarray) -> np.ndarray:
    """x / (‖x‖ + 1e-8): la MISMA normalización que ``add_splats`` aplica a μ."""
    return x / (np.linalg.norm(x) + _EPS)


def _splat_arrays(splat_mu, splat_alpha, splat_kappa, dim: int):
    mu = np.asarray(splat_mu, dtype=np.float64).reshape(-1, dim)
    alpha = np.asarray(splat_alpha, dtype=np.float64).reshape(-1)
    kappa = np.asarray(splat_kappa, dtype=np.float64).reshape(-1)
    return mu, alpha, kappa


def _neg_log_alpha(alpha: np.ndarray) -> np.ndarray:
    """-log αᵢ por splat; +inf para αᵢ ≤ 0 (no aporta a la densidad)."""
    safe = np.where(alpha > 0, alpha, 1.0)
    return np.where(alpha > 0, -np.log(safe), np.inf)


def splat_energies(
    vector: np.ndarray,
    splat_mu: Optional[np.ndarray],
    splat_alpha: Optional[np.ndarray],
    splat_kappa: Optional[np.ndarray],
) -> np.ndarray:
    """
    Energía de ``vector`` bajo CADA splat por separado (sin tope): [N]

        Eᵢ(x) = -log αᵢ + κᵢ · ‖x̂ − μᵢ‖²        (x̂ = x normalizado a la esfera)

    Σᵢ exp(−Eᵢ) es la densidad de la mezcla en x. Devuelve +inf para splats con αᵢ ≤ 0.
    """
    x = _normalize_sphere(np.asarray(vector, dtype=np.float64).reshape(-1))
    mu, alpha, kappa = _splat_arrays(splat_mu, splat_alpha, splat_kappa, x.shape[0])
    diff = mu - x[np.newaxis, :]  # [N, D]
    d2 = np.einsum("ij,ij->i", diff, diff)  # [N]
    return _neg_log_alpha(alpha) + kappa * d2


def splat_energy(
    vector: np.ndarray,
    splat_mu: Optional[np.ndarray],
    splat_alpha: Optional[np.ndarray],
    splat_kappa: Optional[np.ndarray],
) -> float:
    """
    Energía CANÓNICA del paisaje en ``vector``:

        E(x) = -log Σᵢ αᵢ · exp(-κᵢ · ‖x̂ − μᵢ‖²) = -log Σᵢ exp(−Eᵢ(x)),  tope OOD_ENERGY

    - ``vector`` se normaliza a la esfera como μ (misma normalización en todas las APIs).
    - Sin splats (None, vacío o todos con α ≤ 0) → ``OOD_ENERGY``: sin modelo aprendido
      no hay cobertura en ningún punto.
    - Se calcula en log-space (log-sum-exp, float64): sin underflow, y el tope es
      exactamente ``OOD_ENERGY``.

    Returns:
        Energía escalar. Menor = mayor confianza.
    """
    if splat_mu is None or len(splat_mu) == 0:
        return OOD_ENERGY
    e = splat_energies(vector, splat_mu, splat_alpha, splat_kappa)
    m = float(np.min(e))
    if not np.isfinite(m):  # todos los splats con α <= 0: sin cobertura
        return OOD_ENERGY
    total = m - float(np.log(np.sum(np.exp(-(e - m)))))  # = -logsumexp(-e)
    return float(min(total, OOD_ENERGY))


def splat_energy_batch(
    vectors: np.ndarray,
    splat_mu: Optional[np.ndarray],
    splat_alpha: Optional[np.ndarray],
    splat_kappa: Optional[np.ndarray],
) -> np.ndarray:
    """``splat_energy`` para M vectores [M, D] → [M] (mismo valor que uno por uno)."""
    X = np.asarray(vectors, dtype=np.float64)
    if X.ndim == 1:
        X = X[np.newaxis, :]
    if splat_mu is None or len(splat_mu) == 0:
        return np.full(len(X), OOD_ENERGY)
    Xn = _normalize_sphere(X)
    mu, alpha, kappa = _splat_arrays(splat_mu, splat_alpha, splat_kappa, Xn.shape[1])
    # Gram: ‖x−μ‖² = ‖x‖² + ‖μ‖² − 2·x·μ  → [M, N]
    d2 = (Xn**2).sum(axis=1)[:, None] + (mu**2).sum(axis=1)[None, :] - 2.0 * (Xn @ mu.T)
    np.maximum(d2, 0.0, out=d2)
    e = _neg_log_alpha(alpha)[None, :] + kappa[None, :] * d2  # [M, N]
    m = np.min(e, axis=1)  # [M]
    finite_row = np.isfinite(m)
    with np.errstate(invalid="ignore"):  # filas sin ningún splat válido: inf - inf
        z = -(e - m[:, None])
    z = np.where(np.isfinite(e), z, -np.inf)
    with np.errstate(invalid="ignore"):
        lse = m - np.log(np.exp(z).sum(axis=1))
    out = np.where(finite_row, lse, OOD_ENERGY)
    return np.minimum(out, OOD_ENERGY)


def confidence_from_energy(energy: float) -> float:
    """
    Confianza en (0, 1) a partir de la energía: ``sigmoid(−E) = 1 / (1 + e^E)``.

    Por qué esta forma [audit r2-15, r3a-02]:

    - E es un log-densidad negativo en nats (adimensional): exp(−E) =
      Σᵢ αᵢ·e^(−κᵢ‖x̂−μᵢ‖²) es la densidad de la mezcla en el punto, y
      sigmoid(−E) = densidad / (1 + densidad) es la probabilidad "suavizada" de que el
      punto esté cubierto por el modelo. No hay que elegir una escala de referencia.
    - 0.5 es el centro exacto de un splat de amplitud α = 1 (``init_alpha``): > 0.5 con
      α > 1 o con cobertura de splats vecinos; cae con κᵢ·‖x̂−μᵢ‖² al alejarse
      (κ = 10: ≈ 0.27 a 0.3 de distancia cuerda, ≈ 1e-3 a 1 radián).
    - Monotona decreciente en E y acotada a (0, 1): nunca negativa ni infinita, a
      diferencia de 1/(1+E), que daba negativa para E < −1 y división por cero en
      E = −1, justo donde el modelo está MÁS seguro (centros de splat con α > 1).
    - No depende de k ni de un conjunto de referencia: es una propiedad del punto.
    - Límite consciente: al no restar log Z la densidad no está normalizada por el
      volumen, así que re-escalar todos los α desplaza la confianza. No se usa la
      energía normalizada E + log Z porque su término (D/2)·log(κ/π) ≈ 74 nats en
      D = 128 con κ = 10 deja sigmoid(−E_norm) ≈ 0 hasta en el centro de un splat.
    """
    e = float(energy)
    if e >= 0.0:  # estable para E grande
        z = math.exp(-e)
        return z / (1.0 + z)
    z = math.exp(e)
    return 1.0 / (1.0 + z)




@dataclass
class EnergyResult:
    """Resultado de cálculo de energía."""

    vector: np.ndarray
    energy: float
    confidence: float  # sigmoid(-E) ∈ (0, 1) — ver confidence_from_energy
    zone: str  # 'high_confidence', 'moderate', 'uncertain'


class EBMEnergy:
    """
    API de Energía del sistema EBM.

    Expone el paisaje energético aprendido por los Gaussian Splats,
    permitiendo a agentes/LLMs conocer la confianza en cada resultado.

    Toda la clase delega en la implementación canónica del módulo
    (``splat_energy`` / ``splat_energy_batch``): misma normalización de la entrada
    (esfera), mismo tope ``OOD_ENERGY`` y mismo valor "sin splats" que
    ``m2m.energy.EnergyFunction.E_splats`` [audit r2-15].

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
        return self.splat_mu is not None and len(self.splat_mu) > 0

    def energy(self, vector: np.ndarray) -> float:
        """
        Calcula la energía E(x) del vector en el paisaje energético.

        E(x) = -log(Σᵢ αᵢ · exp(-κᵢ · ‖x̂ - μᵢ‖²)),  x̂ = x normalizado a la esfera

        Implementación canónica (``splat_energy``): mismo valor que
        ``EnergyFunction.E_splats`` para la misma entrada. Sin splats → ``OOD_ENERGY``;
        punto fuera de distribución → ``OOD_ENERGY`` (10.0).

        Returns:
            Valor de energía escalar. Menor = mayor confianza.
        """
        return splat_energy(vector, self.splat_mu, self.splat_alpha, self.splat_kappa)

    def energy_per_splat(self, vector: np.ndarray) -> np.ndarray:
        """
        Energía de ``vector`` bajo cada splat por separado (tope ``OOD_ENERGY``): [N].

        Es la energía que lleva cada ``DocResult`` de ``SimpleVectorDB.search(
        include_energy=True)``: la de la query normalizada en la cuenca del splat de
        ESE resultado, Eᵢ = -log αᵢ + κᵢ·‖x̂−μᵢ‖². Así resultados distintos tienen
        energías/confianzas distintas [audit r3a-02] (antes se evaluaba el paisaje en la
        posición del propio documento, donde su splat aporta α·e⁰ y E ≈ -log αᵢ para
        todos). Relación exacta con el total: ``energy(x) = -log Σᵢ exp(−Eᵢ(x)) ≤ minᵢ Eᵢ``.

        Sin splats devuelve un array vacío.
        """
        if not self._has_splats():
            return np.zeros(0, dtype=np.float64)
        e = splat_energies(vector, self.splat_mu, self.splat_alpha, self.splat_kappa)
        return np.minimum(e, OOD_ENERGY)

    def energy_batch(self, vectors: np.ndarray) -> np.ndarray:
        """
        Calcula energía para múltiples vectores (batch vectorizado).

        Args:
            vectors: [M, D]

        Returns:
            energies: [M] float32 — mismo valor que ``energy`` uno por uno.
        """
        X = np.asarray(vectors, dtype=np.float64)
        if X.ndim == 1:
            X = X[np.newaxis, :]
        out = splat_energy_batch(X, self.splat_mu, self.splat_alpha, self.splat_kappa)
        return out.astype(np.float32)

    def energy_gradient(self, vector: np.ndarray) -> np.ndarray:
        """
        Calcula el gradiente de energía ∇E(x).

        El gradiente apunta hacia la dirección de mayor ascenso energético.
        Su negativo (descenso de gradiente) lleva a regiones de menor energía.

        Se evalúa en el MISMO punto que ``energy``: ``x`` normalizado a la esfera (el
        gradiente respecto del punto sobre la esfera; la proyección al tangente, si el
        llamador da pasos sobre la esfera, queda de su lado).

        Útil para:
        - Encontrar documentos similares de forma iterativa
        - Navegar el espacio de manera dirigida

        Returns:
            Gradiente [D] del mismo tamaño que el vector de entrada.
        """
        if not self._has_splats():
            return np.zeros_like(vector, dtype=np.float32)

        x = _normalize_sphere(np.asarray(vector, dtype=np.float32).flatten())
        mu = self.splat_mu
        alpha = self.splat_alpha
        kappa = self.splat_kappa

        diff = x[np.newaxis, :] - mu  # [N, D]
        dist_sq = np.sum(diff**2, axis=1)  # [N]
        exp_terms = np.exp(-kappa * dist_sq)  # [N]
        factors = 2 * kappa * alpha * exp_terms  # [N]

        # Gradiente: Σᵢ 2κᵢαᵢexp(...)(x - μᵢ)
        gradient = np.sum(factors[:, np.newaxis] * diff, axis=0)  # [D]

        # Denominator para normalizar
        total = np.sum(alpha * exp_terms)
        if total > 1e-10:
            gradient = gradient / total

        return gradient.astype(np.float32)

    def confidence(self, vector: np.ndarray) -> float:
        """Confianza en (0, 1) del vector: ``confidence_from_energy(self.energy(x))``."""
        return confidence_from_energy(self.energy(vector))

    def free_energy(self) -> float:
        """
        Energía libre del sistema: F = -log(Z)

        Z = Σᵢ αᵢ (π/κᵢ)^(D/2)  — integral de la mezcla gaussiana
            αᵢ·exp(−κᵢ‖x−μᵢ‖²) sobre ℝᴰ (integral gaussiana estándar).

        La fórmula anterior (Z = Σ αᵢ) omitía el factor (π/κ)^(D/2): con D alto
        el término omitido vale decenas de nats y F quedaba ciega a κ (dos
        sistemas con κ distintos pero igual Σα daban idéntico F). Se computa en
        log-space: F = −logsumexp(log αᵢ + (D/2)(log π − log κᵢ)).

        Útil para:
        - Comparar estados del sistema antes/después de reorganización
        - Detectar necesidad de reorganización (SOC)

        Returns:
            Energía libre escalar.
        """
        if not self._has_splats():
            return float("inf")

        alpha = np.asarray(self.splat_alpha, dtype=np.float64)
        kappa = np.asarray(self.splat_kappa, dtype=np.float64)
        if self.splat_mu is None or self.splat_mu.ndim != 2:
            return float("inf")
        d = int(self.splat_mu.shape[1])

        pos = (alpha > 0) & (kappa > 0) & np.isfinite(alpha) & np.isfinite(kappa)
        if not np.any(pos):
            return float("inf")

        log_terms = (
            np.log(alpha[pos]) + 0.5 * d * (np.log(np.pi) - np.log(kappa[pos]))
        )
        m = float(np.max(log_terms))
        if not np.isfinite(m):
            return float("inf")
        Z = m + float(np.log(np.sum(np.exp(log_terms - m))))
        return -Z

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
        confidence = confidence_from_energy(e)
        zone = self.classify_energy(e)
        return EnergyResult(
            vector=np.asarray(vector, dtype=np.float32),
            energy=e,
            confidence=confidence,
            zone=zone,
        )
