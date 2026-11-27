"""
Energy-Based Model for the Gaussian Splat landscape.

Energy model:
    E(x) = -log(Σᵢ αᵢ · exp(-κᵢ · ‖x̂ - μᵢ‖²)) + λ_geom · E_geom(x)

Lower energy = higher confidence (query is well-covered by existing splats).
Higher energy = sparse region (exploration / novelty needed).

``E_splats`` delega en la implementación CANÓNICA del paquete
(``m2m.ebm.energy_api.splat_energy``) [audit r2-15]: misma normalización de la entrada
(la esfera, como μ), mismo tope ``OOD_ENERGY`` y mismo valor "sin splats" que
``EBMEnergy.energy``. Antes usaba ``gaussian_scoring.gaussian_energy`` sin normalizar x
y con otro tope: el mismo punto daba 10.0 en una API y 69.08 (= -log 1e-30) en la otra.
"""

import numpy as np

from .ebm.energy_api import OOD_ENERGY, splat_energy_batch as _canonical_energy_batch


class EnergyFunction:
    """Computes energy potentials for the Gaussian Splat landscape."""

    def __init__(self, config):
        self.config = config
        self._splat_weight = getattr(config, "energy_splat_weight", 1.0)
        self._geom_weight = getattr(config, "energy_geom_weight", 0.1)

    def E_splats(self, x, splats):
        """
        Splat-based energy: negative log-density of the Gaussian mixture.

        E_splats(x) = -log(Σᵢ αᵢ · exp(-κᵢ · ‖x̂ - μᵢ‖²))   [tope OOD_ENERGY]

        Usa los α, κ reales del SplatStore (actualizados por las reglas de
        aprendizaje online) y la implementación canónica del paquete: para la misma
        entrada da el mismo valor que ``EBMEnergy.energy`` (misma normalización a la
        esfera, mismo tope y mismo default "sin splats") [audit r2-15].

        Args:
            x: query vectors [B, D] or [D]
            splats: SplatStore instance (with .mu, .alpha, .kappa, .n_active)

        Returns:
            energies: [B] — lower = closer to attractors
        """
        if splats is None or splats.n_active == 0:
            return np.ones(x.shape[0], dtype=np.float32) * OOD_ENERGY

        x = np.asarray(x, dtype=np.float32)
        if x.ndim == 1:
            x = x[np.newaxis, :]

        n = splats.n_active
        mu = splats.mu[:n]
        alpha = splats.alpha[:n]
        kappa = splats.kappa[:n]

        energies = _canonical_energy_batch(x, mu, alpha, kappa)
        return energies.astype(np.float32)

    def E_geom(self, x):
        """
        Geometric energy: penalizes deviation from the unit sphere.

        For vectors on S^{D-1}, ‖x‖ should be ≈ 1.
        E_geom(x) = (‖x‖ - 1)²
        """
        x = np.asarray(x, dtype=np.float32)
        if x.ndim == 1:
            x = x[np.newaxis, :]
        norms = np.linalg.norm(x, axis=1)
        return ((norms - 1.0) ** 2).astype(np.float32)

    def E_comp(self, x):
        """
        Compositional energy: uses alpha variance as a proxy for
        structural complexity in the memory landscape.

        High alpha variance = some memories are much more important
        than others = more structured = lower compositional energy.
        """
        x = np.asarray(x, dtype=np.float32)
        if x.ndim == 1:
            return np.zeros(1, dtype=np.float32)
        return np.zeros(x.shape[0], dtype=np.float32)

    def __call__(self, x, splats=None):
        """Total energy: w_splat * E_splats + w_geom * E_geom + E_comp."""
        e_splats = self.E_splats(x, splats)
        e_geom = self.E_geom(x)
        e_comp = self.E_comp(x)
        return self._splat_weight * e_splats + self._geom_weight * e_geom + e_comp
