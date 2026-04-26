"""
Energy-Based Model for the Gaussian Splat landscape.

Energy model:
    E(x) = -log(Σᵢ αᵢ · exp(-κᵢ · ‖x - μᵢ‖²)) + λ_geom · E_geom(x)

Lower energy = higher confidence (query is well-covered by existing splats).
Higher energy = sparse region (exploration / novelty needed).

Now uses the centralized gaussian_scoring module for consistency.
"""

import numpy as np

from .gaussian_scoring import gaussian_energy as _compute_energy


class EnergyFunction:
    """Computes energy potentials for the Gaussian Splat landscape."""

    def __init__(self, config):
        self.config = config
        self._splat_weight = getattr(config, "energy_splat_weight", 1.0)
        self._geom_weight = getattr(config, "energy_geom_weight", 0.1)

    def E_splats(self, x, splats):
        """
        Splat-based energy: negative log-density of the Gaussian mixture.

        E_splats(x) = -log(Σᵢ αᵢ · exp(-κᵢ · ‖x - μᵢ‖²))

        Uses the actual α, κ values from SplatStore (which are now updated
        by the online learning rules).

        Args:
            x: query vectors [B, D] or [D]
            splats: SplatStore instance (with .mu, .alpha, .kappa, .n_active)

        Returns:
            energies: [B] — lower = closer to attractors
        """
        if splats is None or splats.n_active == 0:
            return np.ones(x.shape[0], dtype=np.float32) * 10.0

        x = np.asarray(x, dtype=np.float32)
        if x.ndim == 1:
            x = x[np.newaxis, :]

        n = splats.n_active
        mu = splats.mu[:n]
        alpha = splats.alpha[:n]
        kappa = splats.kappa[:n]

        energies = np.zeros(x.shape[0], dtype=np.float32)
        for i in range(x.shape[0]):
            energies[i] = _compute_energy(x[i], mu, alpha, kappa)

        return energies

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
