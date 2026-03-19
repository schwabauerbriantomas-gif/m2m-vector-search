import numpy as np


class EnergyFunction:
    """Computes energy potentials for the Gaussian Splat landscape.

    Energy model:
        E(x) = -log(Σᵢ αᵢ · exp(-κᵢ · ‖x - μᵢ‖²))

    Lower energy = higher confidence (near splat attractors).
    """

    def __init__(self, config):
        self.config = config

    def E_splats(self, x, splats):
        """
        Splat-based energy: negative log-density of the Gaussian mixture.

        E_splats(x) = -log(Σᵢ αᵢ · exp(-κᵢ · ‖x - μᵢ‖²))

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
        mu = splats.mu[:n]       # [N, D]
        alpha = splats.alpha[:n] # [N]
        kappa = splats.kappa[:n] # [N]

        # Vectorized over batch: for each query, compute energy
        energies = np.zeros(x.shape[0], dtype=np.float32)
        for i in range(x.shape[0]):
            diff = mu - x[i][np.newaxis, :]   # [N, D]
            dist_sq = np.sum(diff ** 2, axis=1)  # [N]
            contributions = alpha * np.exp(-kappa * dist_sq)  # [N]
            total = np.sum(contributions)
            energies[i] = -np.log(max(total, 1e-10))

        return energies

    def E_geom(self, x):
        """
        Geometric energy: penalizes deviation from the unit sphere.

        For vectors on S^{D-1}, ‖x‖ should be ≈ 1.
        E_geom(x) = (‖x‖ - 1)²

        Args:
            x: query vectors [B, D]

        Returns:
            energies: [B]
        """
        x = np.asarray(x, dtype=np.float32)
        if x.ndim == 1:
            x = x[np.newaxis, :]
        norms = np.linalg.norm(x, axis=1)
        return ((norms - 1.0) ** 2).astype(np.float32)

    def E_comp(self, x):
        """
        Compositional energy: placeholder for future compositional features.

        Currently disabled (energy_comp_weight = 0.0 in config).
        Returns zeros.
        """
        x = np.asarray(x, dtype=np.float32)
        if x.ndim == 1:
            return np.zeros(1, dtype=np.float32)
        return np.zeros(x.shape[0], dtype=np.float32)

    def __call__(self, x, splats=None):
        """Total energy: E_splats + E_geom + E_comp."""
        e_splats = self.E_splats(x, splats)
        e_geom = self.E_geom(x)
        e_comp = self.E_comp(x)
        return e_splats + 0.1 * e_geom + e_comp
