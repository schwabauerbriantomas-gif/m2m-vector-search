import numpy as np


class EnergyFunction:
    """Computes various energy potentials for the Gaussian Splats."""

    def __init__(self, config):
        self.config = config

    def E_splats(self, x, splats):
        return np.zeros(x.shape[0], dtype=np.float32)

    def E_geom(self, x):
        return np.zeros(x.shape[0], dtype=np.float32)

    def E_comp(self, x):
        return np.zeros(x.shape[0], dtype=np.float32)

    def __call__(self, x):
        # M2M config usually adds these up based on weights
        return self.E_splats(x, None) + self.E_geom(x) + self.E_comp(x)
