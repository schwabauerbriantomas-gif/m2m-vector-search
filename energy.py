import torch

class EnergyFunction:
    """Computes various energy potentials for the Gaussian Splats."""
    def __init__(self, config):
        self.config = config
        
    def E_splats(self, x, splats):
        return torch.zeros(x.shape[0], device=self.config.device)
        
    def E_geom(self, x):
        return torch.zeros(x.shape[0], device=self.config.device)
        
    def E_comp(self, x):
        return torch.zeros(x.shape[0], device=self.config.device)
        
    def __call__(self, x):
        # M2M config usually adds these up based on weights
        return self.E_splats(x, None) + self.E_geom(x) + self.E_comp(x)
