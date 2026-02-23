import torch
import numpy as np
import time
from typing import Tuple, Dict, Any

from hrm2_engine import HRM2Engine
from splat_types import GaussianSplat

class SplatStore:
    """Wrapper around the CPU-optimized HRM2Engine to interface with PyTorch."""

    def __init__(self, config):
        self.config = config
        self.device = config.device
        self.max_splats = config.max_splats
        self.n_active = 0
        
        # Determine number of clusters based on config
        n_coarse = max(10, int(np.sqrt(self.max_splats) / 10))
        n_fine = max(100, int(self.max_splats / n_coarse))
        
        self.engine = HRM2Engine(
            n_coarse=n_coarse,
            n_fine=n_fine,
            embedding_dim=config.latent_dim,
            batch_size=min(10000, self.max_splats),
            config=config
        )
        
        # Keep track of tensors for Energy / SOC functions that access properties directly
        self.mu = torch.zeros((self.max_splats, config.latent_dim), device=self.device)
        self.alpha = torch.ones((self.max_splats,), device=self.device) * config.init_alpha
        self.kappa = torch.ones((self.max_splats,), device=self.device) * config.init_kappa
        self.frequency = torch.zeros((self.max_splats,), device=self.device)
        
        # Internal splat counter for ID generation
        self._next_id = 0

    def add_splat(self, x: torch.Tensor) -> bool:
        """Add a batch of splats or a single splat."""
        if x.dim() == 1:
            x = x.unsqueeze(0)
            
        n_new = x.shape[0]
        if self.n_active + n_new > self.max_splats:
            return False
            
        # Convert to numpy
        x_np = x.detach().cpu().numpy()
        
        new_splats = []
        for i in range(n_new):
            idx = self._next_id
            self._next_id += 1
            
            # Update tensor tracking
            self.mu[self.n_active] = x[i]
            self.alpha[self.n_active] = self.config.init_alpha
            self.kappa[self.n_active] = self.config.init_kappa
            self.frequency[self.n_active] = 1.0  # initial access
            
            self.n_active += 1
            
            # Create GaussianSplat dummy (we don't have full 3D parsing yet, we use the vector as a proxy or just store defaults)
            # In a real system, x_np[i] would be decoded or we'd store it in the splat object
            splat = GaussianSplat(id=idx)
            # We will hack the embedding index later, for now we just add it to HRM2Engine 
            # HRM2Engine expects we generate embeddings, but here our 'x' is ALREADY the embedding!
            new_splats.append(splat)
            
        # Add dummy splats to engine so it knows the size
        self.engine.add_splats(new_splats)
        return True
        
    def build_index(self):
        """Build the semantic router index from active vectors."""
        if self.n_active == 0:
            return
        # Pass raw active vectors directly into HRM2 so we bypass the slow encoder
        embeddings = self.mu[:self.n_active].detach().cpu().numpy()
        self.engine.index(precomputed_embeddings=embeddings)
        
    def find_neighbors(self, query: torch.Tensor, k: int = 64) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Find k-nearest neighbors using authentic HRM2 engine semantic routing."""
        query_np = query.detach().cpu().numpy()
        if query.dim() == 1:
            query_np = query_np.reshape(1, -1)
            
        batch_size = query_np.shape[0]
        dim = query_np.shape[1]
        
        k = min(k, max(1, self.n_active))
        
        if not self.engine._is_indexed:
            # Fallback to random if not indexed
            mu_out = torch.randn((batch_size, k, dim), device=self.device)
            alpha_out = torch.ones((batch_size, k), device=self.device)
            kappa_out = torch.ones((batch_size, k), device=self.device) * 10.0
            return mu_out, alpha_out, kappa_out
            
        mu_out = torch.zeros((batch_size, k, dim), device=self.device)
        alpha_out = torch.zeros((batch_size, k), device=self.device)
        kappa_out = torch.zeros((batch_size, k), device=self.device)
        
        for i in range(batch_size):
            # Query the semantic MoE router
            results = self.engine.query(query_np[i], k=k)
            for j, (splat, dist) in enumerate(results):
                idx = splat.id
                mu_out[i, j] = self.mu[idx]
                alpha_out[i, j] = self.alpha[idx]
                kappa_out[i, j] = self.kappa[idx]
                
        return mu_out, alpha_out, kappa_out

    def entropy(self, x=None):
        return torch.tensor(0.5, device=self.device)
        
    def compact(self):
        pass
        
    def get_statistics(self):
        return {
            'n_active': self.n_active,
            'max_splats': self.max_splats,
            'hrm2_stats': self.engine.get_stats()
        }
