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
        self.device = config.torch_device
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

        # GPUVectorIndex: lazy-init on first batch_find_neighbors when Vulkan enabled.
        # Rebuilds automatically when index changes (dirty flag).
        self._gpu_index = None
        self._gpu_index_dirty = True

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
        # Mark GPU index dirty so it is rebuilt on next batch_find_neighbors call
        self._gpu_index_dirty = True
        
    def find_neighbors(self, query: torch.Tensor, k: int = 64, lod: int = 2) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
            results = self.engine.query(query_np[i], k=k, lod=lod)
            for j, (splat, dist) in enumerate(results):
                idx = splat.id
                mu_out[i, j] = self.mu[idx]
                alpha_out[i, j] = self.alpha[idx]
                kappa_out[i, j] = self.kappa[idx]
                
        return mu_out, alpha_out, kappa_out

    def batch_find_neighbors(
        self,
        queries: torch.Tensor,
        k: int = 64,
        lod: int = 2,
        max_batch_size: int = 100,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Batch k-NN search — uses GPUVectorIndex (persistent index, single dispatch)
        when Vulkan is enabled, falls back to sequential find_neighbors() on CPU.

        ✅ CORRECT pattern (reference implementation):
          - Index uploaded to GPU ONCE (or when dirty after rebuild)
          - Only queries (small) are transferred per call
          - All B queries dispatched in one vkCmdDispatch(ceil(N/256), B, 1)

        Args:
            queries: [B, D] tensor
            k:       number of neighbours
            lod:     level of detail (CPU path only)
            max_batch_size: max queries per GPU dispatch

        Returns:
            mu_out    [B, k, D]
            alpha_out [B, k]
            kappa_out [B, k]
        """
        if queries.dim() == 1:
            queries = queries.unsqueeze(0)
        batch_size = queries.shape[0]
        dim = queries.shape[1]
        k = min(k, max(1, self.n_active))

        # ── GPU path: GPUVectorIndex ──────────────────────────────────
        vulkan_enabled = getattr(self.config, 'enable_vulkan', False)
        if vulkan_enabled and self.n_active > 0:
            try:
                # Lazy init / rebuild when index vectors changed
                if self._gpu_index is None or self._gpu_index_dirty:
                    from gpu_vector_index import GPUVectorIndex
                    index_vecs = self.mu[:self.n_active].detach().cpu().numpy()
                    self._gpu_index = GPUVectorIndex(
                        index_vecs, max_batch_size=max_batch_size
                    )
                    self._gpu_index_dirty = False

                queries_np = queries.detach().cpu().numpy().astype(np.float32)
                gpu_ids, gpu_dists = self._gpu_index.batch_search(queries_np, k=k)

                mu_out    = torch.zeros((batch_size, k, dim), device=self.device)
                alpha_out = torch.zeros((batch_size, k),      device=self.device)
                kappa_out = torch.zeros((batch_size, k),      device=self.device)

                for i in range(batch_size):
                    for j, idx in enumerate(gpu_ids[i]):
                        if idx < self.n_active:
                            mu_out[i, j]    = self.mu[idx]
                            alpha_out[i, j] = self.alpha[idx]
                            kappa_out[i, j] = self.kappa[idx]

                return mu_out, alpha_out, kappa_out

            except Exception as e:
                print(f"[SplatStore] GPU batch search failed ({e}), falling back to CPU.")

        # ── CPU fallback: sequential find_neighbors ───────────────────
        mu_out    = torch.zeros((batch_size, k, dim), device=self.device)
        alpha_out = torch.zeros((batch_size, k),      device=self.device)
        kappa_out = torch.zeros((batch_size, k),      device=self.device)

        for i in range(batch_size):
            mu_i, a_i, k_i = self.find_neighbors(queries[i:i+1], k=k, lod=lod)
            mu_out[i]    = mu_i[0]
            alpha_out[i] = a_i[0]
            kappa_out[i] = k_i[0]

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

    def _build_hrm2_from_splats(self, splats):
        """Construye índice HRM2 desde splats pre-computados."""
        from splat_types import GaussianSplat as HrmSplat
        
        n_new = len(splats)
        if n_new > self.max_splats:
            raise ValueError(f"Too many splats to load ({n_new} > {self.max_splats})")
            
        new_splats = []
        for i, s in enumerate(splats):
            self.mu[i] = torch.tensor(s.mu, device=self.device)
            self.alpha[i] = s.alpha
            self.kappa[i] = s.kappa
            self.frequency[i] = 1.0
            
            splat_obj = HrmSplat(id=i)
            new_splats.append(splat_obj)
            
        self.n_active = n_new
        self._next_id = n_new
        
        # Clear and rebuild engine
        self.engine.clear()
        self.engine.add_splats(new_splats)
        
        # Bypass encoder
        embeddings = self.mu[:self.n_active].detach().cpu().numpy()
        self.engine.index(precomputed_embeddings=embeddings)
        self._gpu_index_dirty = True
