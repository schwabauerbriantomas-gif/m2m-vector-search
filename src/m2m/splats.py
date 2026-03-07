
import numpy as np
from typing import Tuple

from .hrm2_engine import HRM2Engine
from .splat_types import GaussianSplat

class SplatStore:
    """Wrapper around the CPU-optimized HRM2Engine to interface with NumPy."""

    def __init__(self, config):
        self.config = config

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
        self.mu = np.zeros((self.max_splats, config.latent_dim), dtype=np.float32)
        self.alpha = np.ones((self.max_splats,), dtype=np.float32) * config.init_alpha
        self.kappa = np.ones((self.max_splats,), dtype=np.float32) * config.init_kappa
        self.frequency = np.zeros((self.max_splats,), dtype=np.float32)
        
        # Internal splat counter for ID generation
        self._next_id = 0

        # GPUVectorIndex: lazy-init on first batch_find_neighbors when Vulkan enabled.
        # Rebuilds automatically when index changes (dirty flag).
        self._gpu_index = None
        self._gpu_index_dirty = True

    def add_splat(self, x: np.ndarray) -> bool:
        """Add a batch of splats or a single splat."""
        if x.ndim == 1:
            x = x[np.newaxis, :]
            
        n_new = x.shape[0]
        if self.n_active + n_new > self.max_splats:
            return False
            
        # Data is already numpy
        
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
        embeddings = self.mu[:self.n_active]
        self.engine.index(precomputed_embeddings=embeddings)
        # Mark GPU index dirty so it is rebuilt on next batch_find_neighbors call
        self._gpu_index_dirty = True
        
    def find_neighbors(self, query: np.ndarray, k: int = 64, lod: int = 2) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Find k-nearest neighbors using authentic HRM2 engine semantic routing."""
        query_np = query
        if query_np.ndim == 1:
            query_np = query_np.reshape(1, -1)
            
        batch_size = query_np.shape[0]
        dim = query_np.shape[1]
        
        k = min(k, max(1, self.n_active))
        
        if not self.engine._is_indexed:
            # Fallback to random if not indexed
            mu_out = np.random.randn(batch_size, k, dim).astype(np.float32)
            alpha_out = np.ones((batch_size, k), dtype=np.float32)
            kappa_out = np.ones((batch_size, k), dtype=np.float32) * 10.0
            return mu_out, alpha_out, kappa_out
            
        mu_out = np.zeros((batch_size, k, dim), dtype=np.float32)
        alpha_out = np.zeros((batch_size, k), dtype=np.float32)
        kappa_out = np.zeros((batch_size, k), dtype=np.float32)
        
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
        queries: np.ndarray,
        k: int = 64,
        lod: int = 2,
        max_batch_size: int = 100,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
        if queries.ndim == 1:
            queries = queries[np.newaxis, :]
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
                    index_vecs = self.mu[:self.n_active]
                    self._gpu_index = GPUVectorIndex(
                        index_vecs, max_batch_size=max_batch_size
                    )
                    self._gpu_index_dirty = False

                queries_np = queries.astype(np.float32)
                gpu_ids, gpu_dists = self._gpu_index.batch_search(queries_np, k=k)

                mu_out    = np.zeros((batch_size, k, dim), dtype=np.float32)
                alpha_out = np.zeros((batch_size, k), dtype=np.float32)
                kappa_out = np.zeros((batch_size, k), dtype=np.float32)

                for i in range(batch_size):
                    for j, idx in enumerate(gpu_ids[i]):
                        if idx < self.n_active:
                            mu_out[i, j]    = self.mu[idx]
                            alpha_out[i, j] = self.alpha[idx]
                            kappa_out[i, j] = self.kappa[idx]

                return mu_out, alpha_out, kappa_out

            except Exception as e:
                print(f"[SplatStore] GPU batch search failed ({e}), falling back to CPU.")

        # ── CPU fallback: batched vectorized query ───────────────────
        mu_out    = np.zeros((batch_size, k, dim), dtype=np.float32)
        alpha_out = np.zeros((batch_size, k), dtype=np.float32)
        kappa_out = np.zeros((batch_size, k), dtype=np.float32)

        batch_results = self.engine.query_batch(queries, k=k, lod=lod)
        for i, results in enumerate(batch_results):
            for j, (splat, dist) in enumerate(results):
                idx = splat.id
                mu_out[i, j]    = self.mu[idx]
                alpha_out[i, j] = self.alpha[idx]
                kappa_out[i, j] = self.kappa[idx]

        return mu_out, alpha_out, kappa_out

    def entropy(self, x=None):
        return 0.5
        
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
            self.mu[i] = np.array(s.mu, dtype=np.float32)
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
        embeddings = self.mu[:self.n_active]
        self.engine.index(precomputed_embeddings=embeddings)
        self._gpu_index_dirty = True
