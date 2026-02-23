"""
HRM2 Engine - Hierarchical Retrieval Model 2

Implements a two-level hierarchical index for fast similarity search
in large-scale Gaussian splat datasets.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass, field
import time

from splat_types import GaussianSplat, SplatEmbedding, SplatCluster
from encoding import FullEmbeddingBuilder
from clustering import KMeans, assign_clusters
from engine import M2MEngine


@dataclass
class SearchResult:
    """Result of a similarity search."""
    splat_id: int
    distance: float
    coarse_cluster: int
    fine_cluster: int


@dataclass
class HRM2Config:
    """Configuration for HRM2 Engine."""
    n_coarse: int = 100          # Number of coarse clusters
    n_fine: int = 1000           # Number of fine clusters per coarse
    embedding_dim: int = 640     # Embedding dimension
    n_probe: int = 5             # Clusters to probe during search
    batch_size: int = 10000      # Batch size for K-Means
    random_state: int = 42


@dataclass
class HRM2Stats:
    """Statistics for HRM2 Engine."""
    n_splats: int = 0
    n_coarse_clusters: int = 0
    n_fine_clusters: int = 0
    build_time: float = 0.0
    avg_query_time: float = 0.0
    total_queries: int = 0


class HRM2Engine:
    """
    Hierarchical Retrieval Model 2 (HRM2) Engine.
    
    Implements a two-level hierarchical index:
    - Level 1 (Coarse): K-Means clusters for fast pruning
    - Level 2 (Fine): Additional clustering within each coarse cluster
    
    This provides significant speedup over brute-force search while
    maintaining high recall.
    
    Example:
        >>> engine = HRM2Engine(n_coarse=100, n_fine=1000)
        >>> engine.add_splats(splats)
        >>> engine.index()
        >>> results = engine.query(query_vector, k=10)
    """
    
    def __init__(
        self,
        n_coarse: int = 100,
        n_fine: int = 1000,
        embedding_dim: int = 640,
        n_probe: int = 5,
        batch_size: int = 10000,
        random_state: int = 42,
        config = None
    ):
        """
        Initialize HRM2 Engine.
        
        Args:
            n_coarse: Number of coarse clusters
            n_fine: Fine clusters per coarse cluster
            embedding_dim: Dimension of embedding vectors
            n_probe: Coarse clusters to search
            batch_size: Batch size for K-Means
            random_state: Random seed
            config: M2MConfig for hardware acceleration
        """
        self.n_coarse = n_coarse
        self.n_fine = n_fine
        self.embedding_dim = embedding_dim
        self.n_probe = n_probe
        self.batch_size = batch_size
        self.random_state = random_state
        self.config = config
        
        # Initialize hardware router if configured
        self.router = M2MEngine(config) if config else None
        
        # Storage
        self.splats: List[GaussianSplat] = []
        self.embeddings: Optional[np.ndarray] = None
        
        # Index
        self.coarse_model: Optional[KMeans] = None
        self.coarse_assignments: Optional[np.ndarray] = None
        self.fine_models: Dict[int, KMeans] = {}
        self.fine_assignments: Dict[int, np.ndarray] = {}
        
        # Encoder
        self.encoder = FullEmbeddingBuilder()
        
        # Statistics
        self._is_indexed = False
        self._stats = HRM2Stats()
    
    def add_splats(self, splats: List[GaussianSplat]) -> None:
        """
        Add splats to the engine.
        
        Args:
            splats: List of GaussianSplat objects
        """
        self.splats.extend(splats)
        self._is_indexed = False
    
    def index(self, precomputed_embeddings: np.ndarray = None) -> float:
        """
        Build the hierarchical index.
        
        Args:
            precomputed_embeddings: Optional direct embedding vectors to bypass the encoder
            
        Returns:
            Time taken to build index
        """
        start_time = time.time()
        
        if len(self.splats) == 0 and precomputed_embeddings is None:
            return 0.0
        
        if precomputed_embeddings is not None:
            self.embeddings = np.ascontiguousarray(precomputed_embeddings.astype(np.float32))
            n_samples = self.embeddings.shape[0]
        else:
            # Build embeddings
            positions = np.array([s.position for s in self.splats])
            colors = np.array([s.color for s in self.splats])
            opacities = np.array([s.opacity for s in self.splats])
            scales = np.array([s.scale for s in self.splats])
            rotations = np.array([s.rotation for s in self.splats])
            
            self.embeddings = self.encoder.build(
                positions, colors, opacities, scales, rotations
            )
            # Ensure correct dtype
            self.embeddings = np.ascontiguousarray(self.embeddings.astype(np.float32))
            n_samples = len(self.splats)        
        
        # Level 1: Coarse clustering
        n_coarse_effective = min(self.n_coarse, n_samples // 10)
        n_coarse_effective = max(1, n_coarse_effective)
        
        self.coarse_model = KMeans(
            n_clusters=n_coarse_effective,
            batch_size=self.batch_size,
            random_state=self.random_state
        )
        self.coarse_assignments = self.coarse_model.fit_predict(self.embeddings)
        
        # Level 2: Fine clustering within each coarse cluster
        self.fine_models = {}
        self.fine_assignments = {}
        
        for coarse_id in range(n_coarse_effective):
            mask = self.coarse_assignments == coarse_id
            cluster_indices = np.where(mask)[0]
            
            if len(cluster_indices) < 2:
                self.fine_models[coarse_id] = None
                self.fine_assignments[coarse_id] = np.zeros(len(cluster_indices), dtype=np.int32)
                continue
            
            cluster_embeddings = np.ascontiguousarray(self.embeddings[mask].astype(np.float32))
            
            # Dynamic n_fine based on cluster size
            n_fine_effective = min(self.n_fine, len(cluster_indices) // 5)
            n_fine_effective = max(1, n_fine_effective)
            
            fine_model = KMeans(
                n_clusters=n_fine_effective,
                batch_size=min(self.batch_size, len(cluster_indices)),
                random_state=self.random_state + coarse_id
            )
            
            self.fine_models[coarse_id] = fine_model
            self.fine_assignments[coarse_id] = fine_model.fit_predict(cluster_embeddings)
        
        self._is_indexed = True
        
        # Update stats
        self._stats.n_splats = n_samples
        self._stats.n_coarse_clusters = n_coarse_effective
        self._stats.n_fine_clusters = sum(
            m.n_clusters if m else 0 for m in self.fine_models.values()
        )
        self._stats.build_time = time.time() - start_time
        
        return self._stats.build_time
    
    def query(
        self,
        query_vector: np.ndarray,
        k: int = 10,
        lod: int = 2
    ) -> List[Tuple[GaussianSplat, float]]:
        """
        Query for k most similar splats.
        
        Args:
            query_vector: Query embedding vector
            k: Number of results
            lod: Level of Detail (0=Coarse Approx, 1=Fine Approx, 2=Exact MoE Router)
        
        Returns:
            List of (Splat, distance) tuples
        """
        if not self._is_indexed:
            raise RuntimeError("Index not built. Call index() first.")
        
        query_start = time.time()
        
        query_vector = np.ascontiguousarray(
            np.asarray(query_vector, dtype=np.float32).flatten()
        )
        
        # Find nearest coarse clusters
        coarse_distances = self.coarse_model.transform(query_vector.reshape(1, -1))[0]
        closest_coarse = np.argsort(coarse_distances)[:self.n_probe]
        
        candidates = []
        
        if lod == 0:
            # LOD 0: Ultra-fast coarse approximation. Just return vectors from the nearest macro-clusters.
            for coarse_id in closest_coarse:
                dist = coarse_distances[coarse_id]
                mask = self.coarse_assignments == coarse_id
                indices = np.where(mask)[0][:k]
                for idx in indices:
                    candidates.append((idx, float(dist), int(coarse_id)))
                if len(candidates) >= k:
                    break
                    
        elif lod == 1:
            # LOD 1: Fast fine approximation. Find the closest fine cluster and return its point approximations.
            for coarse_id in closest_coarse:
                fine_model = self.fine_models.get(coarse_id)
                if not fine_model:
                    continue
                fine_dists = fine_model.transform(query_vector.reshape(1, -1))[0]
                closest_fine = np.argsort(fine_dists)[0] # Top 1 fine cluster inside this coarse
                dist = fine_dists[closest_fine]
                
                coarse_mask = self.coarse_assignments == coarse_id
                coarse_indices = np.where(coarse_mask)[0]
                fine_assigns = self.fine_assignments.get(coarse_id, np.zeros(len(coarse_indices), dtype=np.int32))
                
                # Match local fine cluster
                fine_mask = fine_assigns == closest_fine
                indices = coarse_indices[fine_mask][:k]
                for idx in indices:
                    candidates.append((idx, float(dist), int(coarse_id)))
                if len(candidates) >= k:
                    break
        else:
            # LOD 2 (Exact): Collect all points to be scored exactly by MoE router
            expert_embeddings = []
            expert_indices = []
            coarse_ids = []
            fine_ids = []
            
            for coarse_id in closest_coarse:
                mask = self.coarse_assignments == coarse_id
                cluster_indices = np.where(mask)[0]
                
                if len(cluster_indices) == 0:
                    continue
                    
                cluster_embeddings = self.embeddings[mask]
                
                expert_embeddings.append(cluster_embeddings)
                expert_indices.append(cluster_indices)
                coarse_ids.extend([int(coarse_id)] * len(cluster_indices))
                
                # Optionally grab fine IDs if needed
                fine_assigns = self.fine_assignments.get(int(coarse_id), np.zeros(len(cluster_indices), dtype=np.int32))
                fine_ids.extend(fine_assigns.tolist())
                
            if expert_embeddings:
                expert_embeddings = np.vstack(expert_embeddings)
                expert_indices = np.concatenate(expert_indices)
                coarse_ids = np.array(coarse_ids)
                fine_ids = np.array(fine_ids)
                
                if self.router: # Hardware-accelerated MoE distance
                    results = self.router.compute_expert_distances(
                        query_vector, expert_embeddings, expert_indices, coarse_ids, fine_ids
                    )
                    candidates = [(r[0], r[1], r[2]) for r in results]
                else: # CPU fallback
                    distances = np.linalg.norm(expert_embeddings - query_vector, axis=1)
                    for idx, dist, cid in zip(expert_indices, distances, coarse_ids):
                        candidates.append((idx, float(dist), int(cid)))
        
        # Sort by distance and return top-k
        candidates.sort(key=lambda x: x[1])
        results = candidates[:k]
        
        # Update stats
        self._stats.total_queries += 1
        query_time = time.time() - query_start
        self._stats.avg_query_time = (
            (self._stats.avg_query_time * (self._stats.total_queries - 1) + query_time)
            / self._stats.total_queries
        )
        
        return [
            (self.splats[idx], dist) 
            for idx, dist, _ in results
        ]
    
    def query_with_details(
        self,
        query_vector: np.ndarray,
        k: int = 10,
        lod: int = 2
    ) -> List[SearchResult]:
        """
        Query with detailed results including cluster info.
        
        Args:
            query_vector: Query embedding vector
            k: Number of results
            lod: Level of Detail (0=Coarse Approx, 1=Fine Approx, 2=Exact MoE Router)
        
        Returns:
            List of SearchResult objects
        """
        if not self._is_indexed:
            raise RuntimeError("Index not built. Call index() first.")
        
        query_vector = np.ascontiguousarray(
            np.asarray(query_vector, dtype=np.float32).flatten()
        )
        
        # Find nearest coarse clusters
        coarse_distances = self.coarse_model.transform(query_vector.reshape(1, -1))[0]
        closest_coarse = np.argsort(coarse_distances)[:self.n_probe]
        
        candidates = []
        
        if lod == 0:
            for coarse_id in closest_coarse:
                dist = coarse_distances[coarse_id]
                mask = self.coarse_assignments == coarse_id
                indices = np.where(mask)[0][:k]
                for idx in indices:
                    # Provide an approximation for fine_id (e.g. 0) since we didn't search
                    candidates.append((self.splats[idx].id, float(dist), int(coarse_id), 0))
                if len(candidates) >= k:
                    break
                    
        elif lod == 1:
            for coarse_id in closest_coarse:
                fine_model = self.fine_models.get(coarse_id)
                if not fine_model:
                    continue
                fine_dists = fine_model.transform(query_vector.reshape(1, -1))[0]
                closest_fine = np.argsort(fine_dists)[0]
                dist = fine_dists[closest_fine]
                
                coarse_mask = self.coarse_assignments == coarse_id
                coarse_indices = np.where(coarse_mask)[0]
                fine_assigns = self.fine_assignments.get(coarse_id, np.zeros(len(coarse_indices), dtype=np.int32))
                
                fine_mask = fine_assigns == closest_fine
                indices = coarse_indices[fine_mask][:k]
                for idx in indices:
                    candidates.append((self.splats[idx].id, float(dist), int(coarse_id), int(closest_fine)))
                if len(candidates) >= k:
                    break
        else:
            expert_embeddings = []
            expert_indices = []
            coarse_ids = []
            fine_ids = []
            
            for coarse_id in closest_coarse:
                mask = self.coarse_assignments == coarse_id
                cluster_indices = np.where(mask)[0]
                
                if len(cluster_indices) == 0:
                    continue
                    
                cluster_embeddings = self.embeddings[mask]
                
                expert_embeddings.append(cluster_embeddings)
                expert_indices.append(cluster_indices)
                coarse_ids.extend([int(coarse_id)] * len(cluster_indices))
                
                # Form fine assignments
                fine_assigns = self.fine_assignments.get(int(coarse_id), np.zeros(len(cluster_indices), dtype=np.int32))
                fine_ids.extend(fine_assigns.tolist())
                
            if expert_embeddings:
                expert_embeddings = np.vstack(expert_embeddings)
                expert_indices = np.concatenate(expert_indices)
                coarse_ids = np.array(coarse_ids)
                fine_ids = np.array(fine_ids)
                
                if self.router: # Hardware-accelerated MoE distance
                    results = self.router.compute_expert_distances(
                        query_vector, expert_embeddings, expert_indices, coarse_ids, fine_ids
                    )
                    candidates = [(self.splats[r[0]].id, r[1], r[2], r[3]) for r in results]
                else: # CPU fallback
                    distances = np.linalg.norm(expert_embeddings - query_vector, axis=1)
                    for idx, dist, cid, fid in zip(expert_indices, distances, coarse_ids, fine_ids):
                        candidates.append((
                            self.splats[idx].id,
                            float(dist),
                            int(cid),
                            int(fid)
                        ))
        
        # Sort and return top-k
        candidates.sort(key=lambda x: x[1])
        
        return [
            SearchResult(splat_id=sid, distance=d, coarse_cluster=c, fine_cluster=f)
            for sid, d, c, f in candidates[:k]
        ]
    
    def get_stats(self) -> HRM2Stats:
        """Get engine statistics."""
        return self._stats
    
    def clear(self) -> None:
        """Clear all data."""
        self.splats = []
        self.embeddings = None
        self.coarse_model = None
        self.coarse_assignments = None
        self.fine_models = {}
        self.fine_assignments = {}
        self._is_indexed = False
        self._stats = HRM2Stats()


def generate_test_splats(n_splats: int, seed: int = 42) -> List[GaussianSplat]:
    """
    Generate synthetic splats for testing.
    
    Args:
        n_splats: Number of splats
        seed: Random seed
    
    Returns:
        List of GaussianSplat objects
    """
    np.random.seed(seed)
    
    splats = []
    for i in range(n_splats):
        splat = GaussianSplat(
            id=i,
            position=np.random.randn(3).astype(np.float32) * 10,
            color=np.random.rand(3).astype(np.float32),
            opacity=np.random.rand(),
            scale=np.exp(np.random.randn(3).astype(np.float32) * -2),
            rotation=np.random.randn(4).astype(np.float32)
        )
        # Normalize quaternion
        splat.rotation /= np.linalg.norm(splat.rotation)
        splats.append(splat)
    
    return splats
