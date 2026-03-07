#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
M2M (Machine-to-Memory) - High-performance Gaussian Splat Storage and Retrieval
High-performance Gaussian Splat storage and retrieval for AI systems
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Any

# M2M Core Modules
try:
    from .geometry import (
        normalize_sphere,
        geodesic_distance,
        exp_map,
        log_map,
        project_to_tangent,
    )
    from .splats import SplatStore
    from .energy import EnergyFunction
    from .engine import M2MEngine
    from .config import M2MConfig
except ImportError:
    # Forward declarations for type hinting if core modules aren't available
    M2MEngine = None
    from .config import M2MConfig


# Create dummy functions
def normalize_sphere(x, dim=-1):
    norm = np.linalg.norm(x, axis=dim, keepdims=True)
    return x / (norm + 1e-8)


def geodesic_distance(x, y):
    return np.arccos(np.clip(np.dot(x, y.T), -1, 1))


def exp_map(base, tangent):
    return base + tangent


def log_map(base, point):
    return point - base


def project_to_tangent(base, vector):
    return vector - np.dot(vector, base.T) * base


class M2MMemory:
    """
    M2M Memory System

    Manages:
    - Gaussian Splats (μ, α, κ)
    - 3-tier Memory Hierarchy (VRAM Hot, RAM Warm, SSD Cold)
    - SOC Controller (Self-Organized Criticality)
    """

    def __init__(self, config: M2MConfig):
        self.config = config

        # 3-tier Memory Hierarchy
        self.vram_hot = config.enable_3_tier_memory
        self.ram_warm = config.enable_3_tier_memory
        self.ssd_cold = config.enable_3_tier_memory

        # Splat Store
        self.splats = SplatStore(config)

        # SOC Controller
        self.soc_threshold = config.soc_threshold
        self.phi = 0.0  # Order parameter

        # Energy Function
        self.energy_fn = EnergyFunction(config)

        print(
            f"[INFO] M2M initialized on {config.device} (compute_device={config.compute_device})"
        )
        print(f"[INFO] Latent dim: {config.latent_dim}")
        print(f"[INFO] Max splats: {config.max_splats}")
        print(f"[INFO] 3-tier memory: {config.enable_3_tier_memory}")

    def encode(self, x: np.ndarray) -> np.ndarray:
        """Encode input x to spherical latent space S^639."""
        # Normalize to unit sphere
        x_norm = normalize_sphere(x)
        return x_norm

    def compute_energy(self, x: np.ndarray) -> np.ndarray:
        """Compute energy E(x) = E_splats + E_geom + E_comp - H[q(x)]"""
        x_encoded = self.encode(x)

        # Compute splat energy
        E_splats = self.energy_fn.E_splats(x_encoded, self.splats)

        # Compute geometric energy
        E_geom = self.energy_fn.E_geom(x_encoded)

        # Compute composition energy
        E_comp = self.energy_fn.E_comp(x_encoded)

        # Entropy term H[q(x)]
        H_q = self.splats.entropy(x_encoded)

        # Total energy (weighted sum)
        E = (
            self.config.energy_splat_weight * E_splats
            + self.config.energy_geom_weight * E_geom
            + self.config.energy_comp_weight * E_comp
            - self.config.temperature * H_q
        )

        return E

    def retrieve(
        self, query: np.ndarray, k: int = None, lod: int = 2
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Retrieve k-nearest neighbors from splat store."""
        if k is None:
            k = self.config.knn_k

        # Encode query to spherical space
        query_norm = normalize_sphere(query)

        # Find neighbors (using LOD semantics)
        neighbors_mu, neighbors_alpha, neighbors_kappa = self.splats.find_neighbors(
            query_norm, k, lod=lod
        )

        return neighbors_mu, neighbors_alpha, neighbors_kappa

    def sample(self, x: np.ndarray, n_steps: int = None) -> np.ndarray:
        """Generate samples using Langevin dynamics."""
        if n_steps is None:
            n_steps = self.config.langevin_steps

        # Simplified native Langevin behavior (diffusion)
        noise = np.random.randn(*x.shape).astype(np.float32) * self.config.langevin_dt
        return x + noise

    def consolidate(self, threshold: float = None) -> int:
        """Run SOC consolidation based on order parameter."""
        if threshold is None:
            threshold = self.soc_threshold

        # Remove low-alpha splats (proxy for order parameter)
        if self.splats.n_active > 0:
            splats_to_remove = np.where(
                self.splats.alpha[: self.splats.n_active] < threshold
            )[0]
        else:
            splats_to_remove = []

        # Remove splats
        n_removed = 0
        for splat_idx in splats_to_remove:
            self.splats.mu[splat_idx] = float("inf")  # Mark for removal
            n_removed += 1

        if n_removed > 0:
            # Compact splats (remove marked ones)
            self.splats.compact()
            print(f"[INFO] SOC: Consolidated {n_removed} splats")

        return n_removed

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the M2M system."""
        return {
            "n_active_splats": self.splats.n_active,
            "max_splats": self.splats.max_splats,
            "mean_frequency": self.splats.frequency[: self.splats.n_active]
            .mean()
            .item(),
            "mean_kappa": self.splats.kappa[: self.splats.n_active].mean().item(),
            "mean_alpha": self.splats.alpha[: self.splats.n_active].mean().item(),
            "entropy": self.splats.entropy().item(),
            "soc_phi": self.phi.item(),
        }

    def forward(self, x: np.ndarray, mode: str = "energy") -> np.ndarray:
        """Forward pass of M2M system."""
        if mode == "energy":
            return self.compute_energy(x)
        elif mode == "retrieve":
            neighbors_mu, neighbors_alpha, neighbors_kappa = self.retrieve(x)
            return neighbors_mu
        else:
            raise ValueError(f"Unknown mode: {mode}")


class M2MEngine:
    """
    M2M High-Performance Engine with REST/gRPC APIs
    """

    def __init__(self, config: M2MConfig):
        super().__init__()
        self.config = config

        # Initialize M2M Memory
        self.m2m = M2MMemory(config)

        # Initialize Vulkan Engine if enabled
        if config.enable_vulkan:
            try:
                from engine import M2MEngine as VulkanEngine

                self.vulkan_engine = VulkanEngine(config)
            except ImportError:
                print("[WARNING] Vulkan engine module not found, falling back to None.")
                self.vulkan_engine = None
        else:
            self.vulkan_engine = None
            print("[INFO] Vulkan acceleration disabled")

        print(
            f"[INFO] M2M Engine initialized on {config.device} (compute_device={config.compute_device})"
        )
        print(
            f"[INFO] Vulkan GPU Compute: {'Enabled' if config.enable_vulkan else 'Disabled'}"
        )

    def add_splats(self, vectors: np.ndarray, labels: List[str] = None) -> int:
        """Add new splats to the system."""
        # Normalize vectors to sphere
        vectors_norm = normalize_sphere(vectors)

        # Add to splat store
        n_added = 0
        for i, vector_norm in enumerate(vectors_norm):
            if self.m2m.splats.add_splat(vector_norm):
                n_added += 1
            else:
                print(f"[WARNING] Failed to add splat {i}")

        # Build the semantic router index automatically
        if n_added > 0:
            self.m2m.splats.build_index()

        print(f"[INFO] Added {n_added} splats and built HRM2 index")
        return n_added

    def search(self, query: np.ndarray, k: int = None) -> np.ndarray:
        """Search for nearest neighbors."""
        return self.m2m.retrieve(query, k)

    def generate(self, query: np.ndarray, n_samples: int = 10) -> np.ndarray:
        """Generate samples starting from query."""
        return self.m2m.sample(query)

    def get_statistics(self) -> Dict[str, Any]:
        """Get system statistics."""
        return self.m2m.get_statistics()

    def encode(self, x: np.ndarray) -> np.ndarray:
        """Encode input to spherical latent space."""
        return self.m2m.encode(x)

    def compute_energy(self, x: np.ndarray) -> np.ndarray:
        """Compute energy of input."""
        return self.m2m.compute_energy(x)

    def export_to_dataloader(
        self,
        batch_size=32,
        num_workers=0,
        importance_sampling=False,
        generate_samples=False,
    ):
        """Export M2M Data Lake as a numpy-based iterable of batches.

        Returns an iterable that yields numpy arrays of splat μ-vectors
        in batches. Compatible with standard Python iteration.
        """
        from .data_lake import M2MDataLake

        dataset = M2MDataLake(
            m2m_engine=self,
            batch_size=batch_size,
            importance_sampling=importance_sampling,
            generate_samples=generate_samples,
        )
        return dataset

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass (energy computation)."""
        return self.m2m(x)

    def load_optimized(self, path: str):
        """Loads optimized dataset with pre-computed Gaussian Splats."""
        from loaders.optimized_loader import load_m2m_dataset

        splats = load_m2m_dataset(path)

        # Use splats directly without re-computing
        self._build_hrm2_from_splats(splats)

        return self

    def _build_hrm2_from_splats(self, splats):
        """Builds HRM2 index directly from pre-computed splats."""
        # The splats already have a hierarchical structure
        # We only need to build the search tree in the SplatStore
        self.m2m.splats._build_hrm2_from_splats(splats)


class SimpleVectorDB:
    """
    SimpleVectorDB: 'The SQLite of Vector DBs'
    Minimalist, fast configuration tailored purely for vectors storage and search
    on edge devices. Disables expensive agentic logic (like SOC or generative dynamics).
    """

    def __init__(
        self,
        device: Optional[str] = None,
        latent_dim: int = 640,
        enable_lsh_fallback: bool = True,
        lsh_threshold: float = 0.15,
    ):
        config = M2MConfig.simple(device=device)
        config.latent_dim = latent_dim
        self.engine = M2MEngine(config)

        self.enable_lsh_fallback = enable_lsh_fallback
        self.lsh_threshold = lsh_threshold
        self.lsh = None
        self._use_lsh = False

    def _compute_silhouette(
        self, vectors: np.ndarray, sample_size: int = 1000
    ) -> float:
        try:
            from sklearn.cluster import KMeans
            from sklearn.metrics import silhouette_score

            sample_idx = np.random.choice(
                len(vectors), min(sample_size, len(vectors)), replace=False
            )
            sample = vectors[sample_idx]

            # K-Means con k = sqrt(N)
            n_clusters = max(2, int(np.sqrt(len(sample))))
            kmeans = KMeans(n_clusters=n_clusters, n_init=1, random_state=42)
            labels = kmeans.fit_predict(sample)

            return silhouette_score(sample, labels)
        except ImportError:
            print(
                "[WARNING] scikit-learn is required for data distribution analysis. LSH fallback will not activate automatically."
            )
            return 1.0  # High value to bypass LSH fallback

    def add(self, vectors: np.ndarray) -> int:
        """Add vectors to the database."""

        if self.enable_lsh_fallback:
            silhouette = self._compute_silhouette(vectors)

            if silhouette < self.lsh_threshold:
                print(f"[M2M] Distribución uniforme (silhouette={silhouette:.4f})")
                print("[M2M] Activando LSH fallback...")

                try:
                    from .lsh_index import CrossPolytopeLSH, LSHConfig

                    config = LSHConfig(
                        dim=self.engine.config.latent_dim,
                        n_tables=15,
                        n_bits=18,
                        n_probes=50,
                        n_candidates=500,
                    )
                    self.lsh = CrossPolytopeLSH(config)
                    self.lsh.index(vectors)
                    self._use_lsh = True
                    return len(vectors)
                except ImportError as e:
                    print(
                        f"[WARNING] Could not load LSH module: {e}. Falling back to normal HRM2."
                    )

        self._use_lsh = False
        return self.engine.add_splats(vectors)

    def search(self, query: np.ndarray, k: int = 64):
        """Search nearest neighbors."""
        if self._use_lsh and self.lsh is not None:
            # LSH query returns coordinates and distances, we just need indices for engine compatibility
            # Let's align the LSH output with the expected format
            indices, distances = self.lsh.query(query, k=k)
            # The engine search normally returns mu, alpha, kappa
            # For this fallback, we can fetch exactly what is needed or return the raw vectors
            # SimpleVectorDB engine.search(query, k) returns neighbors_mu, neighbors_alpha, neighbors_kappa
            # We'll just return the vector elements + dummy alpha and kappa to preserve the signature
            candidate_vectors = self.lsh.vectors[indices]
            dummy_alpha = np.ones(len(indices), dtype=np.float32)
            dummy_kappa = np.ones(len(indices), dtype=np.float32) * 10.0
            return candidate_vectors, dummy_alpha, dummy_kappa

        return self.engine.search(query, k)

    def load(self, path: str):
        """Load a pre-computed vector index off disk."""
        self.engine.load_optimized(path)
        return self


class AdvancedVectorDB:
    """
    AdvancedVectorDB: The complete agent-ready feature suite.
    Employs industry best practices for Autonomous Generative Agents:
    1. Exploration (Langevin Dynamics): Curiosity-driven latent space walking.
    2. Context Management (3-tier memory): Hot VRAM, Warm RAM, Cold SSD scaling.
    3. Forgetting Mechanisms (SOC): Self-Organized Criticality safely prunes dead memory.
    """

    def __init__(self, device: Optional[str] = None, latent_dim: int = 640):
        config = M2MConfig.advanced(device=device)
        config.latent_dim = latent_dim
        self.engine = M2MEngine(config)

    def add(self, vectors: np.ndarray) -> int:
        """Add vectors to the database."""
        return self.engine.add_splats(vectors)

    def search(self, query: np.ndarray, k: int = 64):
        """Search nearest neighbors."""
        return self.engine.search(query, k)

    def generate(self, query: np.ndarray, n_steps: int = 10) -> np.ndarray:
        """Perform Langevin dynamics generative exploration starting from query."""
        return self.engine.generate(query, n_samples=n_steps)

    def consolidate(self, threshold: float = None) -> int:
        """Run Self-Organized Criticality memory consolidation."""
        return self.engine.m2m.consolidate(threshold)

    def load(self, path: str):
        """Load a pre-computed vector space off disk."""
        self.engine.load_optimized(path)
        return self


# Factory function
def create_m2m(config: M2MConfig) -> M2MEngine:
    """Factory function to create M2M Engine."""
    return M2MEngine(config)


# CLI interface
def main():
    """Main CLI entry point for M2M."""
    import argparse

    parser = argparse.ArgumentParser(
        description="M2M (Machine-to-Memory) - High-performance Gaussian Splat storage and retrieval",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--config", type=str, default=None, help="Path to config file")
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cpu", "cuda", "vulkan"],
        help="Device to use",
    )
    parser.add_argument(
        "--n-splats", type=int, default=10000, help="Initial number of splats"
    )
    parser.add_argument(
        "--max-splats", type=int, default=100000, help="Maximum number of splats"
    )
    parser.add_argument(
        "--knn-k", type=int, default=64, help="K for K-nearest neighbors"
    )
    parser.add_argument(
        "--enable-vulkan", action="store_true", help="Enable Vulkan acceleration"
    )
    parser.add_argument(
        "--disable-vulkan", action="store_true", help="Disable Vulkan acceleration"
    )
    parser.add_argument(
        "--transform-dataset",
        nargs=2,
        metavar=("INPUT.npy", "OUTPUT.bin"),
        help="Transform a flat embeddings dataset to M2M hierarchial splats offline",
    )
    parser.add_argument(
        "--load-optimized",
        type=str,
        metavar="INPUT.bin",
        help="Start M2M from a pre-transformed dataset directly",
    )

    args = parser.parse_args()

    # Check offline transform path first
    if args.transform_dataset:
        input_path, output_path = args.transform_dataset
        print("=" * 60)
        print("M2M Dataset Transformer")
        print("=" * 60)
        print(f"Input: {input_path}")
        print(f"Output: {output_path}")

        import numpy as np
        from dataset_transformer import M2MDatasetTransformer

        print("\n[DATA] Loading input vectors...")
        try:
            vectors = np.load(input_path)
            if not isinstance(vectors, np.ndarray):
                print("[ERROR] Input must be a valid numpy array.")
                return
        except Exception as e:
            print(f"[ERROR] Failed to load {input_path}: {e}")
            return

        print(f"[TRANSFORM] Structure: {vectors.shape}")
        transformer = M2MDatasetTransformer(vectors)
        transformer.save_for_m2m(output_path)
        return

    # Determine device
    device = args.device
    if device is None:
        device = "cpu"  # Default to CPU; use --device vulkan for GPU acceleration

    # Create configuration
    config = M2MConfig(
        device=device,
        n_splats_init=args.n_splats,
        max_splats=args.max_splats,
        knn_k=args.knn_k,
        enable_vulkan=args.enable_vulkan
        or (not args.disable_vulkan and device == "cuda"),
    )

    # Create M2M Engine
    print("=" * 60)
    print("M2M (Machine-to-Memory)")
    print("=" * 60)
    print()
    print(f"Device: {config.device}")
    print(f"Splats: {config.n_splats_init} → {config.max_splats}")
    print(f"KNN: {config.knn_k}")
    print(f"Vulkan: {'Enabled' if config.enable_vulkan else 'Disabled'}")
    print()

    m2m = create_m2m(config)

    # Optional optimized load
    if args.load_optimized:
        print(f"[LOAD] Loading optimized dataset from {args.load_optimized}...")
        try:
            m2m.load_optimized(args.load_optimized)
            print("[LOAD] Successfully booted from pre-computed hierarchy.")
        except Exception as e:
            print(f"[ERROR] Failed to load optimized dataset: {e}")
            return
    else:
        # Example: Add random splats
        print("[EXAMPLE] Adding 100 random splats...")
        random_vectors = np.random.randn(100, config.latent_dim).astype(np.float32)
    n_added = m2m.add_splats(random_vectors)
    print(f"[EXAMPLE] Added {n_added} splats")
    print()

    # Get statistics
    stats = m2m.get_statistics()
    print("[STATS]")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    print()

    print("[SUCCESS] M2M initialized successfully")
    print("=" * 60)


if __name__ == "__main__":
    main()
