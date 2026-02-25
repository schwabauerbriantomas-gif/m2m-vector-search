#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
M2M (Machine-to-Memory) - High-performance Gaussian Splat Storage and Retrieval
High-performance Gaussian Splat storage and retrieval for AI systems
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from pathlib import Path
import time

# M2M Core Modules
try:
    from geometry import (
        normalize_sphere,
        geodesic_distance,
        exp_map,
        log_map,
        project_to_tangent
    )
    from splats import SplatStore
    from energy import EnergyFunction
    from engine import M2MEngine
except ImportError:
    pass

class DummyModule:
    def __init__(self, *args, **kwargs):
        pass
sample_langevin = DummyModule
HistoryBuffer = DummyModule
compute_order_parameter = lambda *args, **kwargs: torch.tensor([0])
EBMDecoder = DummyModule
M2MEngine = DummyModule

# Create dummy functions
def normalize_sphere(x, dim=-1):
    return x / (torch.norm(x, dim=dim, keepdim=True) + 1e-8)

def geodesic_distance(x, y):
    return torch.acos(torch.clamp(torch.matmul(x, y.T), -1, 1))

def exp_map(base, tangent):
    return base + tangent

def log_map(base, point):
    return point - base

def project_to_tangent(base, vector):
    return vector - torch.matmul(vector, base.unsqueeze(-1)) * base


@dataclass
class M2MConfig:
    """Configuration for M2M system."""
    
    # System Configuration
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    latent_dim: int = 640  # S^639 hyper-sphere
    dtype: torch.dtype = torch.float32
    
    # Splat Configuration
    n_splats_init: int = 10000  # Initial number of splats
    max_splats: int = 100000   # Maximum capacity
    knn_k: int = 64             # K-nearest neighbors for retrieval
    
    # Splat Parameters
    init_alpha: float = 1.0       # Initial amplitude
    init_kappa: float = 10.0      # Initial concentration
    min_kappa: float = 1.0       # Minimum concentration (avoid collapse)
    max_kappa: float = 50.0      # Maximum concentration limit
    
    # Temperature (for exploration)
    splat_temperature: float = 0.1
    weight_decay: float = 0.0     # Weight decay per epoch
    weight_decay_start: float = 1.0
    
    # Energy Parameters
    energy_splat_weight: float = 1.0    # Weight for splat energy term
    energy_geom_weight: float = 0.1      # Weight for geometric energy
    energy_comp_weight: float = 0.0      # Weight for composition energy
    temperature: float = 1.0            # Global temperature for sampling
    
    # Langevin Parameters
    langevin_steps: int = 200       # Number of Langevin steps
    langevin_dt: float = 0.001      # Time step
    langevin_gamma: float = 0.1      # Friction coefficient
    langevin_T: float = 1.0           # Temperature
    
    # SOC Parameters
    soc_threshold: float = 0.8       # Criticality threshold for consolidation
    soc_buffer_capacity: int = 1000  # Number of splats to track
    soc_update_interval: int = 100  # Update SOC every N samples
    phi_convergence_threshold: float = 0.95  # Order parameter convergence
    
    # Memory Hierarchy
    enable_3_tier_memory: bool = True  # Enable VRAM/RAM/SSD hierarchy
    
    # Decoder Configuration (MoE)
    vocab_size: int = 50257         # GPT-2 vocabulary
    hidden_dim: int = 1024           # Hidden dimension
    moe_experts: int = 4            # Number of experts
    moe_active: int = 2              # Number of active experts
    
    # Vulkan Configuration
    enable_vulkan: bool = True       # Enable Vulkan compute shaders
    vulkan_device_index: int = 0    # GPU device index
    
    # API Configuration
    rest_port: int = 8080
    grpc_port: int = 9090
    
    def __post_init__(self):
        """Handle 'vulkan' device: enable Vulkan GPU compute shaders."""
        if self.device == 'vulkan':
            self.enable_vulkan = True
    
    @property
    def torch_device(self) -> str:
        """PyTorch-compatible device for tensor allocation.
        
        When device='vulkan', tensors are stored on CPU but heavy compute
        (distances, MoE routing) runs on the GPU via Vulkan compute shaders.
        """
        if self.device == 'vulkan':
            return 'cpu'
        return self.device


class M2MMemory(nn.Module):
    """
    M2M Memory System
    
    Manages:
    - Gaussian Splats (μ, α, κ)
    - 3-tier Memory Hierarchy (VRAM Hot, RAM Warm, SSD Cold)
    - SOC Controller (Self-Organized Criticality)
    """
    
    def __init__(self, config: M2MConfig):
        super().__init__()
        self.config = config
        
        # 3-tier Memory Hierarchy
        self.vram_hot = config.enable_3_tier_memory
        self.ram_warm = config.enable_3_tier_memory
        self.ssd_cold = config.enable_3_tier_memory
        
        # Splat Store
        self.splats = SplatStore(config)
        
        # SOC Controller
        self.soc_history = HistoryBuffer(config)
        self.soc_threshold = config.soc_threshold
        self.phi = 0.0  # Order parameter
        
        # Energy Function
        self.energy_fn = EnergyFunction(config)
        
        # Langevin Sampler
        self.sampler = sample_langevin
        
        # Decoder
        self.decoder = EBMDecoder(config)
        
        # Move to device
        self.to(config.torch_device)
        
        print(f"[INFO] M2M initialized on {config.device} (torch_device={config.torch_device})")
        print(f"[INFO] Latent dim: {config.latent_dim}")
        print(f"[INFO] Max splats: {config.max_splats}")
        print(f"[INFO] 3-tier memory: {config.enable_3_tier_memory}")
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input x to spherical latent space S^639."""
        # Normalize to unit sphere
        x_norm = normalize_sphere(x)
        return x_norm
    
    def compute_energy(self, x: torch.Tensor) -> torch.Tensor:
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
        E = (self.config.energy_splat_weight * E_splats +
              self.config.energy_geom_weight * E_geom +
              self.config.energy_comp_weight * E_comp -
              self.config.temperature * H_q)
        
        return E
    
    def retrieve(self, query: torch.Tensor, k: int = None, lod: int = 2) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Retrieve k-nearest neighbors from splat store."""
        if k is None:
            k = self.config.knn_k
        
        # Encode query to spherical space
        query_norm = normalize_sphere(query)
        
        # Find neighbors (using LOD semantics)
        neighbors_mu, neighbors_alpha, neighbors_kappa = self.splats.find_neighbors(query_norm, k, lod=lod)
        
        return neighbors_mu, neighbors_alpha, neighbors_kappa
    
    def sample(self, x: torch.Tensor, n_steps: int = None) -> torch.Tensor:
        """Generate samples using underdamped Langevin dynamics."""
        if n_steps is None:
            n_steps = self.config.langevin_steps
        
        return self.sampler(
            x=x,
            energy_fn=self.compute_energy,
            dt=self.config.langevin_dt,
            gamma=self.config.langevin_gamma,
            T=self.config.langevin_T,
            n_steps=n_steps
        )
    
    def consolidate(self, threshold: float = None) -> int:
        """Run SOC consolidation based on order parameter."""
        if threshold is None:
            threshold = self.soc_threshold
        
        splats_to_remove = compute_order_parameter(
            self.splats.mu,
            self.splats.alpha,
            self.splats.frequency,
            threshold=threshold
        )
        
        # Remove splats
        n_removed = 0
        for splat_idx in splats_to_remove:
            self.splats.mu[splat_idx] = float('inf')  # Mark for removal
            n_removed += 1
        
        if n_removed > 0:
            # Compact splats (remove marked ones)
            self.splats.compact()
            print(f"[INFO] SOC: Consolidated {n_removed} splats")
        
        return n_removed
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the M2M system."""
        return {
            'n_active_splats': self.splats.n_active,
            'max_splats': self.splats.max_splats,
            'mean_frequency': self.splats.frequency[:self.splats.n_active].mean().item(),
            'mean_kappa': self.splats.kappa[:self.splats.n_active].mean().item(),
            'mean_alpha': self.splats.alpha[:self.splats.n_active].mean().item(),
            'entropy': self.splats.entropy().item(),
            'soc_phi': self.phi.item()
        }
    
    def forward(self, x: torch.Tensor, mode: str = 'energy') -> torch.Tensor:
        """Forward pass of M2M system."""
        if mode == 'energy':
            return self.compute_energy(x)
        elif mode == 'retrieve':
            neighbors_mu, neighbors_alpha, neighbors_kappa = self.retrieve(x)
            return neighbors_mu
        else:
            raise ValueError(f"Unknown mode: {mode}")


class M2MEngine(nn.Module):
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
        
        # Move to device
        self.to(config.torch_device)
        
        print(f"[INFO] M2M Engine initialized on {config.device} (torch_device={config.torch_device})")
        print(f"[INFO] Vulkan GPU Compute: {'Enabled' if config.enable_vulkan else 'Disabled'}")
    
    def add_splats(self, vectors: torch.Tensor, labels: List[str] = None) -> int:
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
    
    def search(self, query: torch.Tensor, k: int = None) -> torch.Tensor:
        """Search for nearest neighbors."""
        return self.m2m.retrieve(query, k)
    
    def generate(self, query: torch.Tensor, n_samples: int = 10) -> torch.Tensor:
        """Generate samples starting from query."""
        return self.m2m.sample(query)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get system statistics."""
        return self.m2m.get_statistics()
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to spherical latent space."""
        return self.m2m.encode(x)
    
    def compute_energy(self, x: torch.Tensor) -> torch.Tensor:
        """Compute energy of input."""
        return self.m2m.compute_energy(x)
        
    def export_to_dataloader(self, batch_size=32, num_workers=0, importance_sampling=False, generate_samples=False):
        """Export M2M Data Lake as a PyTorch DataLoader."""
        from data_lake import M2MDataLake
        from torch.utils.data import DataLoader
        
        dataset = M2MDataLake(
            m2m_engine=self,
            batch_size=batch_size,
            importance_sampling=importance_sampling,
            generate_samples=generate_samples
        )
        return DataLoader(dataset, batch_size=None, num_workers=num_workers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass (energy computation)."""
        return self.m2m(x)

    def load_optimized(self, path: str):
        """Carga dataset transformado con Gaussian Splats pre-computados."""
        from loaders.optimized_loader import load_m2m_dataset
        
        splats = load_m2m_dataset(path)
        
        # Usar splats directamente sin re-computar
        self._build_hrm2_from_splats(splats)
        
        return self

    def _build_hrm2_from_splats(self, splats):
        """Construye índice HRM2 desde splats pre-computados."""
        # Los splats ya tienen estructura jerárquica
        # Solo necesitamos construir el árbol de búsqueda en el SplatStore
        self.m2m.splats._build_hrm2_from_splats(splats)


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
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--config', type=str, default=None, help='Path to config file')
    parser.add_argument('--device', type=str, default=None, choices=['cpu', 'cuda', 'vulkan'], help='Device to use')
    parser.add_argument('--n-splats', type=int, default=10000, help='Initial number of splats')
    parser.add_argument('--max-splats', type=int, default=100000, help='Maximum number of splats')
    parser.add_argument('--knn-k', type=int, default=64, help='K for K-nearest neighbors')
    parser.add_argument('--enable-vulkan', action='store_true', help='Enable Vulkan acceleration')
    parser.add_argument('--disable-vulkan', action='store_true', help='Disable Vulkan acceleration')
    parser.add_argument('--transform-dataset', nargs=2, metavar=('INPUT.npy', 'OUTPUT.bin'), help='Transform a flat embeddings dataset to M2M hierarchial splats offline')
    parser.add_argument('--load-optimized', type=str, metavar='INPUT.bin', help='Start M2M from a pre-transformed dataset directly')
    
    args = parser.parse_args()
    
    # Check offline transform path first
    if args.transform_dataset:
        input_path, output_path = args.transform_dataset
        print("=" * 60)
        print(f"M2M Dataset Transformer")
        print("=" * 60)
        print(f"Input: {input_path}")
        print(f"Output: {output_path}")
        
        import numpy as np
        from dataset_transformer import M2MDatasetTransformer
        
        print("\n[DATA] Loading input vectors...")
        try:
            vectors = np.load(input_path)
            if not isinstance(vectors, np.ndarray):
                print(f"[ERROR] Input must be a valid numpy array.")
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
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create configuration
    config = M2MConfig(
        device=device,
        n_splats_init=args.n_splats,
        max_splats=args.max_splats,
        knn_k=args.knn_k,
        enable_vulkan=args.enable_vulkan or (not args.disable_vulkan and device == 'cuda')
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
            print(f"[LOAD] Successfully booted from pre-computed hierarchy.")
        except Exception as e:
            print(f"[ERROR] Failed to load optimized dataset: {e}")
            return
    else:
        # Example: Add random splats
        print("[EXAMPLE] Adding 100 random splats...")
        random_vectors = torch.randn(100, config.latent_dim).to(config.torch_device)
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
