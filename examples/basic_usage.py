#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
M2M Basic Usage Example

Demonstrates basic functionality of M2M (Machine-to-Memory):
- Encoding embeddings to spherical space
- Adding splats to storage
- Retrieving nearest neighbors
"""

import torch
import numpy as np
from pathlib import Path

# Import M2M modules
import sys
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from m2m import M2MConfig, normalize_sphere, SplatStore, EnergyFunction
except ImportError as e:
    print(f"[ERROR] Could not import M2M modules: {e}")
    sys.exit(1)


def main():
    """Run basic usage example of M2M."""
    print("=" * 60)
    print("M2M (Machine-to-Memory) - Basic Usage Example")
    print("=" * 60)
    print()
    
    # Configuration
    config = M2MConfig(
        device='cpu',  # Use CPU for this example
        latent_dim=640,  # S^639
        n_splats_init=10000,  # Start with 10K splats
        max_splats=100000,  # Max 100K splats
        knn_k=64  # Retrieve 64 nearest neighbors
    )
    
    print(f"Configuration:")
    print(f"  Device: {config.device}")
    print(f"  Latent dim: {config.latent_dim}")
    print(f"  Max splats: {config.max_splats}")
    print(f"  KNN K: {config.knn_k}")
    print()
    
    # Initialize components
    print("Initializing M2M components...")
    splat_store = SplatStore(config)
    energy_fn = EnergyFunction(config)
    print("[SUCCESS] M2M initialized")
    print()
    
    # Generate random embeddings
    print("Generating random embeddings (100 vectors)...")
    embeddings = torch.randn(100, config.latent_dim)
    print("[INFO] Embeddings shape:", embeddings.shape)
    print()
    
    # Encode to spherical space
    print("Encoding to spherical space (S^639)...")
    embeddings_sphere = normalize_sphere(embeddings)
    print("[INFO] Encoded embeddings shape:", embeddings_sphere.shape)
    print()
    
    # Add to splat store
    print("Adding embeddings to Splat Store...")
    n_added = splat_store.add_splat(embeddings_sphere)
    print(f"[SUCCESS] Added {n_added} splats")
    print()
    
    # Retrieve nearest neighbors
    print("Retrieving nearest neighbors (K=64)...")
    query = embeddings_sphere[:5]  # Use first 5 as queries
    neighbors_mu, neighbors_alpha, neighbors_kappa = splat_store.find_neighbors(query, k=64)
    print(f"[SUCCESS] Retrieved neighbors")
    print(f"  Query shape: {query.shape}")
    print(f"  Neighbors shape: {neighbors_mu.shape}")
    print()
    
    # Compute energy
    print("Computing energy for queries...")
    energy = energy_fn(query)
    print(f"[INFO] Energy shape: {energy.shape}")
    print(f"  Mean energy: {energy.mean().item():.4f}")
    print()
    
    # Statistics
    print("Splat Store Statistics:")
    stats = splat_store.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    print()
    
    print("=" * 60)
    print("Basic usage example completed successfully")
    print("=" * 60)


if __name__ == "__main__":
    main()
