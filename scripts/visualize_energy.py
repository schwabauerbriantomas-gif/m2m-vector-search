#!/usr/bin/env python3
"""
M2M Energy Landscape Visualizer

Visualizes the energy landscape around a query vector in 2D PCA projection.
Shows:
- Splats as colored points (color = alpha/intensity)
- Query vector
- Energy gradient direction
- Contour map of energy

Usage:
    python scripts/visualize_energy.py --vectors data.npy --query query.npy
    python scripts/visualize_energy.py --random  # Demo with random data
"""
import argparse
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

sys.path.insert(0, "src")

from m2m.config import M2MConfig
from m2m import SimpleVectorDB


def pca_project(vectors, n_components=2):
    """Project vectors to 2D using PCA."""
    from sklearn.decomposition import PCA
    pca = PCA(n_components=n_components)
    return pca.fit_transform(vectors), pca


def visualize_energy_landscape(db, query_vector, k=10, output_path=None):
    """
    Visualize the energy landscape around a query.
    
    Args:
        db: SimpleVectorDB with vectors already added
        query_vector: Query vector [D]
        k: Number of neighbors to highlight
        output_path: If provided, save to file instead of showing
    """
    # Get all vectors from the database
    all_vectors = np.array(list(db._vectors.values()))
    n_vectors = len(all_vectors)
    
    if n_vectors < 10:
        print(f"Need at least 10 vectors for visualization (got {n_vectors})")
        return
    
    # Project to 2D
    projected, pca = pca_project(all_vectors)
    query_2d = pca.transform(query_vector.reshape(1, -1))[0]
    
    # Get neighbors
    results = db.search(query_vector, k=k)
    if isinstance(results, tuple):
        neighbor_vecs, neighbor_alphas, neighbor_kappas = results
        neighbor_vecs = neighbor_vecs[0]  # Remove batch dim
        neighbor_alphas = neighbor_alphas[0]
    else:
        neighbor_vecs = np.array([r.vector for r in results])
        neighbor_alphas = np.ones(len(results))
    
    # Ensure 2D for PCA
    if neighbor_vecs.ndim == 1:
        neighbor_vecs = neighbor_vecs.reshape(1, -1)
    neighbor_2d = pca.transform(neighbor_vecs)
    
    # Compute energy for each point
    energies = db.engine.compute_energy(all_vectors)
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Splats colored by energy
    ax1 = axes[0]
    sc = ax1.scatter(projected[:, 0], projected[:, 1], c=energies, 
                     cmap='viridis_r', alpha=0.6, s=20)
    ax1.scatter(query_2d[0], query_2d[1], c='red', marker='*', s=200, 
                label='Query', zorder=10)
    ax1.scatter(neighbor_2d[:, 0], neighbor_2d[:, 1], c='yellow', 
                marker='o', s=50, label=f'k={k} neighbors', 
                edgecolors='black', linewidths=0.5, zorder=9)
    
    ax1.set_title(f'Energy Landscape (n={n_vectors})')
    ax1.set_xlabel('PC1')
    ax1.set_ylabel('PC2')
    ax1.legend()
    plt.colorbar(sc, ax=ax1, label='Energy')
    
    # Plot 2: Energy distribution
    ax2 = axes[1]
    ax2.hist(energies, bins=50, edgecolor='black', alpha=0.7)
    query_energy = db.engine.compute_energy(query_vector.reshape(1, -1))[0]
    ax2.axvline(query_energy, color='red', linestyle='--', 
                label=f'Query energy: {query_energy:.2f}')
    ax2.set_title('Energy Distribution')
    ax2.set_xlabel('Energy')
    ax2.set_ylabel('Count')
    ax2.legend()
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {output_path}")
    else:
        plt.show()


def demo_with_random(n_vectors=500, dim=64, k=10):
    """Demo visualization with random vectors."""
    print(f"Creating demo with {n_vectors} random {dim}D vectors...")
    
    config = M2MConfig.simple()
    config.latent_dim = dim
    
    db = SimpleVectorDB(latent_dim=dim, enable_ebm=True, mode='ebm')
    
    # Generate clustered random vectors
    n_clusters = 5
    vectors = []
    for i in range(n_clusters):
        center = np.random.randn(dim).astype(np.float32)
        cluster_vectors = center + np.random.randn(n_vectors // n_clusters, dim).astype(np.float32) * 0.3
        vectors.append(cluster_vectors)
    vectors = np.vstack(vectors)
    
    # Normalize
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    vectors /= norms
    
    # Add to database
    ids = [f"vec_{i}" for i in range(len(vectors))]
    db.add(ids=ids, vectors=vectors)
    
    # Generate query from one cluster
    query_cluster = np.random.randint(n_clusters)
    query = vectors[query_cluster * (n_vectors // n_clusters)] + np.random.randn(dim).astype(np.float32) * 0.2
    query /= np.linalg.norm(query)
    
    print(f"Query from cluster {query_cluster}")
    print(f"Visualizing...")
    
    visualize_energy_landscape(db, query, k=k)


def main():
    parser = argparse.ArgumentParser(description="M2M Energy Landscape Visualizer")
    parser.add_argument("--random", action="store_true", help="Demo with random data")
    parser.add_argument("--vectors", type=str, help="Path to vectors .npy file")
    parser.add_argument("--query", type=str, help="Path to query vector .npy file")
    parser.add_argument("--dim", type=int, default=64, help="Vector dimension")
    parser.add_argument("--k", type=int, default=10, help="Number of neighbors")
    parser.add_argument("--output", type=str, help="Output image path")
    parser.add_argument("--n-vectors", type=int, default=500, help="Number of vectors for demo")
    args = parser.parse_args()
    
    if args.random:
        demo_with_random(n_vectors=args.n_vectors, dim=args.dim, k=args.k)
    elif args.vectors and args.query:
        vectors = np.load(args.vectors).astype(np.float32)
        query = np.load(args.query).astype(np.float32)
        
        db = SimpleVectorDB(latent_dim=vectors.shape[1], enable_ebm=True, mode='ebm')
        ids = [f"vec_{i}" for i in range(len(vectors))]
        db.add(ids=ids, vectors=vectors)
        
        visualize_energy_landscape(db, query, k=args.k, output_path=args.output)
    else:
        print("Use --random for demo or provide --vectors and --query")
        parser.print_help()


if __name__ == "__main__":
    main()
