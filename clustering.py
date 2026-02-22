"""
KMeans clustering with Numba JIT optimization.

This module provides fast K-Means clustering implementation
optimized for Gaussian splat embeddings using Mini-batch approach.
"""

import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass

try:
    from numba import njit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        if len(args) == 1 and callable(args[0]):
            return args[0]
        return decorator
    prange = range


# ==================== K-MEANS++ INITIALIZATION ====================

@njit(fastmath=True, cache=True)
def kmeans_plusplus_init(
    data: np.ndarray,
    n_clusters: int,
    random_state: int = 42
) -> np.ndarray:
    """
    KMeans++ initialization for better centroid selection.
    
    Args:
        data: (N, D) array of data points
        n_clusters: Number of clusters
        random_state: Random seed
    
    Returns:
        (n_clusters, D) array of initial centroids
    """
    np.random.seed(random_state)
    N, D = data.shape
    centroids = np.zeros((n_clusters, D), dtype=np.float32)
    
    # Choose first centroid randomly
    idx = np.random.randint(0, N)
    centroids[0] = data[idx]
    
    # Choose remaining centroids with probability proportional to distance squared
    for k in range(1, n_clusters):
        # Compute distances to nearest centroid
        distances = np.full(N, np.inf, dtype=np.float32)
        
        for i in range(N):
            for j in range(k):
                dist = 0.0
                for d in range(D):
                    diff = data[i, d] - centroids[j, d]
                    dist += diff * diff
                if dist < distances[i]:
                    distances[i] = dist
        
        # Sample proportional to distance squared
        total = np.sum(distances) + 1e-10
        probs = distances / total
        cumprobs = np.cumsum(probs)
        
        r = np.random.random()
        idx = np.searchsorted(cumprobs, r)
        idx = min(idx, N - 1)
        
        centroids[k] = data[idx]
    
    return centroids


# ==================== CLUSTER ASSIGNMENT ====================

@njit(fastmath=True, cache=True)
def assign_clusters(
    data: np.ndarray,
    centroids: np.ndarray
) -> np.ndarray:
    """
    Assign each point to nearest centroid.
    
    Args:
        data: (N, D) array
        centroids: (K, D) array
    
    Returns:
        (N,) array of cluster labels
    """
    N = data.shape[0]
    K = centroids.shape[0]
    labels = np.zeros(N, dtype=np.int32)
    
    for i in prange(N):
        min_dist = np.inf
        min_label = 0
        
        for k in range(K):
            dist = 0.0
            for d in range(data.shape[1]):
                diff = data[i, d] - centroids[k, d]
                dist += diff * diff
            
            if dist < min_dist:
                min_dist = dist
                min_label = k
        
        labels[i] = min_label
    
    return labels


# ==================== CENTROID UPDATE ====================

@njit(fastmath=True, cache=True)
def update_centroids(
    data: np.ndarray,
    labels: np.ndarray,
    n_clusters: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Update centroids based on cluster assignments.
    
    Args:
        data: (N, D) array
        labels: (N,) cluster labels
        n_clusters: Number of clusters
    
    Returns:
        Tuple of (new_centroids, cluster_sizes)
    """
    N, D = data.shape
    centroids = np.zeros((n_clusters, D), dtype=np.float32)
    counts = np.zeros(n_clusters, dtype=np.int32)
    
    for i in range(N):
        k = labels[i]
        counts[k] += 1
        for d in range(D):
            centroids[k, d] += data[i, d]
    
    for k in range(n_clusters):
        if counts[k] > 0:
            for d in range(D):
                centroids[k, d] /= counts[k]
    
    return centroids, counts


# ==================== MINI-BATCH K-MEANS ====================

@njit(fastmath=True, cache=True)
def mini_batch_kmeans(
    data: np.ndarray,
    initial_centroids: np.ndarray,
    batch_size: int = 1000,
    max_iter: int = 100
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Mini-batch KMeans for large datasets.
    
    More efficient than standard KMeans for large N.
    
    Args:
        data: (N, D) array
        initial_centroids: (K, D) initial centroids
        batch_size: Mini-batch size
        max_iter: Maximum iterations
    
    Returns:
        Tuple of (centroids, labels)
    """
    centroids = initial_centroids.copy()
    N, D = data.shape
    K = centroids.shape[0]
    counts = np.ones(K, dtype=np.float32)
    
    for iteration in range(max_iter):
        # Sample mini-batch
        n_samples = min(batch_size, N)
        indices = np.random.choice(N, n_samples, replace=False)
        batch = data[indices]
        
        # Assign batch points
        batch_labels = assign_clusters(batch, centroids)
        
        # Update centroids with learning rate
        for i in range(len(batch)):
            k = batch_labels[i]
            counts[k] += 1
            eta = 1.0 / counts[k]
            
            for d in range(D):
                centroids[k, d] = (1.0 - eta) * centroids[k, d] + eta * batch[i, d]
    
    # Final assignment
    labels = assign_clusters(data, centroids)
    
    return centroids, labels


# ==================== FULL K-MEANS ====================

@njit(fastmath=True, cache=True)
def kmeans_full(
    data: np.ndarray,
    n_clusters: int,
    max_iter: int = 100,
    tol: float = 1e-4,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Full K-Means clustering with Numba JIT.
    
    Args:
        data: (N, D) array
        n_clusters: Number of clusters
        max_iter: Maximum iterations
        tol: Convergence tolerance
        random_state: Random seed
    
    Returns:
        Tuple of (centroids, labels, inertia)
    """
    # Initialize with K-Means++
    centroids = kmeans_plusplus_init(data, n_clusters, random_state)
    
    for iteration in range(max_iter):
        # Assign clusters
        labels = assign_clusters(data, centroids)
        
        # Update centroids
        new_centroids, counts = update_centroids(data, labels, n_clusters)
        
        # Handle empty clusters
        for k in range(n_clusters):
            if counts[k] == 0:
                idx = np.random.randint(0, data.shape[0])
                new_centroids[k] = data[idx]
        
        # Check convergence
        shift = 0.0
        for k in range(n_clusters):
            for d in range(data.shape[1]):
                diff = new_centroids[k, d] - centroids[k, d]
                shift += diff * diff
        
        centroids = new_centroids
        
        if shift < tol:
            break
    
    # Compute final inertia
    labels = assign_clusters(data, centroids)
    inertia = 0.0
    for i in range(data.shape[0]):
        k = labels[i]
        for d in range(data.shape[1]):
            diff = data[i, d] - centroids[k, d]
            inertia += diff * diff
    
    return centroids, labels, inertia


# ==================== PYTHON WRAPPER ====================

@dataclass
class KMeansResult:
    """Result of K-Means clustering."""
    centroids: np.ndarray
    labels: np.ndarray
    inertia: float
    n_iter: int


class KMeans:
    """
    K-Means clustering with Mini-batch optimization.
    
    Uses Mini-batch K-Means for efficiency on large datasets.
    
    Example:
        >>> kmeans = KMeans(n_clusters=100, batch_size=1000)
        >>> result = kmeans.fit(data)
        >>> labels = kmeans.predict(new_data)
    """
    
    def __init__(
        self,
        n_clusters: int = 100,
        batch_size: int = 1000,
        max_iter: int = 100,
        random_state: int = 42,
        use_mini_batch: bool = True
    ):
        """
        Initialize K-Means.
        
        Args:
            n_clusters: Number of clusters
            batch_size: Mini-batch size (for mini-batch mode)
            max_iter: Maximum iterations
            random_state: Random seed
            use_mini_batch: Use mini-batch K-Means
        """
        self.n_clusters = n_clusters
        self.batch_size = batch_size
        self.max_iter = max_iter
        self.random_state = random_state
        self.use_mini_batch = use_mini_batch
        
        self.centroids_: Optional[np.ndarray] = None
        self.labels_: Optional[np.ndarray] = None
        self.inertia_: Optional[float] = None
        self.n_iter_: int = 0
    
    def fit(self, data: np.ndarray) -> 'KMeans':
        """
        Fit K-Means to data.
        
        Args:
            data: (N, D) array of data points
        
        Returns:
            self
        """
        data = np.ascontiguousarray(data.astype(np.float32))
        
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        
        n_clusters = min(self.n_clusters, data.shape[0])
        
        # Initialize centroids
        initial_centroids = kmeans_plusplus_init(data, n_clusters, self.random_state)
        
        if self.use_mini_batch:
            self.centroids_, self.labels_ = mini_batch_kmeans(
                data, initial_centroids, self.batch_size, self.max_iter
            )
        else:
            self.centroids_, self.labels_, self.inertia_ = kmeans_full(
                data, n_clusters, self.max_iter, 1e-4, self.random_state
            )
        
        return self
    
    def predict(self, data: np.ndarray) -> np.ndarray:
        """
        Predict cluster labels for new data.
        
        Args:
            data: (N, D) array
        
        Returns:
            (N,) array of cluster labels
        """
        if self.centroids_ is None:
            raise RuntimeError("Must call fit() before predict()")
        
        data = np.ascontiguousarray(data.astype(np.float32))
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        
        return assign_clusters(data, self.centroids_)
    
    def fit_predict(self, data: np.ndarray) -> np.ndarray:
        """
        Fit and return labels.
        
        Args:
            data: (N, D) array
        
        Returns:
            (N,) array of cluster labels
        """
        self.fit(data)
        return self.labels_
    
    def transform(self, data: np.ndarray) -> np.ndarray:
        """
        Transform data to cluster-distance space.
        
        Args:
            data: (N, D) array
        
        Returns:
            (N, n_clusters) array of distances
        """
        if self.centroids_ is None:
            raise RuntimeError("Must call fit() before transform()")
        
        data = np.ascontiguousarray(data.astype(np.float32))
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        
        N = data.shape[0]
        K = self.n_clusters
        D = data.shape[1]
        
        distances = np.zeros((N, K), dtype=np.float32)
        
        for i in range(N):
            for k in range(K):
                dist = 0.0
                for d in range(D):
                    diff = data[i, d] - self.centroids_[k, d]
                    dist += diff * diff
                distances[i, k] = np.sqrt(dist)
        
        return distances


# Alias for backward compatibility
KMeansJIT = KMeans
