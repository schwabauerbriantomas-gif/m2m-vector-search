"""
KMeans clustering using scikit-learn acceleration.

This module provides fast K-Means clustering implementation
optimized for Gaussian splat embeddings using Mini-batch approach.
"""

import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass
from sklearn.cluster import MiniBatchKMeans, KMeans as SKLearnKMeans

@dataclass
class KMeansResult:
    """Result of K-Means clustering."""
    centroids: np.ndarray
    labels: np.ndarray
    inertia: float
    n_iter: int


class KMeans:
    """
    K-Means clustering utilizing scikit-learn for optimization.
    """
    
    def __init__(
        self,
        n_clusters: int = 100,
        batch_size: int = 10000,
        max_iter: int = 100,
        random_state: int = 42,
        use_mini_batch: bool = True
    ):
        self.n_clusters = n_clusters
        self.batch_size = batch_size
        self.max_iter = max_iter
        self.random_state = random_state
        self.use_mini_batch = use_mini_batch
        
        self.centroids_: Optional[np.ndarray] = None
        self.labels_: Optional[np.ndarray] = None
        self.inertia_: float = 0.0
        self.n_iter_: int = 0
        
        # We will initialize the model under the hood
        if self.use_mini_batch:
            self._model = MiniBatchKMeans(
                n_clusters=n_clusters,
                batch_size=batch_size,
                max_iter=max_iter,
                random_state=random_state,
                n_init='auto'
            )
        else:
            self._model = SKLearnKMeans(
                n_clusters=n_clusters,
                max_iter=max_iter,
                random_state=random_state,
                n_init='auto'
            )
    
    def fit(self, data: np.ndarray) -> 'KMeans':
        data = np.ascontiguousarray(data.astype(np.float32))
        if data.ndim == 1:
            data = data.reshape(-1, 1)
            
        n_samples = data.shape[0]
        actual_clusters = min(self.n_clusters, n_samples)
        
        # Dynamically adjust model if clusters exceed samples
        if actual_clusters != self._model.n_clusters:
            if self.use_mini_batch:
                self._model = MiniBatchKMeans(
                    n_clusters=actual_clusters,
                    batch_size=min(self.batch_size, n_samples),
                    max_iter=self.max_iter,
                    random_state=self.random_state,
                    n_init='auto'
                )
            else:
                self._model = SKLearnKMeans(
                    n_clusters=actual_clusters,
                    max_iter=self.max_iter,
                    random_state=self.random_state,
                    n_init='auto'
                )
                
        self._model.fit(data)
        
        self.centroids_ = self._model.cluster_centers_
        self.labels_ = self._model.labels_
        self.inertia_ = self._model.inertia_
        self.n_iter_ = self._model.n_iter_
        
        return self
    
    def predict(self, data: np.ndarray) -> np.ndarray:
        if self.centroids_ is None:
            raise RuntimeError("Must call fit() before predict()")
            
        data = np.ascontiguousarray(data.astype(np.float32))
        if data.ndim == 1:
            data = data.reshape(-1, 1)
            
        return self._model.predict(data)
    
    def fit_predict(self, data: np.ndarray) -> np.ndarray:
        self.fit(data)
        return self.labels_
    
    def transform(self, data: np.ndarray) -> np.ndarray:
        if self.centroids_ is None:
            raise RuntimeError("Must call fit() before transform()")
            
        data = np.ascontiguousarray(data.astype(np.float32))
        if data.ndim == 1:
            data = data.reshape(-1, 1)
            
        return self._model.transform(data)


# Alias for backward compatibility
KMeansJIT = KMeans
assign_clusters = lambda data, centroids: np.argmin(np.linalg.norm(data[:, None] - centroids, axis=2), axis=1)
