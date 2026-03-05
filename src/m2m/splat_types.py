"""
Data types for Gaussian Splatting.

This module defines the core data structures for representing
Gaussian splats and their embeddings.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import numpy as np


@dataclass
class GaussianSplat:
    """
    Represents a single Gaussian Splat.
    
    A Gaussian splat is a 3D ellipsoid with position, color, opacity,
    scale, and rotation that represents a small piece of a 3D scene.
    
    Attributes:
        id: Unique identifier for the splat
        position: 3D position (x, y, z) in world coordinates
        color: RGB color values, typically in [0, 1] or [0, 255]
        opacity: Transparency value in [0, 1]
        scale: Scale factors (sx, sy, sz) for the ellipsoid axes
        rotation: Quaternion rotation (w, x, y, z)
        sh_coeffs: Optional spherical harmonics coefficients for view-dependent color
    
    Example:
        >>> splat = GaussianSplat(
        ...     id=0,
        ...     position=np.array([1.0, 2.0, 3.0]),
        ...     color=np.array([0.8, 0.6, 0.4]),
        ...     opacity=0.95,
        ...     scale=np.array([0.01, 0.01, 0.01]),
        ...     rotation=np.array([1.0, 0.0, 0.0, 0.0])
        ... )
    """
    
    id: int
    position: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    color: np.ndarray = field(default_factory=lambda: np.ones(3, dtype=np.float32))
    opacity: float = 1.0
    scale: np.ndarray = field(default_factory=lambda: np.ones(3, dtype=np.float32))
    rotation: np.ndarray = field(default_factory=lambda: np.array([1, 0, 0, 0], dtype=np.float32))
    sh_coeffs: Optional[np.ndarray] = None  # Spherical harmonics
    
    def __post_init__(self):
        """Ensure all arrays have correct dtype."""
        self.position = np.asarray(self.position, dtype=np.float32)
        self.color = np.asarray(self.color, dtype=np.float32)
        self.scale = np.asarray(self.scale, dtype=np.float32)
        self.rotation = np.asarray(self.rotation, dtype=np.float32)
        
        # Normalize quaternion
        norm = np.linalg.norm(self.rotation)
        if norm > 0:
            self.rotation = self.rotation / norm
    
    @property
    def covariance_3d(self) -> np.ndarray:
        """
        Compute the 3D covariance matrix from scale and rotation.
        
        Returns:
            3x3 covariance matrix
        """
        R = self._quaternion_to_matrix(self.rotation)
        S = np.diag(self.scale)
        M = R @ S
        return M @ M.T
    
    @staticmethod
    def _quaternion_to_matrix(q: np.ndarray) -> np.ndarray:
        """Convert quaternion to rotation matrix."""
        w, x, y, z = q
        return np.array([
            [1 - 2*(y*y + z*z), 2*(x*y + w*z), 2*(x*z - w*y)],
            [2*(x*y - w*z), 1 - 2*(x*x + z*z), 2*(y*z + w*x)],
            [2*(x*z + w*y), 2*(y*z - w*x), 1 - 2*(x*x + y*y)]
        ], dtype=np.float32)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'id': self.id,
            'position': self.position.tolist(),
            'color': self.color.tolist(),
            'opacity': self.opacity,
            'scale': self.scale.tolist(),
            'rotation': self.rotation.tolist(),
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GaussianSplat':
        """Create from dictionary."""
        return cls(
            id=data['id'],
            position=np.array(data['position'], dtype=np.float32),
            color=np.array(data['color'], dtype=np.float32),
            opacity=data['opacity'],
            scale=np.array(data['scale'], dtype=np.float32),
            rotation=np.array(data['rotation'], dtype=np.float32),
        )


@dataclass
class SplatEmbedding:
    """
    Embedding vector for a Gaussian Splat.
    
    The embedding is a 640-dimensional vector composed of:
    - Position encoding (64 dims): Sinusoidal encoding of 3D position
    - Color encoding (512 dims): Histogram-based color representation
    - Attribute encoding (64 dims): Opacity, scale, rotation features
    
    This embedding enables fast similarity search using HRM2.
    
    Attributes:
        splat_id: ID of the corresponding splat
        position_encoding: 64D positional embedding
        color_encoding: 512D color histogram embedding
        attribute_encoding: 64D attribute embedding
    
    Example:
        >>> embedding = SplatEmbedding(
        ...     splat_id=0,
        ...     position_encoding=pos_enc,
        ...     color_encoding=color_enc,
        ...     attribute_encoding=attr_enc
        ... )
        >>> full = embedding.full_embedding  # 640D vector
    """
    
    splat_id: int
    position_encoding: np.ndarray = field(default_factory=lambda: np.zeros(64, dtype=np.float32))
    color_encoding: np.ndarray = field(default_factory=lambda: np.zeros(512, dtype=np.float32))
    attribute_encoding: np.ndarray = field(default_factory=lambda: np.zeros(64, dtype=np.float32))
    
    @property
    def full_embedding(self) -> np.ndarray:
        """
        Concatenate all encodings into a single 640D vector.
        
        Returns:
            640-dimensional embedding vector
        """
        return np.concatenate([
            self.position_encoding,
            self.color_encoding,
            self.attribute_encoding
        ])
    
    @property
    def embedding_dim(self) -> int:
        """Total embedding dimension."""
        return 640


@dataclass
class SplatCluster:
    """
    A cluster of similar splats.
    
    Used in HRM2 for hierarchical organization.
    
    Attributes:
        id: Cluster identifier
        centroid: Center of the cluster in embedding space
        splat_ids: List of splat IDs in this cluster
        bounds: Bounding box (min, max) in 3D space
    """
    
    id: int
    centroid: np.ndarray
    splat_ids: list = field(default_factory=list)
    bounds: tuple = field(default_factory=lambda: (np.zeros(3), np.zeros(3)))
    
    @property
    def size(self) -> int:
        """Number of splats in cluster."""
        return len(self.splat_ids)
    
    def contains_point(self, point: np.ndarray) -> bool:
        """Check if a point is within the cluster bounds."""
        min_b, max_b = self.bounds
        return np.all(point >= min_b) and np.all(point <= max_b)


# Type aliases for clarity
SplatID = int
ClusterID = int
EmbeddingVector = np.ndarray
