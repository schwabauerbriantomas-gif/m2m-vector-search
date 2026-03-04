"""
Encoding functions optimized with Numba JIT.

This module provides fast encoding functions for converting
Gaussian splat attributes into embedding vectors for indexing.
"""

import numpy as np
from typing import Tuple, Optional

try:
    from numba import njit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    # Fallback decorators
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        if len(args) == 1 and callable(args[0]):
            return args[0]
        return decorator
    prange = range


# ==================== POSITION ENCODING ====================

@njit(fastmath=True, cache=True)
def _sinusoidal_position_encoding_numba(
    positions: np.ndarray,
    dim: int = 64
) -> np.ndarray:
    """
    Sinusoidal Position Encoding for 3D coordinates.
    
    Uses multi-frequency sinusoids similar to NeRF positional encoding.
    
    Args:
        positions: (N, 3) array of 3D positions
        dim: Output dimension (must be divisible by 6)
    
    Returns:
        (N, dim) array of position encodings
    """
    N = positions.shape[0]
    encodings = np.zeros((N, dim), dtype=np.float32)
    
    # Compute normalization bounds
    min_x = positions[0, 0]
    max_x = positions[0, 0]
    min_y = positions[0, 1]
    max_y = positions[0, 1]
    min_z = positions[0, 2]
    max_z = positions[0, 2]
    
    for i in range(1, N):
        if positions[i, 0] < min_x:
            min_x = positions[i, 0]
        if positions[i, 0] > max_x:
            max_x = positions[i, 0]
        if positions[i, 1] < min_y:
            min_y = positions[i, 1]
        if positions[i, 1] > max_y:
            max_y = positions[i, 1]
        if positions[i, 2] < min_z:
            min_z = positions[i, 2]
        if positions[i, 2] > max_z:
            max_z = positions[i, 2]
    
    range_x = max_x - min_x + 1e-8
    range_y = max_y - min_y + 1e-8
    range_z = max_z - min_z + 1e-8
    
    n_freq = dim // 6
    
    for i in prange(N):
        # Normalize to [0, 1]
        x = (positions[i, 0] - min_x) / range_x
        y = (positions[i, 1] - min_y) / range_y
        z = (positions[i, 2] - min_z) / range_z
        
        for d in range(n_freq):
            freq = 2.0 ** d
            idx = d * 6
            
            encodings[i, idx] = np.sin(x * freq)
            encodings[i, idx + 1] = np.cos(x * freq)
            encodings[i, idx + 2] = np.sin(y * freq)
            encodings[i, idx + 3] = np.cos(y * freq)
            encodings[i, idx + 4] = np.sin(z * freq)
            encodings[i, idx + 5] = np.cos(z * freq)
    
    return encodings


class SinusoidalPositionEncoder:
    """
    Encoder for 3D positions using sinusoidal functions.
    
    Similar to positional encoding in Transformers and NeRF.
    """
    
    def __init__(self, dim: int = 64):
        """
        Initialize encoder.
        
        Args:
            dim: Output dimension (will be adjusted to be divisible by 6)
        """
        # Ensure dim is divisible by 6 (for x, y, z sin/cos)
        self.dim = (dim // 6) * 6
        if self.dim < 6:
            self.dim = 6
    
    def encode(self, positions: np.ndarray) -> np.ndarray:
        """
        Encode 3D positions.
        
        Args:
            positions: (N, 3) or (3,) array
        
        Returns:
            (N, dim) or (dim,) array
        """
        positions = np.asarray(positions, dtype=np.float32)
        if positions.ndim == 1:
            positions = positions.reshape(1, -1)
        
        result = _sinusoidal_position_encoding_numba(positions, self.dim)
        
        if result.shape[0] == 1:
            return result[0]
        return result


# ==================== COLOR ENCODING ====================

@njit(fastmath=True, cache=True)
def _color_histogram_encoding_numba(
    colors: np.ndarray,
    n_bins: int = 8
) -> np.ndarray:
    """
    Histogram-based color encoding.
    
    Creates a sparse histogram with Gaussian-smoothed contributions.
    
    Args:
        colors: (N, 3) array of RGB colors in [0, 1]
        n_bins: Number of bins per channel (8 -> 512 dims)
    
    Returns:
        (N, 512) array of color encodings
    """
    N = colors.shape[0]
    n_bins_cubed = n_bins * n_bins * n_bins  # 512 for n_bins=8
    encodings = np.zeros((N, 512), dtype=np.float32)
    
    for i in prange(N):
        r, g, b = colors[i]
        
        # Quantize to bins
        bin_r = min(int(r * n_bins), n_bins - 1)
        bin_g = min(int(g * n_bins), n_bins - 1)
        bin_b = min(int(b * n_bins), n_bins - 1)
        
        # Create histogram with Gaussian smoothing
        idx = 0
        for br in range(n_bins):
            for bg in range(n_bins):
                for bb in range(n_bins):
                    dr = float(br - bin_r)
                    dg = float(bg - bin_g)
                    db = float(bb - bin_b)
                    
                    # Gaussian kernel
                    dist_sq = dr*dr + dg*dg + db*db
                    weight = np.exp(-dist_sq / 4.0)
                    
                    if idx < 512:
                        encodings[i, idx] = weight
                    idx += 1
    
    return encodings


class ColorHistogramEncoder:
    """
    Encoder for RGB colors using histogram representation.
    """
    
    def __init__(self, n_bins: int = 8):
        """
        Initialize encoder.
        
        Args:
            n_bins: Bins per channel (output dim = n_bins^3)
        """
        self.n_bins = n_bins
        self.dim = n_bins ** 3  # 512 for n_bins=8
    
    def encode(self, colors: np.ndarray) -> np.ndarray:
        """
        Encode RGB colors.
        
        Args:
            colors: (N, 3) array in [0, 1] or [0, 255]
        
        Returns:
            (N, 512) array
        """
        colors = np.asarray(colors, dtype=np.float32)
        if colors.ndim == 1:
            colors = colors.reshape(1, -1)
        
        # Normalize to [0, 1]
        if colors.max() > 1.0:
            colors = colors / 255.0
        
        return _color_histogram_encoding_numba(colors, self.n_bins)


# ==================== ATTRIBUTE ENCODING ====================

@njit(fastmath=True, cache=True)
def _attribute_encoding_numba(
    opacities: np.ndarray,
    scales: np.ndarray,
    rotations: np.ndarray
) -> np.ndarray:
    """
    Attribute encoding for opacity, scale, and rotation.
    
    Creates hand-crafted features from splat attributes.
    
    Args:
        opacities: (N,) array
        scales: (N, 3) array
        rotations: (N, 4) array (quaternions)
    
    Returns:
        (N, 64) array of attribute encodings
    """
    N = opacities.shape[0]
    encodings = np.zeros((N, 64), dtype=np.float32)
    
    for i in prange(N):
        o = opacities[i]
        
        # Opacity features (8 dims)
        encodings[i, 0] = o
        encodings[i, 1] = o * o
        encodings[i, 2] = o * o * o
        encodings[i, 3] = np.sqrt(o + 1e-8)
        encodings[i, 4] = np.log(o + 1e-8)
        encodings[i, 5] = 1.0 - o
        encodings[i, 6] = 1.0 if o > 0.5 else 0.0
        encodings[i, 7] = 1.0 if o < 0.5 else 0.0
        
        # Scale features (24 dims)
        sx, sy, sz = scales[i]
        encodings[i, 8] = sx
        encodings[i, 9] = sy
        encodings[i, 10] = sz
        encodings[i, 11] = sx * sy * sz
        encodings[i, 12] = (sx + sy + sz) / 3.0
        encodings[i, 13] = np.sqrt(sx*sx + sy*sy + sz*sz)
        encodings[i, 14] = sx / (sy + 1e-8)
        encodings[i, 15] = sx / (sz + 1e-8)
        encodings[i, 16] = sy / (sz + 1e-8)
        encodings[i, 17] = sx * sx
        encodings[i, 18] = sy * sy
        encodings[i, 19] = sz * sz
        encodings[i, 20] = np.log(sx + 1e-8)
        encodings[i, 21] = np.log(sy + 1e-8)
        encodings[i, 22] = np.log(sz + 1e-8)
        # 23-31: zero padding
        
        # Rotation features (32 dims)
        qw, qx, qy, qz = rotations[i]
        encodings[i, 32] = qw
        encodings[i, 33] = qx
        encodings[i, 34] = qy
        encodings[i, 35] = qz
        encodings[i, 36] = qw * qw
        encodings[i, 37] = qx * qx
        encodings[i, 38] = qy * qy
        encodings[i, 39] = qz * qz
        encodings[i, 40] = qw * qx
        encodings[i, 41] = qw * qy
        encodings[i, 42] = qw * qz
        encodings[i, 43] = qx * qy
        encodings[i, 44] = qx * qz
        encodings[i, 45] = qy * qz
        # 46-63: zero padding
    
    return encodings


class AttributeEncoder:
    """
    Encoder for splat attributes (opacity, scale, rotation).
    """
    
    def __init__(self, dim: int = 64):
        """Initialize encoder."""
        self.dim = dim
    
    def encode(
        self,
        opacities: np.ndarray,
        scales: np.ndarray,
        rotations: np.ndarray
    ) -> np.ndarray:
        """
        Encode splat attributes.
        
        Args:
            opacities: (N,) array
            scales: (N, 3) array
            rotations: (N, 4) array
        
        Returns:
            (N, 64) array
        """
        opacities = np.asarray(opacities, dtype=np.float32)
        scales = np.asarray(scales, dtype=np.float32)
        rotations = np.asarray(rotations, dtype=np.float32)
        
        if opacities.ndim == 0:
            opacities = opacities.reshape(1)
            scales = scales.reshape(1, 3)
            rotations = rotations.reshape(1, 4)
        
        return _attribute_encoding_numba(opacities, scales, rotations)


# ==================== FULL EMBEDDING ====================

class FullEmbeddingBuilder:
    """
    Builder for complete 640D splat embeddings.
    """
    
    def __init__(self):
        """Initialize all encoders."""
        self.pos_encoder = SinusoidalPositionEncoder(dim=64)  # Will use 60
        self.color_encoder = ColorHistogramEncoder(n_bins=8)   # 512 dims
        self.attr_encoder = AttributeEncoder(dim=64)          # 64 dims
        
        # Actual dimensions
        self._pos_dim = self.pos_encoder.dim
        self._color_dim = self.color_encoder.dim
        self._attr_dim = 64
        self._total_dim = self._pos_dim + self._color_dim + self._attr_dim
    
    @property
    def embedding_dim(self) -> int:
        """Total embedding dimension."""
        return self._total_dim
    
    def build(
        self,
        positions: np.ndarray,
        colors: np.ndarray,
        opacities: np.ndarray,
        scales: np.ndarray,
        rotations: np.ndarray
    ) -> np.ndarray:
        """
        Build full 640D embeddings.
        
        Args:
            positions: (N, 3) positions
            colors: (N, 3) colors
            opacities: (N,) opacities
            scales: (N, 3) scales
            rotations: (N, 4) quaternions
        
        Returns:
            (N, 640) embeddings
        """
        pos_enc = self.pos_encoder.encode(positions)
        color_enc = self.color_encoder.encode(colors)
        attr_enc = self.attr_encoder.encode(opacities, scales, rotations)
        
        if pos_enc.ndim == 1:
            return np.concatenate([pos_enc, color_enc, attr_enc])
        
        return np.concatenate([pos_enc, color_enc, attr_enc], axis=1)


# Convenience function
def build_full_embedding(
    positions: np.ndarray,
    colors: np.ndarray,
    opacities: np.ndarray,
    scales: np.ndarray,
    rotations: np.ndarray
) -> np.ndarray:
    """
    Build full 640D embedding for splats.
    
    Convenience function using FullEmbeddingBuilder.
    """
    builder = FullEmbeddingBuilder()
    return builder.build(positions, colors, opacities, scales, rotations)
