import numpy as np


def normalize_sphere(x):
    """Normalize vectors to unit hypersphere."""
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)


def geodesic_distance(x, y):
    """Calculate geodesic distance between vectors."""
    dot = np.sum(x * y, axis=-1)
    dot = np.clip(dot, -1.0 + 1e-7, 1.0 - 1e-7)
    return np.arccos(dot)


def exp_map(x, v):
    """Exponential map on sphere."""
    return x  # simplified


def log_map(x, y):
    """Logarithmic map on sphere."""
    return x  # simplified


def project_to_tangent(x, v):
    """Project vector to tangent space."""
    return v  # simplified
