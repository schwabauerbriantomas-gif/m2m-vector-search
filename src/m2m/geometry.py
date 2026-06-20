"""
Sphere manifold (S^d) helpers for M2M.

Implements standard Riemannian operations on the unit hypersphere:
- normalize_sphere: project to S^d
- geodesic_distance: arc-length between two points
- exp_map: tangent vector → point on sphere (geodesic shooting)
- log_map: point on sphere → tangent vector at base point
- project_to_tangent: project Euclidean vector to tangent space at base point

Reference: Absil, Mahony, Sepulchre, "Optimization Algorithms on Matrix
Manifolds" (2008), Chapter 3.
"""

import numpy as np


def normalize_sphere(x):
    """Normalize vectors to unit hypersphere S^(d-1)."""
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)


def geodesic_distance(x, y):
    """
    Geodesic distance on S^d: the great-circle arc length.

    d(x, y) = arccos(clip(x·y, -1+ε, 1-ε))

    Handles both single vectors [D] and batches [N, D].
    """
    dot = np.sum(x * y, axis=-1)
    dot = np.clip(dot, -1.0 + 1e-7, 1.0 - 1e-7)
    return np.arccos(dot)


def project_to_tangent(base, v):
    """
    Project vector v onto the tangent space T_base S^d.

    Formula: v_proj = v - (v · base) * base

    The tangent space at base ∈ S^d is the hyperplane orthogonal to base.
    This removes the radial component, leaving only the tangential part.

    Args:
        base: point on S^d, shape [D] or [N, D]
        v: vector(s) to project, same shape as base

    Returns:
        Projected tangent vector(s), same shape
    """
    # Component of v along base
    dot = np.sum(v * base, axis=-1, keepdims=True)
    return v - dot * base


def exp_map(base, v):
    r"""
    Exponential map on S^d: shoot a geodesic from base in direction v.

    Given base ∈ S^d and tangent vector v ∈ T_base S^d:

        ‖v‖ = θ  (geodesic distance)
        v_hat = v / ‖v‖

        exp_base(v) = cos(θ) · base + sin(θ) · v_hat

    For ‖v‖ < ε, returns base unchanged (identity for zero tangent).

    Args:
        base: point on S^d, shape [D] or [N, D]
        v: tangent vector at base, same shape

    Returns:
        Point on S^d reached by following the geodesic for distance ‖v‖.
    """
    theta = np.linalg.norm(v, axis=-1, keepdims=True)  # [1] or [N, 1]

    # Near-zero tangent: geodesic is identity
    small_mask = theta < 1e-10

    # Safe division (avoid NaN where theta≈0)
    theta_safe = np.where(small_mask, 1.0, theta)
    v_hat = v / (theta_safe + 1e-30)

    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    result = cos_t * base + sin_t * v_hat

    # Where tangent was ~zero, return base
    return np.where(small_mask, base, result)


def log_map(base, target):
    r"""
    Logarithmic map on S^d: find the tangent vector at base that reaches target.

    Inverse of exp_map. Given base, target ∈ S^d:

        v_dir = target - (target · base) * base   # project to tangent
        ‖v_dir‖ = sin(θ)  where θ = geodesic distance

        v = θ / sin(θ) · v_dir

    where θ = arccos(clip(base · target, ...)).

    For θ < ε (base ≈ target), returns zero vector.

    Args:
        base: point on S^d, shape [D] or [N, D]
        target: point on S^d, same shape

    Returns:
        Tangent vector at base pointing toward target, same shape
    """
    theta = geodesic_distance(base, target)  # [1] or [N]
    theta = np.expand_dims(theta, axis=-1)  # [1, 1] or [N, 1]

    small_mask = theta < 1e-10

    # Tangent direction: project target onto tangent space at base
    v_dir = project_to_tangent(base, target)  # already in tangent space

    sin_theta = np.sin(theta)
    sin_theta_safe = np.where(small_mask, 1.0, sin_theta)

    scale = np.where(small_mask, 0.0, theta / (sin_theta_safe + 1e-30))

    return scale * v_dir
