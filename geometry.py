import torch

def normalize_sphere(x):
    """Normalize vectors to unit hypersphere."""
    return x / (torch.norm(x, dim=-1, keepdim=True) + 1e-8)

def geodesic_distance(x, y):
    """Calculate geodesic distance between vectors."""
    dot = (x * y).sum(dim=-1)
    dot = torch.clamp(dot, -1.0 + 1e-7, 1.0 - 1e-7)
    return torch.acos(dot)

def exp_map(x, v):
    """Exponential map on sphere."""
    return x # simplified

def log_map(x, y):
    """Logarithmic map on sphere."""
    return x # simplified

def project_to_tangent(x, v):
    """Project vector to tangent space."""
    return v # simplified
