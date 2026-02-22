"""
Core module for M2M Gaussian Splatting.
"""

from .splat_types import GaussianSplat, SplatEmbedding
from .encoding import (
    SinusoidalPositionEncoder,
    ColorHistogramEncoder,
    AttributeEncoder,
    FullEmbeddingBuilder,
)
from .clustering import KMeansJIT
from .hrm2_engine import HRM2Engine

__all__ = [
    "GaussianSplat",
    "SplatEmbedding",
    "SinusoidalPositionEncoder",
    "ColorHistogramEncoder",
    "AttributeEncoder",
    "FullEmbeddingBuilder",
    "KMeansJIT",
    "HRM2Engine",
]
