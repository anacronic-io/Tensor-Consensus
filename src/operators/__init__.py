"""Core tensor operators for multi-agent coordination."""

from .projection import adaptive_projection, AdaptiveProjection
from .assignment import differentiable_assignment, sinkhorn_assignment, DifferentiableAssignment
from .consensus import robust_consensus, RobustConsensus

__all__ = [
    "adaptive_projection",
    "AdaptiveProjection",
    "differentiable_assignment",
    "sinkhorn_assignment",
    "DifferentiableAssignment",
    "robust_consensus",
    "RobustConsensus",
]
