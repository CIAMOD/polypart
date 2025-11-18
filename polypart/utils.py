"""
Utility functions for polytopes and hyperplanes.
"""

import numpy as np

from .geometry import Polytope


def sample_point_in_polytope(polytope: Polytope) -> np.ndarray:
    """Sample a point uniformly at random from the polytope using rejection sampling."""
    mins = np.min(polytope.vertices, axis=0).astype(float)
    maxs = np.max(polytope.vertices, axis=0).astype(float)

    assert np.all(maxs > mins), "Polytope has no volume."

    while True:
        point = np.random.uniform(mins, maxs)
        if polytope.contains(point):
            return point


def compute_hyperplane_from_points(
    points: np.ndarray, decimals: int = None
) -> tuple[np.ndarray, float] | None:
    """
    Compute hyperplane from d points in d-dimensional space.
    Returns (normal, offset) or None if degenerate.
    """
    P = np.asarray(points, dtype=float)

    if P.ndim != 2 or P.shape[0] != P.shape[1]:
        raise ValueError("points must be a (d, d) array: d points in d-D space")

    p0 = P[0]
    V = P[1:] - p0  # (d-1, d); rows span the hyperplane
    if np.linalg.matrix_rank(V) < P.shape[1] - 1:
        return None

    _, _, vh = np.linalg.svd(V)
    normal = vh[-1]
    norm = np.linalg.norm(normal)
    if norm == 0:
        return None

    normal /= norm
    offset = float(normal @ p0)

    # Round normal and offset to get prettier numbers
    if decimals is not None:
        normal = np.round(normal, decimals=decimals)
        offset = np.round(offset, decimals=decimals)

    return normal, offset
