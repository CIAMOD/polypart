"""
Functions to generate predifined arrangements of hyperplanes.
"""

from functools import reduce
from math import gcd

import numpy as np

from .geometry import Hyperplane, Polytope
from .moduli import get_planes
from .utils import sample_intersecting_hyperplanes


def get_moduli_arrangement(
    n: int, r: int, d: int, use_epsilons: bool = True
) -> list[Hyperplane]:
    """Get arrangement of hyperplanes from moduli space construction.
    Dimension is n·(r-1) if use_epsilons is True, else n·r
    Args:
        n: number of parabolic points
        r: rank of vector bundles
        d: degree of vector bundles
        use_epsilons: whether to reduce dimensionality
    """
    planes = get_planes(n, r, d, use_epsilons=use_epsilons)
    hyperplanes = []
    for v, ks in planes:
        for k in ks:
            coeffs = np.append(v, k)
            hyperplanes.append(Hyperplane.from_coefficients(coeffs))
    return hyperplanes


def get_resonance_arrangement(d: int) -> list[Hyperplane]:
    """Get resonance arrangement in dimension d.
        Equivalently, for d ≥ 1 the resonance arrangement is
    Rd := {{c1x1+c2x2+· · ·+cdxd = 0} with ci ∈ {0, 1} and not all ci are zero}
    """
    if d < 1:
        raise ValueError("Dimension must be at least 1")
    hyperplanes = []
    for i in range(1, 2**d):
        coeffs = [(1 if (i & (1 << j)) else 0) for j in range(d)]
        hyperplanes.append(Hyperplane.from_coefficients(np.append(coeffs, 0)))
    return hyperplanes


def get_braid_arrangement(d: int) -> list[Hyperplane]:
    """Get braid arrangement in dimension d.
        For d ≥ 2 the braid arrangement is
    Bd := {{xi - xj = 0} for 1 ≤ i < j ≤ d}
    """
    if d < 2:
        raise ValueError("Dimension must be at least 2")
    hyperplanes = []
    for i in range(d):
        for j in range(i + 1, d):
            coeffs = [0] * d
            coeffs[i] = 1
            coeffs[j] = -1
            hyperplanes.append(Hyperplane.from_coefficients(np.append(coeffs, 0)))
    return hyperplanes


def get_random_arrangement(
    polytope: Polytope, m: int, decimals: int = None
) -> list[Hyperplane]:
    """Sample m hyperplanes intersecting the polytope.
    If decimals is set, limit denominators of normal coefficients to 10**decimals.
    """
    return sample_intersecting_hyperplanes(polytope, m, decimals=decimals)
