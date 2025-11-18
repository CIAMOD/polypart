"""
Functions to generate predifined arrangements of hyperplanes.
"""

from math import gcd
from functools import reduce

import numpy as np

from .geometry import Polytope, Hyperplane
from .moduli import get_planes
from .utils import sample_point_in_polytope, compute_hyperplane_from_points


def get_random_arrangement(
    polytope: Polytope, m: int, decimals: int = None
) -> list[Hyperplane]:
    """Get m random hyperplanes intersecting the polytope.
    If decimals is set, round coefficients to that many decimals so
    that hyperplanes have nicer coefficients and reduce coefficients
    fractions as much as possible.
    (for small polytopes it will take longer and could get stuck).
    """
    hyperplanes = []
    dim = polytope.A.shape[1]
    while len(hyperplanes) < m:
        points = np.array([sample_point_in_polytope(polytope) for _ in range(dim)])
        result = compute_hyperplane_from_points(points, decimals=decimals)
        if result is None:
            print("Degenerate points, resampling...")
            continue  # Degenerate points, resample

        hyperplane = Hyperplane.from_coefficients([*result[0], result[1]])

        if not polytope.intersecting_hyperplanes([hyperplane])[0]:
            continue  # Does not intersect, resample

        hyperplanes.append(hyperplane)

    if decimals is not None:
        # Find gcd of all denominators and multiply all coefficients by it
        for i, h in enumerate(hyperplanes):
            coeffs = np.append(h.normal, h.offset)
            denominators = [frac.denominator for frac in coeffs if frac != 0]
            common_denom = reduce(gcd, denominators)
            new_coeffs = coeffs * common_denom
            hyperplanes[i] = Hyperplane.from_coefficients(new_coeffs)

    return hyperplanes


def get_moduli_arrangement(n: int, r: int, d: int) -> list[Hyperplane]:
    """Get arrangement of hyperplanes from moduli space construction."""
    planes = get_planes(n, r, d, use_epsilons=True)
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
