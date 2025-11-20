import sys
from functools import reduce
from math import gcd
from pathlib import Path

import numpy as np

from .ftyping import FractionVector, as_fraction_vector
from .geometry import Hyperplane, Polytope


def sample_point_in_polytope(
    polytope: Polytope, decimals: int = None
) -> FractionVector:
    """Sample a point uniformly at random from the polytope using rejection sampling."""
    # Lower bound on decimals based on diameter of the polytope
    if decimals is not None:
        assert decimals >= 0, "decimals must be non-negative"
        assert int(-np.log10(float(polytope.diameter))) + 1 <= decimals, (
            f"decimals = {decimals} is too small for the polytope's diameter = {polytope.diameter}"
        )

    # Get bounding box of the polytope
    mins = np.min(polytope.vertices, axis=0).astype(float)  # shape (dim,)
    maxs = np.max(polytope.vertices, axis=0).astype(float)  # shape (dim,)

    while True:
        # Sample a random point in the bounding box
        point = as_fraction_vector(np.random.uniform(mins, maxs))
        # Limit denominator of mpq fractions to avoid very large numbers
        if decimals is not None:
            point = as_fraction_vector(
                [frac.limit_denominator(10**decimals) for frac in point]
            )
        # Check if the point is in the polytope
        if polytope.contains(point):
            return point


def _simplify_coefficients(coeffs: FractionVector) -> FractionVector:
    """Simplify the coefficients by dividing by their GCD of denominators."""
    denominators = [frac.denominator for frac in coeffs if frac != 0]
    if not denominators:
        return coeffs
    common_denom = reduce(gcd, denominators)
    new_coeffs = coeffs * common_denom
    return new_coeffs


def sample_intersecting_hyperplanes(
    polytope: Polytope, m: int, decimals: int = None
) -> list[Hyperplane]:
    hyperplanes = []
    dim = polytope.A.shape[1]

    while len(hyperplanes) < m:
        point = sample_point_in_polytope(polytope, decimals=4)
        normal = np.random.normal(size=dim)
        normal = normal / np.linalg.norm(normal)  # Normalize to unit length
        normal = as_fraction_vector(normal)
        # limit denominator of mpq fractions to avoid very large numbers
        if decimals is not None:
            normal = as_fraction_vector(
                [frac.limit_denominator(10**decimals) for frac in normal]
            )
        offset = np.dot(normal, point)
        coefficients = np.append(normal, offset)
        coefficients = _simplify_coefficients(coefficients)
        hyperplanes.append(Hyperplane.from_coefficients(coefficients))

    return hyperplanes
