"""Sampling utilities for rational arithmetic."""

from functools import reduce
from math import gcd

import numpy as np

from polypart.core.geometry import Arrangement, Hyperplane, Polytope
from polypart.core.typing import Fraction as F
from polypart.core.typing import FractionVector, as_fraction_vector


def _simplify_coefficients(coeffs: FractionVector) -> FractionVector:
    """Simplify coefficients by dividing by GCD of denominators."""
    denominators = [frac.denominator for frac in coeffs if frac != 0]
    if not denominators:
        return coeffs
    common_denom = reduce(gcd, denominators)
    return coeffs * common_denom


def _random_nonfullzero_coefficients(
    rng: np.random.Generator, size: int, decimals: int | None
) -> list[F]:
    """Generate random coefficients where not all are zero."""
    random_coeffs: list[F] = []
    while all(c == 0 for c in random_coeffs):
        if decimals is None:
            random_coeffs = [F(float(rng.uniform(-1, 1))) for _ in range(size)]
        else:
            random_coeffs = [
                F(float(rng.uniform(-1, 1))).limit_denominator(10**decimals)
                for _ in range(size)
            ]
    return random_coeffs


def _sample_unit_normal(
    d: int, rng: np.random.Generator, decimals: int | None
) -> FractionVector:
    """Sample isotropic unit normal, convert to Fractions."""
    v = rng.normal(size=d)
    v = v / np.linalg.norm(v)
    return as_fraction_vector(v, decimals)


def sample_point_in_polytope(
    polytope: Polytope,
    decimals: int | None = None,
    rng: np.random.Generator | None = None,
) -> FractionVector:
    """Sample a point uniformly at random from the polytope using rejection sampling."""
    if decimals is not None:
        assert decimals >= 0, "decimals must be non-negative"
        if polytope._vertices is None:
            polytope.extreme()
        assert int(-np.log10(float(polytope.diameter))) + 1 <= decimals, (
            f"decimals={decimals} too small for polytope diameter={polytope.diameter}"
        )

    if polytope._vertices is None:
        polytope.extreme()
    mins = np.min(polytope.vertices, axis=0).astype(float)  # Shape [dim]
    maxs = np.max(polytope.vertices, axis=0).astype(float)

    if rng is None:
        rng = np.random.default_rng()

    while True:
        point = as_fraction_vector(rng.uniform(mins, maxs))
        if decimals is not None:
            point = as_fraction_vector(
                [frac.limit_denominator(10**decimals) for frac in point]
            )
        if polytope.contains(point):
            return point


def sample_intersecting_hyperplanes(
    polytope: Polytope, m: int, decimals: int | None = None, seed: int | None = None
) -> Arrangement:
    """Sample m hyperplanes that intersect the polytope.

    Returns:
        Arrangement of sampled hyperplanes.
    """
    hyperplanes: list[Hyperplane] = []
    dim = polytope.A.shape[1]
    rng = np.random.default_rng(seed)
    vertices = polytope.vertices

    while len(hyperplanes) < m:
        normal = _sample_unit_normal(dim, rng, decimals)
        values = vertices @ normal
        lbound, ubound = np.min(values), np.max(values)
        offset = None
        while offset is None:
            offset = F(np.random.uniform(float(lbound), float(ubound)))
            if decimals is not None:
                offset = offset.limit_denominator(10**decimals)
                if offset <= lbound or offset >= ubound:
                    offset = None
                    decimals += 1
                    print(f"Warning: increasing decimals to {decimals}")

        coefficients = np.append(normal, offset)
        coefficients = _simplify_coefficients(coefficients)
        hyperplanes.append(Hyperplane.from_coefficients(coefficients))

    return Arrangement(hyperplanes)
