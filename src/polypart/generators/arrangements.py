"""Generators for predefined hyperplane arrangements."""

from __future__ import annotations

from functools import reduce

import numpy as np

from polypart.apps.moduli import get_planes
from polypart.core.geometry import Arrangement, Hyperplane, Polytope
from polypart.core.typing import Fraction
from polypart.utils.sampling import (
    _random_nonfullzero_coefficients,
    _sample_unit_normal,
    sample_point_in_polytope,
)


def get_moduli_arrangement(
    n: int, r: int, d: int, use_epsilons: bool = True
) -> Arrangement:
    """Get hyperplane arrangement from moduli space construction.

    Args:
        n: Number of parabolic points.
        r: Rank of vector bundles.
        d: Degree of vector bundles.
        use_epsilons: Reduce dimensionality to n*(r-1) if True, else n*r.

    Returns:
        Arrangement of hyperplanes defining the moduli space structure.
    """
    planes = get_planes(n, r, d, use_epsilons=use_epsilons)
    hyperplanes = []
    for v, ks in planes:
        for k in ks:
            coeffs = np.append(v, k)
            hyperplanes.append(Hyperplane.from_coefficients(coeffs))
    return Arrangement(hyperplanes)


def get_resonance_arrangement(d: int) -> Arrangement:
    """Get resonance arrangement in dimension d.

    The resonance arrangement R_d consists of hyperplanes
    {c_1*x_1 + ... + c_d*x_d = 0} with c_i in {0, 1}, not all zero.

    Returns:
        Arrangement of resonance hyperplanes.
    """
    if d < 1:
        raise ValueError("Dimension must be at least 1")
    hyperplanes = []
    for i in range(1, 2**d):
        coeffs = [(1 if (i & (1 << j)) else 0) for j in range(d)]
        hyperplanes.append(Hyperplane.from_coefficients(np.append(coeffs, 0)))
    return Arrangement(hyperplanes)


def get_braid_arrangement(d: int) -> Arrangement:
    """Get braid arrangement in dimension d.

    The braid arrangement B_d consists of hyperplanes {x_i - x_j = 0}
    for 1 <= i < j <= d. This is the dual of the permutohedron.

    Returns:
        Arrangement of braid hyperplanes.
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
    return Arrangement(hyperplanes)


def combine_hyperplanes(
    indices: list[int],
    hyperplanes: list[Hyperplane],
    polytope: Polytope,
    decimals: int | None = None,
    dim: int | None = None,
    rng: np.random.Generator = None,
    size: int | None = None,
) -> np.ndarray:
    """Combine hyperplanes into a new hyperplane passing through a point in polytope."""
    if rng is None:
        rng = np.random.default_rng()

    coeffs_list = [hyperplanes[i].as_coefficients() for i in indices]
    normals = [c[:-1] for c in coeffs_list]
    offsets = [c[-1] for c in coeffs_list]

    p = sample_point_in_polytope(polytope, decimals, rng)

    r = []
    for n, b in zip(normals, offsets):
        dot = sum(n[j] * p[j] for j in range(dim))
        r.append(dot - b)

    while True:
        alpha = _random_nonfullzero_coefficients(rng, size, decimals)

        if any(ri != 0 for ri in r):
            s = sum(ri * ai for ri, ai in zip(r, alpha))
            if s != 0:
                pivot = next(i for i, ri in enumerate(r) if ri != 0)
                alpha[pivot] -= s / r[pivot]

        if not all(ai == 0 for ai in alpha):
            combined_coeffs = reduce(
                lambda a, b: a + b,
                [coeffs_list[i] * alpha[i] for i in range(size)],
            )
            if not all(c == 0 for c in combined_coeffs[:-1]):
                break

    return combined_coeffs


def get_random_arrangement(
    polytope: Polytope,
    m: int,
    degen_ratio: float = 0.0,
    decimals: int | None = None,
    seed: int | None = None,
) -> Arrangement:
    """Sample m hyperplanes intersecting the polytope.

    Args:
        polytope: Polytope to be intersected.
        m: Number of hyperplanes to sample.
        degen_ratio: Fraction of degenerate (linearly dependent) hyperplanes.
        decimals: Limit denominators to 10**decimals if set.
        seed: Random seed for reproducibility.

    Returns:
        Arrangement of sampled hyperplanes.
    """
    hyperplanes: list[Hyperplane] = []
    dim = polytope.A.shape[1]
    rng = np.random.default_rng(seed)

    if polytope._vertices is None:
        polytope.extreme()
    vertices = polytope.vertices

    while len(hyperplanes) < m:
        # Degenerate hyperplane: linear combination passing through random point
        if len(hyperplanes) >= dim and rng.random() < degen_ratio:
            size = rng.integers(2, dim) if dim > 2 else 2
            indices = rng.choice(len(hyperplanes), size=size, replace=False)
            combined_coeffs = combine_hyperplanes(
                indices, hyperplanes, polytope, decimals, dim, rng, size
            )
            hyperplanes.append(Hyperplane.from_coefficients(combined_coeffs))
            continue

        # General position hyperplane
        normal = _sample_unit_normal(dim, rng, decimals)
        values = vertices @ normal
        lbound, ubound = np.min(values), np.max(values)

        offset = None
        current_decimals = decimals
        while offset is None:
            offset = Fraction(rng.uniform(float(lbound), float(ubound)))
            if current_decimals is not None:
                offset = offset.limit_denominator(10**current_decimals)
                if offset <= lbound or offset >= ubound:
                    offset = None
                    current_decimals += 1
                    print(f"Warning: increasing decimals to {current_decimals}")

        coefficients = np.append(normal, offset)
        hyperplanes.append(Hyperplane.from_coefficients(coefficients))

    return Arrangement(hyperplanes)
