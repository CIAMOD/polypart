"""
Functions to generate predifined arrangements of hyperplanes.
"""

from functools import reduce
from math import gcd

import numpy as np

from .ftyping import Fraction
from .geometry import Hyperplane, Polytope
from .moduli import get_planes
from .utils import (
    _random_nonfullzero_coefficients,
    _sample_unit_normal,
    sample_point_in_polytope,
)


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
    This arrangement is the dual of the permutohedron.
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


# def get_random_arrangement(
#     polytope: Polytope,
#     m: int,
#     degen_ratio: float = 0.0,
#     decimals: int = None,
#     seed: int = None,
# ) -> list[Hyperplane]:
#     """Sample m hyperplanes intersecting the polytope.

#     Args:
#         polytope: Polytope to be intersected.
#         m: number of hyperplanes to sample.
#         degen_ratio: degeneracy ratio of arrangement.
#         decimals: if set, limit denominators to 10**decimals.
#         seed: random seed for reproducibility.
#     """
#     hyperplanes = []
#     dim = polytope.A.shape[1]

#     rng = np.random.default_rng(seed)

#     # Compute vertices
#     vertices = polytope.vertices

#     while len(hyperplanes) < m:
#         # Degenerate hyperplane
#         if len(hyperplanes) >= dim and rng.random() < degen_ratio:
#             size = rng.integers(2, dim) if dim > 2 else 2
#             indices = rng.choice(len(hyperplanes), size=size, replace=False)
#             # Linearly combine existing hyperplanes
#             coeffs_list = [hyperplanes[i].as_coefficients() for i in indices]
#             random_coeffs = _random_nonfullzero_coefficients(rng, size, decimals)
#             combined_coeffs = reduce(
#                 lambda a, b: a + b,
#                 [coeffs_list[i] * random_coeffs[i] for i in range(size)],
#             )
#             hyperplanes.append(Hyperplane.from_coefficients(combined_coeffs))
#             continue

#         # Sample random hyperplane
#         normal = _sample_unit_normal(dim, rng, decimals)

#         values = vertices @ normal
#         lbound, ubound = np.min(values), np.max(values)
#         offset = None
#         while offset is None:
#             offset = Fraction(rng.uniform(float(lbound), float(ubound)))
#             if decimals is not None:
#                 offset = offset.limit_denominator(10**decimals)
#                 if offset <= lbound or offset >= ubound:
#                     offset = None  # resample
#                     decimals += 1  # increase precision to avoid infinite loop
#                     print(
#                         "Warning: increasing decimals to",
#                         decimals,
#                         "for offset sampling",
#                     )

#         coefficients = np.append(normal, offset)
#         hyperplanes.append(Hyperplane.from_coefficients(coefficients))

#     return hyperplanes


def combine_hyperplanes(
    indices: list[int],
    hyperplanes: list[Hyperplane],
    polytope: Polytope,
    decimals: int = None,
    dim: int = None,
    rng: np.random.Generator = np.random.default_rng(),
    size: int = None,
) -> Hyperplane:
    # Coefficients [n | b] for chosen hyperplanes (all rational)
    coeffs_list = [hyperplanes[i].as_coefficients() for i in indices]

    # Split into normals and offsets
    normals = [c[:-1] for c in coeffs_list]  # list of length-d arrays
    offsets = [c[-1] for c in coeffs_list]  # list of scalars

    # Sample a rational point p ∈ P
    p = sample_point_in_polytope(polytope, decimals, rng)

    # r_i = n_i · p − b_i (all exact rationals)
    r = []
    for n, b in zip(normals, offsets):
        dot = sum(n[j] * p[j] for j in range(dim))
        r.append(dot - b)

    # Sample coefficients α_i and enforce Σ r_i α_i = 0
    while True:
        alpha = _random_nonfullzero_coefficients(rng, size, decimals)

        # If at least one r_i ≠ 0, enforce the constraint
        if any(ri != 0 for ri in r):
            s = sum(ri * ai for ri, ai in zip(r, alpha))  # Σ r_i α_i
            if s != 0:
                # Choose a pivot index with r_pivot ≠ 0
                pivot = next(i for i, ri in enumerate(r) if ri != 0)
                # Adjust α_pivot so that Σ r_i α_i = 0 holds exactly
                alpha[pivot] -= s / r[pivot]

        # Reject if all α_i are zero
        if not all(ai == 0 for ai in alpha):
            # Form combined coefficients [n | b]
            combined_coeffs = reduce(
                lambda a, b: a + b,
                [coeffs_list[i] * alpha[i] for i in range(size)],
            )
            # Reject if normal is zero
            if not all(c == 0 for c in combined_coeffs[:-1]):
                break

    return combined_coeffs


def get_random_arrangement(
    polytope: Polytope,
    m: int,
    degen_ratio: float = 0.0,
    decimals: int = None,
    seed: int = None,
) -> list[Hyperplane]:
    """Sample m hyperplanes intersecting the polytope.

    Args:
        polytope: Polytope to be intersected.
        m: number of hyperplanes to sample.
        degen_ratio: degeneracy ratio of arrangement.
        decimals: if set, limit denominators to 10**decimals.
        seed: random seed for reproducibility.
    """
    hyperplanes = []
    dim = polytope.A.shape[1]

    rng = np.random.default_rng(seed)

    # Compute vertices (for general-position sampling)
    vertices = polytope.vertices

    while len(hyperplanes) < m:
        # Degenerate hyperplane: linear combination of existing ones,
        # constrained to pass through a random rational point in P.
        if len(hyperplanes) >= dim and rng.random() < degen_ratio:
            size = rng.integers(2, dim) if dim > 2 else 2
            indices = rng.choice(len(hyperplanes), size=size, replace=False)
            combined_coeffs = combine_hyperplanes(
                indices,
                hyperplanes,
                polytope,
                decimals,
                dim,
                rng,
                size,
            )
            hyperplanes.append(Hyperplane.from_coefficients(combined_coeffs))
            continue

        # Sample random hyperplane in general position, intersecting P
        normal = _sample_unit_normal(dim, rng, decimals)  # rational normal

        values = vertices @ normal
        lbound, ubound = np.min(values), np.max(values)
        offset = None
        while offset is None:
            offset = Fraction(rng.uniform(float(lbound), float(ubound)))
            if decimals is not None:
                offset = offset.limit_denominator(10**decimals)
                if offset <= lbound or offset >= ubound:
                    offset = None  # resample
                    decimals += 1  # increase precision to avoid infinite loop
                    print(
                        "Warning: increasing decimals to",
                        decimals,
                        "for offset sampling",
                    )

        coefficients = np.append(normal, offset)
        hyperplanes.append(Hyperplane.from_coefficients(coefficients))

    return hyperplanes
