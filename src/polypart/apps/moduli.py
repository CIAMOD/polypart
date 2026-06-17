"""
Moduli space of parabolic vector bundles utilities.

See:
- Alfaya, D., et al. "On the isomorphisms between moduli spaces of parabolic
  vector bundles." (In preparation)
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Generator

import numpy as np


def generate_admissible_matrices_fixed_r_prime(
    n: int, r: int, r_prime: int, remove_even_symmetry: bool = False
) -> Generator[np.ndarray, None, None]:
    """Generate admissible n x r matrices with row sum r' and column sum <= 1."""
    combs = itertools.combinations(range(r), r_prime)
    variations = itertools.product(combs, repeat=n)
    N = math.comb(r, r_prime) ** n

    for i, variation in enumerate(variations):
        if remove_even_symmetry and r == 2 * r_prime and i >= N // 2:
            return
        n_ = np.zeros((n, r), dtype=int)
        for ii in range(n):
            for jj in variation[ii]:
                n_[ii, jj] = 1
        yield n_


def get_plane_intercept_bounds(w: np.ndarray) -> tuple[int, int]:
    """Get lower and upper bounds for stability wall intercepts."""
    w = w[:, ::-1]
    cumsums = np.cumsum(w, axis=1)
    cumsums = np.hstack((np.zeros_like(cumsums[:, :1]), cumsums))
    lower_bound = cumsums.min(axis=1).sum()
    upper_bound = cumsums.max(axis=1).sum()
    return lower_bound, upper_bound


def get_planes(
    n: int, r: int, d: int, use_epsilons: bool = False
) -> list[tuple[np.ndarray, list[int]]]:
    """Get stability walls for moduli space.

    Args:
        n: Number of parabolic points (>= 1).
        r: Rank of vector bundles.
        d: Degree of vector bundles.
        use_epsilons: Reduce dimension by n if True.

    Returns:
        List of (normal vector, list of valid intercepts) tuples.
    """
    if n < 1:
        raise ValueError("Number of parabolic points must be at least 1")

    planes = []
    for r_prime in range(1, r // 2 + 1):
        new_planes = []
        for n_ in generate_admissible_matrices_fixed_r_prime(n, r, r_prime, True):
            if use_epsilons:
                n_ = n_[:, 1:]
            v = r_prime - r * n_.flatten()
            lower, upper = get_plane_intercept_bounds(r_prime - r * n_)
            ks2 = [kp for kp in range(lower + 1, upper) if (kp + r_prime * d) % r == 0]
            if ks2:
                new_planes.append((v, ks2))
        planes += new_planes
    return planes


def pullback(
    alphas: np.ndarray, d: np.ndarray, sigma: list[int] | tuple[int, ...]
) -> tuple[np.ndarray, np.ndarray]:
    """Apply pullback transformation via permutation sigma."""
    return alphas[:, sigma, :], d


def hecke(
    alphas: np.ndarray, d: np.ndarray, H: list[int] | tuple[int, ...]
) -> tuple[np.ndarray, np.ndarray]:
    """Apply Hecke transformation with shift vector H."""
    rows, column_indices = np.ogrid[: alphas.shape[1], : alphas.shape[2]]
    shifts = -np.array(H) % alphas.shape[2]
    column_indices = column_indices - shifts[:, np.newaxis]

    new_alphas = alphas[:, rows, column_indices]
    new_alphas = new_alphas - new_alphas[:, :, 0:1]
    new_alphas[new_alphas < 0] += 1

    return new_alphas, (d - sum(H)) % alphas.shape[-1]


def dualization(alphas: np.ndarray, d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Apply dualization transformation."""
    alphas = 1 - np.flip(alphas, axis=-1)
    alphas = alphas - alphas[:, :, 0:1]
    return alphas, -d % alphas.shape[-1]


def basic_transformation(
    alphas: np.ndarray,
    d: np.ndarray,
    sigma: list[int] | tuple[int, ...],
    s: int,
    H: list[int] | tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray]:
    """Apply basic transformation: pullback, Hecke, then optional dualization.

    Args:
        alphas: Weight matrices of shape [batch, n, r].
        d: Degrees of shape [batch].
        sigma: Permutation of parabolic points.
        s: Sign (-1 for dualization, 1 otherwise).
        H: Hecke shift vector of length n.

    Returns:
        Transformed (alphas, d) tuple.
    """
    alphas, d = pullback(alphas, d, sigma)
    alphas, d = hecke(alphas, d, H)
    if s == -1:
        alphas, d = dualization(alphas, d)
    return alphas, d


def generate_d_invariant_transformations(
    n: int, r: int, d: int
) -> Generator[tuple[tuple[int, ...], int, tuple[int, ...]], None, None]:
    """Generate basic transformations that preserve degree.

    Yields:
        Tuples of (sigma, s, H) where sigma is a permutation,
        s is the dualization sign, and H is the Hecke vector.
    """
    for sigma in itertools.permutations(range(n)):
        for s in [1, -1] if -d % r == d else [1]:
            for H in itertools.product(range(r), repeat=n):
                if (d - sum(H)) % r != d:
                    continue
                yield sigma, s, H


def generate_d_invariant_transformations_no_pullback(
    n: int, r: int, d: int
) -> Generator[tuple[int], int, tuple[int]]:
    """
    Generator that yields all possible basic transformations that can lead to an automorphism.
    they must not change degree -> Hecke is restricted to sum(H) % r == 0

    A basic transformation is a tuple of the form (sigma, s, H) where:
    - sigma is a list of integers representing a permutation of the alphas
    - s is -1 if dualization is applied, 1 otherwise
    - H is a list of integers representing the Hecke operation for each alpha
    """
    for s in [1, -1] if -d % r == d else [1]:
        for H in itertools.product(range(r), repeat=n):
            if (d - sum(H)) % r != d:
                continue
            yield tuple(range(n)), s, H
