"""
Functions to get inequalities of ambient space and stability walls
for moduli spaces of parabolic vector bundles.
"""

import itertools
import math

import numpy as np


def get_simplex_inequalities(n: int, r: int, use_epsilons=False):
    """Get inequalities for product of n simplices of dimension r.
    If use_epsilons is True, the dimension is reduced by n.
    """
    if use_epsilons:
        r -= 1
    A = np.zeros((n * (r + 1), n * r), dtype=int)
    b = np.zeros(n * (r + 1), dtype=int)
    for i in range(n):
        for j in range(r):
            A[i * (r + 1) + j, i * r + j] = -1
            A[i * (r + 1) + j + 1, i * r + j] = 1
    for i in range(n):
        b[i * (r + 1) : (i + 1) * (r + 1)] = 0
        b[i * (r + 1) + r] = 1
    return A, b


def generate_admissible_matrices_fixed_r_prime(
    n: int, r: int, r_prime: int, remove_even_symetry: bool = False
):
    """Generate all admissible n x r matrices with row sum r' and column sum <= 1.
    If remove_even_symetry is True and r == 2 * r', only return half
    of the matrices to avoid duplicates under n -> n, r' -> r - r'.
    """
    combs = itertools.combinations(range(r), r_prime)
    variations = itertools.product(combs, repeat=n)
    N = math.comb(r, r_prime) ** n
    for i, variation in enumerate(variations):
        if remove_even_symetry and r == 2 * r_prime and i >= N // 2:
            return
        n_ = np.zeros((n, r), dtype=int)
        for ii in range(n):
            for jj in variation[ii]:
                n_[ii, jj] = 1
        yield n_


def get_plane_intercept_bounds(w: np.ndarray):
    """Get lower and upper bounds for intercepts of stability walls."""
    w = w[:, ::-1]
    cumsums = np.cumsum(w, axis=1)
    cumsums = np.hstack((np.zeros_like(cumsums[:, :1]), cumsums))
    lower_bound = cumsums.min(axis=1).sum()
    upper_bound = cumsums.max(axis=1).sum()
    return lower_bound, upper_bound


def get_planes(
    n: int, r: int, d: int, use_epsilons=False
) -> list[tuple[np.ndarray, list[int]]]:
    """Get stability walls for moduli space of n points with rank r and degree d.
    If use_epsilons is True, dimension is reduced by n.
    """
    planes = []
    for r_prime in range(1, r // 2 + 1):
        new_planes = []
        for n_ in generate_admissible_matrices_fixed_r_prime(n, r, r_prime, True):
            if use_epsilons:
                n_ = n_[:, 1:]
            v = r_prime - r * n_.flatten()
            lower, upper = get_plane_intercept_bounds(r_prime - r * n_)
            ks2 = [kp for kp in range(lower + 1, upper) if (kp + r_prime * d) % r == 0]
            if len(ks2) > 0:
                new_planes.append((v, ks2))
        planes += new_planes
    return planes
