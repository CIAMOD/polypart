"""Generators for standard polytopes."""

from __future__ import annotations

import numpy as np

from polypart.core.geometry import Polytope
from polypart.core.typing import Fraction, NumberLike, as_fraction_vector
from polypart.utils.sampling import _sample_unit_normal


def get_simplex_inequalities(
    n: int, r: int, use_epsilons: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """Get inequalities for product of n simplices of dimension r.

    Args:
        n: Number of simplices.
        r: Dimension of each simplex.
        use_epsilons: If True, reduce dimension by n.

    Returns:
        Tuple of (A, b) defining Ax <= b.
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


def get_hypercube(d: int) -> Polytope:
    """Return the d-dimensional unit hypercube [0,1]^d."""
    A = np.vstack((np.eye(d), -np.eye(d)))
    b = np.hstack((np.ones(d), np.zeros(d)))
    return Polytope(A, b)


def get_centered_hypercube(d: int, r: int) -> Polytope:
    """Return the d-dimensional centered hypercube [-r,r]^d."""
    A = np.vstack((np.eye(d), -np.eye(d)))
    b = np.ones(2 * d) * r
    return Polytope(A, b)


def get_simplex(d: int) -> Polytope:
    """Return the d-dimensional standard simplex 0 <= x1 <= ... <= xd <= 1."""
    A, b = get_simplex_inequalities(1, d, use_epsilons=False)
    return Polytope(A, b)


def get_product_of_simplices(n: int, d: int) -> Polytope:
    """Return the product of n d-dimensional simplices."""
    A, b = get_simplex_inequalities(n, d, use_epsilons=False)
    return Polytope(A, b)


def sample_circumscribed_polytope(
    d: int, m: int, radius: NumberLike = 1.0, seed: int | None = None
) -> Polytope:
    """Generate random polytope circumscribed about a sphere.

    Intersects m random supporting halfspaces tangent to the sphere.

    Args:
        d: Dimension.
        m: Number of halfspaces (must be >= d+1).
        radius: Sphere radius.
        seed: Random seed.

    Returns:
        Random circumscribed polytope.
    """
    if m < d + 1:
        raise ValueError(f"Need at least d+1={d + 1} halfspaces, got m={m}.")

    radius = Fraction(radius)
    rng = np.random.default_rng(seed)

    while True:
        normals = rng.normal(size=(m, d))
        normals /= np.linalg.norm(normals, axis=1, keepdims=True)

        A = normals
        b = radius * as_fraction_vector(np.ones(m))
        P = Polytope(A, b)
        try:
            P.extreme()
        except ValueError:
            continue
        return P


def sample_poisson_zero_cell_polytope(
    d: int,
    intensity: float,
    window_radius: float,
    decimals: int | None = None,
    max_tries: int = 10_000,
    seed: int | None = None,
) -> Polytope:
    """Sample zero cell of Poisson hyperplane tessellation.

    Approximates the zero cell by restricting to hyperplanes intersecting
    a ball of given radius and keeping the cell containing the origin.

    Args:
        d: Dimension.
        intensity: Poisson intensity parameter.
        window_radius: Ball radius for hyperplane filtering.
        decimals: Limit denominators to 10**decimals if set.
        max_tries: Maximum sampling attempts.
        seed: Random seed.

    Returns:
        Random Poisson zero cell polytope.
    """
    if d < 1:
        raise ValueError("d must be >= 1.")
    if intensity <= 0:
        raise ValueError("intensity must be positive.")
    if window_radius <= 0:
        raise ValueError("window_radius must be positive.")

    rng = np.random.default_rng(seed)
    R = float(window_radius)
    intensity = float(intensity)

    tries = 0
    while tries < max_tries:
        tries += 1

        lambda_window = 2.0 * intensity * R
        N = rng.poisson(lam=lambda_window)
        if N < d + 1:
            continue

        normals = np.empty((N, d), dtype=object)
        offsets = np.empty(N, dtype=object)

        for i in range(N):
            normals[i, :] = _sample_unit_normal(d, rng, decimals)
            t = rng.uniform(low=-R, high=R)
            t_frac = Fraction(float(t))
            if decimals is not None:
                t_frac = t_frac.limit_denominator(10**decimals)
            offsets[i] = t_frac

        # Orient halfspaces so origin is inside
        A = np.empty((N, d), dtype=object)
        b = np.empty(N, dtype=object)
        for i in range(N):
            sign = 1 if offsets[i] >= 0 else -1
            A[i, :] = sign * normals[i, :]
            b[i] = sign * offsets[i]

        Q = Polytope(A, b)
        try:
            Q.extreme()
        except ValueError:
            continue

        return Q

    raise RuntimeError(f"Failed to obtain a bounded polytope after {max_tries} tries.")
