"""
Functions to generate predifined polytopes.
"""

import numpy as np

from .ftyping import Fraction, FractionVector, NumberLike, as_fraction_vector
from .geometry import Polytope
from .moduli import get_simplex_inequalities
from .utils import _sample_unit_normal, _simplify_coefficients


# Define two examples of polytope, one is the hypercube and the other is a simplex
def get_hypercube(d: int) -> Polytope:
    """Return the d-dimensional hypercube as a Polytope. 0 <= xi <= 1 for all i."""
    A = np.vstack((np.eye(d), -np.eye(d)))
    b = np.hstack((np.ones(d), np.zeros(d)))
    return Polytope(A, b)


def get_centered_hypercube(d: int, r: int) -> Polytope:
    """Return the d-dimensional centered hypercube as a Polytope. -r <= xi <= r for all i."""
    A = np.vstack((np.eye(d), -np.eye(d)))
    b = np.ones(2 * d) * r
    return Polytope(A, b)


def get_simplex(d: int) -> Polytope:
    """Return the d-dimensional simplex as a Polytope. 0 <= x1 <= x2 <= ... <= xd <= 1"""
    A, b = get_simplex_inequalities(1, d, use_epsilons=False)
    return Polytope(A, b)


def get_product_of_simplices(n: int, d: int) -> Polytope:
    """Return the product of n d-dimensional simplices as a Polytope.
    0 <= x1_1 <= ... <= x1_d <= 1
    0 <= x2_1 <= ... <= x2_d <= 1
    ...
    0 <= xn_1 <= ... <= xn_d <= 1
    """
    A, b = get_simplex_inequalities(n, d, use_epsilons=False)
    return Polytope(A, b)


def sample_circumscribed_polytope(
    d: int, m: int, radius: NumberLike = 1.0, seed: int = None
) -> Polytope:
    """
    Generate a random circumscribed polytope in R^d using the boundary model on
    the convex body K = S^{d-1} of radius `radius`, and compute its vertices.

    The construction intersects m random supporting halfspaces tangent to the
    sphere and repeats the sampling until a bounded polytope is obtained.
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
            P.extreme()  # raises if infeasible or unbounded
        except ValueError:
            continue

        return P


def sample_zero_cell_polytope(
    d: int,
    m: int,
    offset_scale: float = 1.0,
    decimals: int = None,
    max_tries: int = 10_000,
    seed: int = None,
) -> Polytope:
    """
    Standard intersection-of-random-halfspaces model in R^d, using exact GMP rationals.

    Steps:
        1) Sample m i.i.d. unit normals u_i (here isotropic via normalised Gaussian).
            Convert to bounded-denominator Fractions if decimals is set.
        2) Sample m i.i.d. positive offsets b_i from abs(N(0, offset_scale^2)).
            Convert to bounded-denominator Fractions if decimals is set.
            Define halfspaces H_i := {x : <u_i, x> <= b_i }.
        3) Define polytope Q := intersection of H_i. (Almost surely bounded.)

    """
    if m < d + 1:
        raise ValueError(f"Need at least d+1={d + 1} halfspaces, got m={m}.")

    rng = np.random.default_rng(seed)
    offset_scale = float(offset_scale)

    tries = 0
    while tries < max_tries:
        tries += 1

        # Sample normals as Fractions
        normals = np.empty((m, d), dtype=object)
        for i in range(m):
            normals[i, :] = _sample_unit_normal(d, rng, decimals)

        # Sample offsets as positive Fractions
        offsets = np.empty(m, dtype=object)
        for i in range(m):
            offset = rng.normal(loc=0.0, scale=offset_scale)
            offset = Fraction(abs(float(offset)))
            if decimals is not None:
                offset = offset.limit_denominator(10**decimals)
            offsets[i] = offset

        # Define polytope
        A = normals
        b = offsets
        Q = Polytope(A, b)
        try:
            Q.extreme()  # raises if infeasible or unbounded
        except ValueError:
            continue
        if tries > 1:
            print(f"Polytope obtained after {tries} tries.")
        return Q
    raise ValueError(f"Failed to obtain a bounded polytope after {max_tries} tries.")


def sample_poisson_zero_cell_polytope(
    d: int,
    intensity: float,
    window_radius: float,
    decimals: int | None = None,
    max_tries: int = 10_000,
    seed: int | None = None,
) -> Polytope:
    """
    Approximate the zero cell of a stationary isotropic Poisson hyperplane
    tessellation in R^d, by restricting to hyperplanes intersecting a ball
    of radius `window_radius` and keeping the cell containing the origin.

    Model:
        - Hyperplanes: { x : <u, x> = t }.
        - Directions u: isotropic on S^{d-1} (via normalised Gaussian).
        - Signed distances t | {hyperplane hits B(0, R)} ~ Uniform[-R, R].
        - Number N of such hyperplanes is Poisson with mean
              lambda_window = c * intensity * R
          where c>0 is an arbitrary constant absorbed into 'intensity'.
          Here we set lambda_window = 2 * intensity * R.

    Construction:
        - Sample N hyperplanes intersecting B(0, R).
        - For each, orient the halfspace so that it contains the origin.
        - Intersect all oriented halfspaces to obtain a polytope Q.
        - Reject if Q is infeasible or unbounded; otherwise return Q.

    As window_radius -> infinity, this converges (in law) to the true Poisson
    hyperplane zero cell, provided intensity and directional distribution are
    kept fixed.
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

        # Poisson number of hyperplanes intersecting B(0, R).
        lambda_window = 2.0 * intensity * R
        N = rng.poisson(lam=lambda_window)
        if N < d + 1:
            continue

        normals = np.empty((N, d), dtype=object)
        offsets = np.empty(N, dtype=object)

        for i in range(N):
            # Direction
            normals[i, :] = _sample_unit_normal(d, rng, decimals)

            # Signed distance t | {hit B(0,R)} ~ Uniform[-R, R]
            t = rng.uniform(low=-R, high=R)
            t_frac = Fraction(float(t))
            if decimals is not None:
                t_frac = t_frac.limit_denominator(10**decimals)
            offsets[i] = t_frac

        # Orient halfspaces so that 0 is inside
        A = np.empty((N, d), dtype=object)
        b = np.empty(N, dtype=object)

        for i in range(N):
            sign = 1 if offsets[i] >= 0 else -1
            A[i, :] = sign * normals[i, :]
            b[i] = sign * offsets[i]

        Q = Polytope(A, b)
        try:
            Q.extreme()  # raises if infeasible or unbounded
        except ValueError:
            continue

        # if tries > 1:
        #     print(f"Poisson zero-cell polytope obtained after {tries} tries (N={N}).")
        return Q

    raise RuntimeError(f"Failed to obtain a bounded polytope after {max_tries} tries.")
