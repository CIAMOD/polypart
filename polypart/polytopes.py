"""
Functions to generate predifined polytopes.
"""

import numpy as np

from .ftyping import Fraction, NumberLike, as_fraction_vector
from .geometry import Polytope
from .moduli import get_simplex_inequalities


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


def get_random_polytope(
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
