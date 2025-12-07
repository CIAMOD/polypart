from functools import reduce
from math import gcd
from typing import Optional

import cdd.gmp  # rational backend (pycddlib >= 3 uses cdd.gmp for GMP rationals)
import numpy as np

from .ftyping import Fraction as F
from .ftyping import FractionMatrix, FractionVector, as_fraction_vector
from .geometry import Hyperplane, Polytope


def sample_point_in_polytope(
    polytope: Polytope, decimals: int = None, rng: np.random.Generator = None
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
        if rng is None:
            rng = np.random.default_rng()
        point = as_fraction_vector(rng.uniform(mins, maxs))
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


def _random_nonfullzero_coefficients(
    rng: np.random.Generator,
    size: int,
    decimals: Optional[int],
) -> list[F]:
    """Generate random coefficients not all zero."""
    random_coeffs = []
    while all(c == 0 for c in random_coeffs):
        if decimals is None:
            random_coeffs = [F(float(rng.uniform(-1, 1))) for _ in range(size)]
        else:
            random_coeffs = [
                F(float(rng.uniform(-1, 1))).limit_denominator(10**decimals)
                for _ in range(size)
            ]
    return random_coeffs


def _sample_unit_normal(d: int, rng: np.random.Generator, decimals: Optional[int]):
    """Sample isotropic unit normal, then convert to Fractions."""
    v = rng.normal(size=d)
    v = v / np.linalg.norm(v)
    return as_fraction_vector(v, decimals)


def sample_intersecting_hyperplanes(
    polytope: Polytope, m: int, decimals: int = None, seed: int = None
) -> list[Hyperplane]:
    hyperplanes = []
    dim = polytope.A.shape[1]

    rng = np.random.default_rng(seed)

    # Compute vertices
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
                    offset = None  # resample
                    decimals += 1  # increase precision to avoid infinite loop
                    print(
                        "Warning: increasing decimals to",
                        decimals,
                        "for offset sampling",
                    )

        coefficients = np.append(normal, offset)
        coefficients = _simplify_coefficients(coefficients)
        hyperplanes.append(Hyperplane.from_coefficients(coefficients))

    return hyperplanes


def _positively_spanning_cdd(normals: FractionMatrix) -> bool:
    """
    Exact feasibility test (GMP rationals) for:
        normals.T w = 0
        sum_i w_i = 1
        w_i >= 0

    Uses cdd's LP convention: each row [b, a1, ..., am] encodes
        0 <= b + a1*w1 + ... + am*wm    (i.e., a·w >= -b).

    Returns True iff the system is feasible, i.e. 0 ∈ conv{normals}.
    For continuous sampling, feasibility implies 0 is in the interior
    almost surely, hence the normals positively span R^d.
    """
    m, d = normals.shape
    normals = np.asarray(normals, dtype=object)

    rows = []

    # (1) w_i >= 0  ->  0 <= 0 + w_i
    for i in range(m):
        coeffs = [F(0)] * m
        coeffs[i] = F(1)
        rows.append([F(0)] + coeffs)

    # (2) sum w_i = 1 as two inequalities
    rows.append([F(-1)] + [F(1)] * m)
    rows.append([F(1)] + [F(-1)] * m)

    # (3) normals.T w = 0 as two inequalities per coordinate j
    for j in range(d):
        col = [F(normals[i, j]) for i in range(m)]
        rows.append([F(0)] + col)
        rows.append([F(0)] + [-c for c in col])

    # Objective row for feasibility: maximise 0.
    obj_row = [F(0)] + [F(0)] * m

    lp = cdd.gmp.linprog_from_array(rows + [obj_row], obj_type=cdd.LPObjType.MAX)

    try:
        cdd.gmp.linprog_solve(lp, solver=cdd.LPSolverType.DUAL_SIMPLEX)
    except Exception:
        return False

    # If infeasible, cdd typically raises or returns no primal solution.
    sol = lp.primal_solution
    return sol is not None and len(sol) == m
