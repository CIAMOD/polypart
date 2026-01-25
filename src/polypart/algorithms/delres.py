"""Deletion-Restriction algorithm for counting arrangement regions."""

from __future__ import annotations

from typing import Sequence

import numpy as np
from cdd.gmp import LPSolverType

from polypart.core.geometry import (
    Arrangement,
    HalfSpace,
    Hyperplane,
    Polyhedron,
)
from polypart.core.typing import Fraction as F
from polypart.utils.solvers import kFaceCDDBackend


def rref(A: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Compute reduced row echelon form."""
    m, n = A.shape
    row = 0
    pivots: list[int] = []

    for col in range(n):
        if row == m:
            break
        col_slice = A[row:, col]
        nz = np.nonzero(col_slice != 0)[0]
        if nz.size == 0:
            continue
        piv = row + nz[0]

        if piv != row:
            A[[row, piv]] = A[[piv, row]]
            b[[row, piv]] = b[[piv, row]]

        inv = F(1, 1) / A[row, col]
        A[row, :] = A[row, :] * inv
        b[row] = b[row] * inv

        for r in range(m):
            if r != row:
                factor = A[r, col]
                if factor != 0:
                    A[r, :] = A[r, :] - factor * A[row, :]
                    b[r] = b[r] - factor * b[row]

        pivots.append(col)
        row += 1

    return A, b, pivots


def nullspace_basis_from_rref(
    A_rref: np.ndarray, pivots: list[int], dim: int
) -> np.ndarray:
    """Compute nullspace basis from RREF."""
    pivot_row_for_col = {p: i for i, p in enumerate(pivots)}
    free_cols = [j for j in range(dim) if j not in pivot_row_for_col]
    k = len(free_cols)
    if k == 0:
        return np.zeros((dim, 0), dtype=object)

    B = np.zeros((dim, k), dtype=object)
    for idx, f in enumerate(free_cols):
        B[f, idx] = F(1, 1)
        for p, r in pivot_row_for_col.items():
            B[p, idx] = -A_rref[r, f]
    return B


def particular_solution_from_rref(
    b_rref: np.ndarray, pivots: list[int], dim: int
) -> np.ndarray:
    """Extract particular solution from RREF."""
    x = np.zeros(dim, dtype=object)
    for r, p in enumerate(pivots):
        x[p] = b_rref[r]
    return x


def _is_zero_vec(v) -> bool:
    return all(x == 0 for x in v)


def _canonical_affine(alpha: list[F], c: F) -> tuple:
    if _is_zero_vec(alpha):
        return ("const", 0) if c == 0 else ("const", "empty")
    v = list(alpha) + [c]
    p = next(i for i, z in enumerate(v) if z != 0)
    return ("affine", tuple(z / v[p] for z in v))


def _prepare_LI(
    hyperplanes: list[tuple[np.ndarray, F]],
    I: list[int],  # noqa: E741
) -> tuple[bool, np.ndarray | None, np.ndarray | None]:
    """Represent subspace L_I = intersection of hyperplanes in I
    in the form {x0 + B*y | y in R^k}.
    Returns (is_empty, x0, B) where is_empty indicates if L_I is empty,
    x0 is a particular solution, and B is a basis for the nullspace.
    """
    dim = hyperplanes[0][0].shape[0]  # dimension
    if not I:
        x0 = np.zeros(dim, dtype=object)
        B = np.eye(dim, dtype=object)
        return False, x0, B
    A = np.array([hyperplanes[i][0] for i in I], dtype=object)
    b = np.array([hyperplanes[i][1] for i in I], dtype=object)
    A_rref, b_rref, pivots = rref(A, b)
    m, d = A_rref.shape
    for r in range(m):
        if all(A_rref[r, c] == 0 for c in range(d)) and b_rref[r] != 0:
            return True, None, None
    x0 = particular_solution_from_rref(b_rref, pivots, dim)
    B = nullspace_basis_from_rref(A_rref, pivots, dim)
    if B.size == 0:
        B = np.zeros((dim, 0), dtype=object)
    return False, x0, B


def _restrict_key(
    hyperplanes: list[tuple[np.ndarray, F]],
    x0: np.ndarray,
    B: np.ndarray,
    j: int,
) -> tuple:
    """Get canonical key for hyperplane j restricted to L_I."""
    a, b = hyperplanes[j]
    if B.shape[1] == 0:
        c = b - (a @ x0)
        return _canonical_affine([], c)
    alpha = (a @ B).tolist()
    c = b - (a @ x0)
    return _canonical_affine(alpha, c)


def _canonical_fullspace(h: tuple[np.ndarray, F]) -> tuple:
    a, b = h
    v = list(a) + [b]
    if _is_zero_vec(v):
        return ("zero",)
    p = next(i for i, z in enumerate(v) if z != 0)
    return tuple([z / v[p] for z in v])


def intersects_support(
    j: int,
    I: list[int],  # noqa: E741
    lp_solver: kFaceCDDBackend,
) -> bool:
    """Check if hyperplane j intersects W = intersection of hyperplanes in I."""
    lp_result = lp_solver.solve(eq_idx=I + [j], solver=LPSolverType.DUAL_SIMPLEX)
    return lp_result["interior"]


def _filter_unique_nonempty_J(
    I: list[int],  # noqa: E741
    J: list[int],
    hyperplanes: list[tuple[np.ndarray, F]],
    inequalities: list[tuple[np.ndarray, F]],
    lp_solver: kFaceCDDBackend | None,
) -> list[int] | None:
    """Filter J to unique non-empty restrictions."""
    if not I:
        seen: set[tuple] = set()
        filtered: list[int] = []
        for j in J:
            if inequalities and not intersects_support(j, I, lp_solver):
                continue
            key = _canonical_fullspace(hyperplanes[j])
            if key in seen:
                continue
            seen.add(key)
            filtered.append(j)
        return filtered

    li_empty, x0, B = _prepare_LI(hyperplanes, I)
    if li_empty:
        return None

    seen: set[tuple] = set()
    filtered: list[int] = []
    for j in J:
        key = _restrict_key(hyperplanes, x0, B, j)
        if key in seen:
            continue
        seen.add(key)
        if key == ("const", "empty"):
            continue
        if inequalities and not intersects_support(j, I, lp_solver):
            continue
        filtered.append(j)
    return filtered


def deletion_restriction(
    I: list[int],  # noqa: E741
    J: list[int],
    hyperplanes: list[tuple[np.ndarray, F]],
    inequalities: list[tuple[np.ndarray, F]],
    lp_solver: kFaceCDDBackend | None,
    depth: int = 0,
    verbose: bool = False,
    counter: list[int] | None = None,
) -> int:
    """Perform deletion-restriction recursion."""
    J = _filter_unique_nonempty_J(I, J, hyperplanes, inequalities, lp_solver)

    if J is None:
        return 0

    if not J:
        if counter is not None:
            counter[0] += 1
            if verbose and counter[0] % 1000 == 0:
                print(f"Found {counter[0]} chambers...")
        return 1

    j = J[-1]
    c_del = deletion_restriction(
        I, J[:-1], hyperplanes, inequalities, lp_solver, depth + 1, verbose, counter
    )
    c_res = deletion_restriction(
        I + [j],
        J[:-1],
        hyperplanes,
        inequalities,
        lp_solver,
        depth + 1,
        verbose,
        counter,
    )
    return c_del + c_res


def parse_inputs(
    hyperplanes: Sequence[Hyperplane], support: Polyhedron | Sequence[HalfSpace]
) -> tuple[list[tuple[np.ndarray, F]], list[tuple[np.ndarray, F]]]:
    """Parse hyperplanes and support into tuple format for internal use."""
    parsed_hyperplanes = [(h.normal, h.offset) for h in hyperplanes]
    if isinstance(support, Polyhedron):
        parsed_support = [(A_row, b_val) for A_row, b_val in zip(support.A, support.b)]
    else:
        parsed_support = [(h.normal, h.offset) for h in support]
    return parsed_hyperplanes, parsed_support


def number_of_regions(
    hyperplanes: Arrangement | Sequence[Hyperplane],
    support: Polyhedron | Sequence[HalfSpace] | None = None,
    verbose: bool = False,
) -> int:
    """Compute the number of regions of a hyperplane arrangement.

    Args:
        hyperplanes: Arrangement or sequence of hyperplanes defining the arrangement.
        support: Bounding polyhedron or sequence of halfspace constraints.
            Unlike ppart, delres does not require a bounded polytope.
            Defaults to full space.
        verbose: Print progress messages.

    Returns:
        Number of regions in the arrangement.
    """
    # Normalize hyperplanes to a sequence
    if isinstance(hyperplanes, Arrangement):
        hyperplanes_seq = hyperplanes.as_list()
    else:
        hyperplanes_seq = list(hyperplanes)

    # Normalize support
    if support is None:
        support_seq: Sequence[HalfSpace] = []
    elif isinstance(support, Polyhedron):
        support_seq = support.halfspaces
    else:
        support_seq = list(support)

    # Handle edge case: no hyperplanes means 1 region (the whole support)
    if not hyperplanes_seq:
        return 1

    hyperplanes_parsed, support_parsed = parse_inputs(hyperplanes_seq, support_seq)
    I: list[int] = []  # noqa: E741
    J = list(range(len(hyperplanes_parsed)))

    lp_solver = kFaceCDDBackend(hyperplanes=hyperplanes_parsed, support=support_parsed)
    counter = [0]
    return deletion_restriction(
        I,
        J,
        hyperplanes_parsed,
        support_parsed,
        lp_solver,
        verbose=verbose,
        counter=counter,
    )
