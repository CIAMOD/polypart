from typing import List, Set, Tuple

import numpy as np
from cdd.gmp import LPSolverType

from .ftyping import Fraction as F
from .geometry import Hyperplane, Polytope
from .lp_backends import kFaceCDDBackend


def rref(A, b):
    m, n = A.shape
    row = 0
    pivots = []

    for col in range(n):
        if row == m:
            break
        # find pivot in column col at or below row
        col_slice = A[row:, col]
        # np.nonzero works with dtype=object for != 0
        nz = np.nonzero(col_slice != 0)[0]
        if nz.size == 0:
            continue
        piv = row + nz[0]

        # swap rows row <-> piv
        if piv != row:
            A[[row, piv]] = A[[piv, row]]
            b[[row, piv]] = b[[piv, row]]

        # scale pivot row to make pivot exactly 1
        inv = F(1, 1) / A[row, col]
        A[row, :] = A[row, :] * inv
        b[row] = b[row] * inv

        # eliminate this column in all other rows
        # vectorised row ops over dtype=object
        for r in range(m):
            if r != row:
                factor = A[r, col]
                if factor != 0:
                    A[r, :] = A[r, :] - factor * A[row, :]
                    b[r] = b[r] - factor * b[row]

        pivots.append(col)
        row += 1

    return A, b, pivots


def nullspace_basis_from_rref(A_rref, pivots, n):
    pivot_row_for_col = {p: i for i, p in enumerate(pivots)}
    free_cols = [j for j in range(n) if j not in pivot_row_for_col]
    k = len(free_cols)
    if k == 0:
        return np.zeros((n, 0), dtype=object)

    B = np.zeros((n, k), dtype=object)
    for idx, f in enumerate(free_cols):
        B[f, idx] = F(1, 1)
        for p, r in pivot_row_for_col.items():
            B[p, idx] = -A_rref[r, f]
    return B  # shape (n, k)


def particular_solution_from_rref(b_rref, pivots, n):
    x = np.zeros(n, dtype=object)
    for r, p in enumerate(pivots):
        x[p] = b_rref[r]
    return x


def _is_zero_vec(v):  # v: sequence of Fractions
    return all(x == 0 for x in v)


def _canonical_affine(alpha, c):
    # alpha: list[Fraction], c: Fraction
    if _is_zero_vec(alpha):
        return ("const", 0) if c == 0 else ("const", "empty")
    v = list(alpha) + [c]
    p = next(i for i, z in enumerate(v) if z != 0)
    return ("affine", tuple(z / v[p] for z in v))


def _prepare_LI(hyperplanes, I):
    # Returns (empty, x0, B); B has shape (n, k) with dtype=object
    n = hyperplanes[0][0].shape[0]
    if not I:
        x0 = np.zeros(n, dtype=object)
        B = np.eye(n, dtype=object)
        return False, x0, B
    A = np.array([hyperplanes[i][0] for i in I], dtype=object)
    b = np.array([hyperplanes[i][1] for i in I], dtype=object)
    A_rref, b_rref, pivots = rref(A, b)
    m, d = A_rref.shape
    for r in range(m):
        if all(A_rref[r, c] == 0 for c in range(d)) and b_rref[r] != 0:
            return True, None, None
    x0 = particular_solution_from_rref(b_rref, pivots, n)
    B = nullspace_basis_from_rref(A_rref, pivots, n)
    if B.size == 0:
        B = np.zeros((n, 0), dtype=object)
    return False, x0, B


def _restrict_key(hyperplanes, x0, B, j):
    a, b = hyperplanes[j]
    if B.shape[1] == 0:
        # L_I is a singleton; restriction is 0 = c
        c = b - (a @ x0)
        return _canonical_affine([], c)
    alpha = (a @ B).tolist()
    c = b - (a @ x0)
    return _canonical_affine(alpha, c)


def _canonical_fullspace(h):
    # For I = ∅: canonicalise augmented vector (a, b)
    a, b = h
    v = list(a) + [b]
    if _is_zero_vec(v):
        return ("zero",)  # degenerate 0 = 0 on full space
    p = next(i for i, z in enumerate(v) if z != 0)
    return tuple([z / v[p] for z in v])


def intersects_support(
    j: int,
    I: list[int],
    lp_solver: kFaceCDDBackend | None,
) -> bool:
    """Check if hyperplane at index j intersects W = ∩_{i in I} h_i inside the support inequalities."""
    lp_result = lp_solver.solve(eq_idx=I + [j], solver=LPSolverType.DUAL_SIMPLEX)
    return lp_result["interior"]


def _filter_unique_nonempty_J(
    I: list[int],
    J: list[int],
    hyperplanes: list[Tuple[np.ndarray, F]],
    inequalities: list[Tuple[np.ndarray, F]],
    lp_solver: kFaceCDDBackend | None,
) -> list[int]:
    """
    From the candidate index set J, remove:
      - any j with h_j ∩ W = ∅ (respecting inequalities if provided), and
      - any j whose restriction to L_I duplicates a previously seen restricted hyperplane.

    Order is preserved: first occurrence is kept.
    If L_I is detected empty by _prepare_LI, return J unchanged to preserve original control flow.
    """
    # if not J:
    #     return J

    # Fast path when I = ∅
    if not I:
        seen: Set[Tuple] = set()
        filtered: List[int] = []
        for j in J:
            if inequalities and not intersects_support(j, I, lp_solver):
                continue
            key = _canonical_fullspace(hyperplanes[j])
            if key in seen:
                continue
            seen.add(key)
            filtered.append(j)
        return filtered

    # Prepare coordinates on L_I for restricted keys
    # print(hyperplanes, I)
    li_empty, x0, B = _prepare_LI(hyperplanes, I)
    if li_empty:
        # Cannot form restricted keys; keep J as is and let recursion discard via original logic
        return None

    seen: Set[Tuple] = set()
    filtered: List[int] = []
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


counter = 0
max_depth = 0
avg_depth = 0


def delition_restriction(
    I: list[int],
    J: list[int],
    hyperplanes: list[tuple[np.ndarray, F]],
    inequalities: list[tuple[np.ndarray, F]],
    lp_solver: kFaceCDDBackend | None,
    depth: int = 0,
    verbose: bool = False,
):
    """Perform deletion-restriction on hyperplanes with indices in I and J."""
    # print(f"Depth {depth}: I={I}, J={J}")
    J = _filter_unique_nonempty_J(I, J, hyperplanes, inequalities, lp_solver)
    # print(f"Depth {depth}: I={I}, filtered J={J}")
    if J is None:
        # print(f"L_I is empty at depth {depth}, returning 0")
        return 0  # L_I is empty

    if not J:  # Whole space LI is one chamber
        global counter, max_depth, avg_depth
        counter += 1
        max_depth = max(max_depth, depth)
        avg_depth += depth
        if verbose and counter % 1000 == 0:
            print(f"Found {counter} chambers...")
        # print(
        #     f"L_I non-empty with no remaining hyperplanes at depth {depth}, returning 1"
        # )
        return 1

    j = J[-1]
    c_del = delition_restriction(
        I, J[:-1], hyperplanes, inequalities, lp_solver, depth + 1, verbose
    )
    c_res = delition_restriction(
        I + [j], J[:-1], hyperplanes, inequalities, lp_solver, depth + 1, verbose
    )
    return c_del + c_res


def parse_inputs(hyperplanes, support):
    """Parse hyperplanes and support into required formats."""
    parsed_hyperplanes = [(h.normal, h.offset) for h in hyperplanes]
    if isinstance(support, Polytope):
        parsed_support = [(A_row, b_val) for A_row, b_val in zip(support.A, support.b)]
    else:
        parsed_support = [(h.normal, h.offset) for h in support]
    return parsed_hyperplanes, parsed_support


def number_of_regions(
    hyperplanes: list[Hyperplane],
    support: list[Hyperplane] | Polytope = [],
    verbose: bool = False,
):
    """Compute the number of regions of the arrangement of hyperplanes
    restricted to the feasible region defined by the support.

    Args:
        hyperplanes: list of Hyperplane objects defining the arrangement.
        support: list of Hyperplane objects defining the support region,
                 or a Polytope object. Defaults to empty list (no support).
        verbose: whether to print progress messages. Defaults to False.
    Returns:
        int: number of regions.
    """
    global counter, max_depth, avg_depth
    counter, max_depth, avg_depth = 0, 0, 0
    hyperplanes, support = parse_inputs(hyperplanes, support)
    I: list[int] = []
    J = list(range(len(hyperplanes)))
    lp_solver = kFaceCDDBackend(
        hyperplanes=hyperplanes,
        support=support,
    )
    return delition_restriction(I, J, hyperplanes, support, lp_solver, verbose=verbose)
