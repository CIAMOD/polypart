from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import cdd.gmp as cdd  # CDD library for exact rational LP solving
import numpy as np

from .ftyping import Fraction as F
from .ftyping import as_fraction_vector
from .geometry import Hyperplane


class IncEnuCDDBackend:
    """
    Efficient backend for repeated interior-feasibility solves with a fixed inequality system.

    Model: maximise t subject to
        s_i · (a_i · x - b_i) ≤ 0  for all hyperplanes i \in {0,...,len(s)}
        t ≤ K           (adds numerical stability; K defaults to 1)
        g · x + t ≤ h   for all (g, h) in `support`

    All exact Fraction conversions and per-row matrix constructions are performed at initialisation.
    Solves then reuse the prebuilt rows and only select which hyperplane rows become equalities.

    Parameters
    ----------
    hyperplanes : list[tuple[np.ndarray, float]]
        Candidate equalities a · x = b. Only a subset is activated at solve time.
    support : list[tuple[np.ndarray, float]]
        Fixed constraints g · x ≤ h. Always active.
    K : int | F, default 1
        Bound for t: t ≤ K. Keep small and rational.

    Notes
    -----
    - Fractions are created once. Rows are stored and reused by reference.
    - Row assembly uses compact Python lists; no recomputation across solves.
    - NumPy is used only for fast extraction to floats for the primal x.
    - Structure of the LP and solving approach are unchanged relative to the original.
    """

    def __init__(
        self,
        hyperplanes: Sequence[Hyperplane],
        support: Optional[Sequence[Hyperplane]] = [],
    ) -> None:
        if not hyperplanes:
            raise ValueError("At least one hyperplane is required.")

        d = hyperplanes[0].normal.shape[0]
        self.d: int = d

        # Prebuild support rows: [h] + [-g] + [-1] for g · x + t ≤ h  ⇔  h − g·x − t ≥ 0
        ineq_rows: List[List[F]] = []
        for hp in support:
            g, h = hp.normal, hp.offset
            if len(g) != d:
                raise ValueError("Dimension mismatch in support.")
            row = [h]
            row.extend(-g[i] for i in range(d))
            row.append(F(-1))  # coefficient for t
            ineq_rows.append(row)

        # Slack variable for strict interior
        t_cap_row: List[F] = [F(1)] + [F(0)] * d + [F(-1)]  # t ≤ 1

        # Prebuild hyperplane rows
        hp_rows: List[List[F]] = []
        hp_rows_neg: List[List[F]] = []
        for hp in hyperplanes:
            a, b = hp.normal, hp.offset
            if len(a) != d:
                raise ValueError("Dimension mismatch in hyperplanes.")
            row = [b]
            row.extend(-a[i] for i in range(d))
            row.append(F(-1))
            hp_rows.append(row)

            row_neg = [-b]
            row_neg.extend(a[i] for i in range(d))
            row_neg.append(F(-1))
            hp_rows_neg.append(row_neg)

        # Base rows are constant across solves: support + t-bound
        self._base_rows: List[List[F]] = ineq_rows + [t_cap_row]
        self._hp_rows: List[List[F]] = hp_rows
        self._hp_rows_neg: List[List[F]] = hp_rows_neg

        # Objective: maximise t
        # Matches the original API shape: obj_func has length d + 2 (leading constant omitted by API)
        self._obj: List[F] = [F(0)] * (d + 1) + [F(1)]

    def _build_matrix(self, sign_vector: Sequence[int]) -> object:
        """Create the cdd matrix for the given LP configuration.

        Returns the matrix object and the starting index of equality rows.
        """
        # Compose rows by reference to avoid per-entry allocations.
        rows: List[List[F]] = list(self._base_rows)

        for j in range(len(sign_vector)):
            # Inequality row with sign according to signature
            if sign_vector[j] == 1:
                rows.append(self._hp_rows[j])
            else:
                rows.append(self._hp_rows_neg[j])

        return cdd.matrix_from_array(
            rows,
            rep_type=cdd.RepType.INEQUALITY,
            obj_type=cdd.LPObjType.MAX,
            obj_func=self._obj,
        )

    def solve(
        self,
        sign_vector: Sequence[int],
        solver: Optional[int] = None,
        compute_x: bool = False,
    ) -> Dict:
        """
        Solve with the subset of hyperplanes indexed by `eq_idx` active as equalities.

        Parameters
        ----------
        sign_vector : sequence of int \in {+1, -1}
            Sign vector of hyperplanes of variable length.
        solver : optional cdd.LPSolverType
            Defaults to DUAL_SIMPLEX if None.

        Returns
        -------
        dict with keys:
            interior: bool
            x: np.ndarray | None
            t_opt: Fraction | None
        """
        mat = self._build_matrix(sign_vector)
        lp = cdd.linprog_from_matrix(mat)
        cdd.linprog_solve(lp, solver=solver or cdd.LPSolverType.DUAL_SIMPLEX)

        if lp.status != cdd.LPStatusType.OPTIMAL:
            raise RuntimeError("LP solver did not find an optimal solution.")

        if not compute_x:
            t_opt = lp.obj_value  # Fraction
            interior = t_opt > 0.0
            return dict(interior=interior, x=None, t_opt=t_opt)

        # Extract primal x quickly; cdd exposes Fraction in primal_solution.
        x = as_fraction_vector(lp.primal_solution[: self.d])
        t_opt = lp.obj_value  # Fraction
        interior = t_opt > 0.0

        return dict(interior=interior, x=x, t_opt=t_opt)


class kFaceCDDBackend:
    """
    Efficient backend for repeated interior-feasibility solves with a fixed inequality system.

    Model: maximise t subject to
        g · x + t ≤ h   for all (g, h) in `support`
        t ≤ K           (adds numerical stability; K defaults to 1)
        a · x = b       for a chosen subset of `hyperplanes` per call

    All exact Fraction conversions and per-row matrix constructions are performed at initialisation.
    Solves then reuse the prebuilt rows and only select which hyperplane rows become equalities.

    Parameters
    ----------
    hyperplanes : list[tuple[np.ndarray, float]]
        Candidate equalities a · x = b. Only a subset is activated at solve time.
    support : list[tuple[np.ndarray, float]]
        Fixed constraints g · x ≤ h. Always active.
    K : int | F, default 1
        Bound for t: t ≤ K. Keep small and rational.

    Notes
    -----
    - Fractions are created once. Rows are stored and reused by reference.
    - Row assembly uses compact Python lists; no recomputation across solves.
    - NumPy is used only for fast extraction to floats for the primal x.
    - Structure of the LP and solving approach are unchanged relative to the original.
    """

    def __init__(
        self,
        hyperplanes: Sequence[Tuple[np.ndarray, float]],
        support: Sequence[Tuple[np.ndarray, float]],
    ) -> None:
        if not hyperplanes:
            raise ValueError("At least one hyperplane is required.")

        d = len(hyperplanes[0][0])
        self.d: int = d

        # Prebuild inequality rows: [h] + [-g] + [-1] for g · x + t ≤ h  ⇔  h − g·x − t ≥ 0
        ineq_rows: List[List[F]] = []
        for g, h in support:
            if len(g) != d:
                raise ValueError("Dimension mismatch in support.")
            row = [F(h)]
            row.extend([-F(g[i]) for i in range(d)])
            row.append(F(-1))  # coefficient for t
            ineq_rows.append(row)

        # slack variable for strict interior
        t_cap_row: List[F] = [F(1)] + [F(0)] * d + [F(-1)]  # t ≤ 1

        # Prebuild hyperplane rows: [b] + [-a] + [0] for a · x = b (marked as linear at solve time)
        hp_rows: List[List[F]] = []
        for a, b in hyperplanes:
            if len(a) != d:
                raise ValueError("Dimension mismatch in hyperplanes.")
            row = [F(b)]
            row.extend([-F(a[i]) for i in range(d)])
            row.append(F(0))  # t has 0 coefficient in equalities
            hp_rows.append(row)

        # Base rows are constant across solves: support + t-bound
        self._base_rows: List[List[F]] = ineq_rows + [t_cap_row]
        self._hp_rows: List[List[F]] = hp_rows

        # Objective: maximise t
        # Matches the original API shape: obj_func has length d + 2 (leading constant omitted by API)
        self._obj: List[F] = [F(0)] * (d + 1) + [F(1)]

        # Counter for number of solves performed
        self.solve_count: int = 0

    def _build_matrix(self, eq_idx: Sequence[int]) -> tuple[object, int]:
        """Create the cdd matrix for the given set of active equalities.

        Returns the matrix object and the starting index of equality rows.
        """
        # Compose rows by reference to avoid per-entry allocations.
        rows: List[List[F]] = list(self._base_rows)
        lin_start = len(rows)

        # Append selected hyperplane rows.
        # Using list comprehension over indices is already optimal in Python space here.
        rows.extend(self._hp_rows[i] for i in eq_idx)

        # Build linear index set: the last len(eq_idx) rows are equalities.
        lin_idx = set(range(lin_start, lin_start + len(eq_idx)))

        mat = cdd.matrix_from_array(
            rows,
            lin_set=lin_idx,
            rep_type=cdd.RepType.INEQUALITY,
            obj_type=cdd.LPObjType.MAX,
            obj_func=self._obj,
        )
        return mat, lin_start

    def solve(
        self,
        eq_idx: Sequence[int],
        solver: Optional[int] = None,
        compute_x: bool = False,
    ) -> Dict:
        """
        Solve with the subset of hyperplanes indexed by `eq_idx` active as equalities.

        Parameters
        ----------
        eq_idx : sequence of int
            Indices into the provided hyperplane list.
        solver : optional cdd.LPSolverType
            Defaults to DUAL_SIMPLEX if None.

        Returns
        -------
        dict with keys:
            feasible: bool
            interior: bool
            x: np.ndarray | None
            t_opt: Fraction | None
        """
        if not eq_idx:
            raise ValueError("At least one equality index is required.")
        n_hp = len(self._hp_rows)
        for i in eq_idx:
            if i < 0 or i >= n_hp:
                raise IndexError("Equality index out of range.")

        mat, _ = self._build_matrix(eq_idx)
        lp = cdd.linprog_from_matrix(mat)
        cdd.linprog_solve(lp, solver=solver or cdd.LPSolverType.DUAL_SIMPLEX)
        self.solve_count += 1

        if lp.status != cdd.LPStatusType.OPTIMAL:
            return dict(feasible=False, interior=False, x=None, t_opt=None)

        if not compute_x:
            t_opt = F(lp.obj_value)  # Fraction
            interior = t_opt > 0.0
            return dict(feasible=True, interior=interior, x=None, t_opt=t_opt)

        # Extract primal x quickly; cdd exposes Fraction in primal_solution.
        x = as_fraction_vector(lp.primal_solution[: self.d])

        t_opt = F(lp.obj_value)  # Fraction
        interior = t_opt > 0.0

        return dict(feasible=True, interior=interior, x=x, t_opt=t_opt)
