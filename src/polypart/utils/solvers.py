"""Exact LP solvers wrapping CDD."""

from __future__ import annotations

from typing import Sequence

import cdd.gmp
import numpy as np

from polypart.core.geometry import HalfSpace, Hyperplane
from polypart.core.typing import Fraction as F


def _to_mpq_array(data) -> np.ndarray:
    """Helper to ensure data is a numpy array of objects (mpq)."""
    return np.array(data, dtype=object)


class IncEnuCDDBackend:
    """Backend for incremental enumeration feasibility checks.

    Solves: maximise t subject to
        s_i * (a_i . x - b_i) <= 0  for activated hyperplanes
        t <= 1
        g . x + t <= h  for support constraints
    """

    def __init__(
        self,
        hyperplanes: Sequence[Hyperplane],
        support: Sequence[HalfSpace] | None = None,
    ) -> None:
        support = support or []
        if not hyperplanes and not support:
            raise ValueError(
                "At least one hyperplane or support constraint is required."
            )

        # Determine dimension
        d = (
            hyperplanes[0].normal.shape[0]
            if hyperplanes
            else support[0].normal.shape[0]
        )
        self.d = d

        # Structure: [h] | [-g] | [-1]
        if support:
            h_col = _to_mpq_array([hp.offset for hp in support])[:, None]
            g_mat = _to_mpq_array([hp.normal for hp in support])
            const_col = np.full((len(support), 1), F(-1), dtype=object)

            self._base_rows = np.hstack([h_col, -g_mat, const_col])
        else:
            self._base_rows = np.empty((0, d + 2), dtype=object)

        # Append t_cap_row: [1, 0...0, -1]
        t_cap = np.zeros(d + 2, dtype=object)
        t_cap[0] = F(1)
        t_cap[-1] = F(-1)
        self._base_rows = np.vstack([self._base_rows, t_cap])

        # We build two large matrices: one for Positive signs, one for Negative
        if hyperplanes:
            b_col = _to_mpq_array([hp.offset for hp in hyperplanes])[:, None]
            a_mat = _to_mpq_array([hp.normal for hp in hyperplanes])
            const_col = np.full((len(hyperplanes), 1), F(-1), dtype=object)

            # Positive: [b, -a, -1]
            self._hp_rows = np.hstack([b_col, -a_mat, const_col])

            # Negative: [-b, a, -1]
            self._hp_rows_neg = np.hstack([-b_col, a_mat, const_col])
        else:
            self._hp_rows = np.empty((0, d + 2), dtype=object)
            self._hp_rows_neg = np.empty((0, d + 2), dtype=object)

        self._obj: list[F] = [F(0)] * (d + 1) + [F(1)]

    def _build_matrix(self, sign_vector: Sequence[int]) -> object:
        n = len(sign_vector)
        mask = _to_mpq_array(sign_vector) == -1
        selected_rows = np.where(
            mask[:, None], self._hp_rows[:n], self._hp_rows_neg[:n]
        )
        final_matrix = np.vstack([self._base_rows, selected_rows])

        return cdd.gmp.matrix_from_array(
            final_matrix,
            rep_type=cdd.gmp.RepType.INEQUALITY,
            obj_type=cdd.gmp.LPObjType.MAX,
            obj_func=self._obj,
        )

    def solve(
        self,
        sign_vector: Sequence[int],
        solver: int | None = None,
        compute_x: bool = False,
    ) -> dict:
        mat = self._build_matrix(sign_vector)
        lp = cdd.gmp.linprog_from_matrix(mat)
        cdd.gmp.linprog_solve(lp, solver=solver or cdd.gmp.LPSolverType.DUAL_SIMPLEX)

        if lp.status != cdd.gmp.LPStatusType.OPTIMAL:
            raise RuntimeError("LP solver did not find optimal solution.")

        t_opt = lp.obj_value
        interior = t_opt > 0

        if not compute_x:
            return {"interior": interior, "x": None, "t_opt": t_opt}

        x = np.array(lp.primal_solution[: self.d], dtype=object)

        return {"interior": interior, "x": x, "t_opt": t_opt}


class kFaceCDDBackend:
    """Backend for k-face feasibility checks in deletion-restriction.

    Solves: maximise t subject to
        g . x + t <= h  for support constraints
        t <= 1
        a . x = b  for chosen hyperplanes
    """

    def __init__(
        self,
        hyperplanes: Sequence[tuple[np.ndarray, F]],
        support: Sequence[tuple[np.ndarray, F]],
    ) -> None:
        if not hyperplanes:
            raise ValueError("At least one hyperplane is required.")

        d = len(hyperplanes[0][0])
        self.d = d

        # Support structure: [h, -g, -1]
        if support:
            g_list, h_list = zip(*support)
            g_mat = _to_mpq_array(g_list)
            h_col = _to_mpq_array(h_list)[:, None]
            const_col = np.full((len(support), 1), F(-1), dtype=object)

            self._base_rows = np.hstack([h_col, -g_mat, const_col])
        else:
            self._base_rows = np.empty((0, d + 2), dtype=object)

        # Append t_cap_row: [1, 0...0, -1]
        t_cap = np.zeros(d + 2, dtype=object)
        t_cap[0] = F(1)
        t_cap[-1] = F(-1)
        self._base_rows = np.vstack([self._base_rows, t_cap])

        # Hyperplane structure: [b, -a, 0]
        a_list, b_list = zip(*hyperplanes)
        a_mat = _to_mpq_array(a_list)
        b_col = _to_mpq_array(b_list)[:, None]
        const_col_zero = np.zeros((len(hyperplanes), 1), dtype=object)

        self._hp_rows = np.hstack([b_col, -a_mat, const_col_zero])

        self._obj: list[F] = [F(0)] * (d + 1) + [F(1)]
        self.solve_count = 0

    def _build_matrix(self, eq_idx: Sequence[int]) -> tuple[object, int]:
        rows_to_add = self._hp_rows[eq_idx]
        full_arr = np.vstack([self._base_rows, rows_to_add])

        # Calculate indices of linear rows
        lin_start = len(self._base_rows)
        lin_set = set(range(lin_start, lin_start + len(eq_idx)))

        mat = cdd.gmp.matrix_from_array(
            full_arr,
            lin_set=lin_set,
            rep_type=cdd.gmp.RepType.INEQUALITY,
            obj_type=cdd.gmp.LPObjType.MAX,
            obj_func=self._obj,
        )
        return mat, lin_start

    def solve(
        self,
        eq_idx: Sequence[int],
        solver: int | None = None,
        compute_x: bool = False,
    ) -> dict:
        if not eq_idx:
            raise ValueError("At least one equality index is required.")

        mat, _ = self._build_matrix(eq_idx)
        lp = cdd.gmp.linprog_from_matrix(mat)
        cdd.gmp.linprog_solve(lp, solver=solver or cdd.gmp.LPSolverType.DUAL_SIMPLEX)
        self.solve_count += 1

        if lp.status != cdd.gmp.LPStatusType.OPTIMAL:
            return {"feasible": False, "interior": False, "x": None, "t_opt": None}

        t_opt = lp.obj_value
        interior = t_opt > 0

        if not compute_x:
            return {"feasible": True, "interior": interior, "x": None, "t_opt": t_opt}

        x = np.array(lp.primal_solution[: self.d], dtype=object)
        return {"feasible": True, "interior": interior, "x": x, "t_opt": t_opt}
