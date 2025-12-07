from fractions import Fraction

import numpy as np

from polypart.geometry import Hyperplane
from polypart.lp_backends import IncEnuCDDBackend


def test_incenu_cddbackend_empty_support():
    # Hyperplanes defining a square in R^2
    hyperplanes = [
        Hyperplane(normal=[1, 0], offset=Fraction(1)),  # x = 1
        Hyperplane(normal=[-1, 0], offset=Fraction(0)),  # x = 0
        Hyperplane(normal=[0, 1], offset=Fraction(1)),  # y = 1
        Hyperplane(normal=[0, -1], offset=Fraction(0)),  # y = 0
    ]

    backend = IncEnuCDDBackend(hyperplanes, support=[])

    # Empty sign vector: no constraints
    res = backend.solve([], compute_x=True)
    assert res["interior"] is True
    assert res["x"] is not None


def test_incenu_cddbackend_simplex_parallel_hyperplanes():
    # Simplex in R^2: x >= 0, y >= 0, x + y <= 1
    support = [
        Hyperplane.from_coefficients([-1, 0, 0]),  # -x <= 0
        Hyperplane.from_coefficients([0, -1, 0]),  # -y <= 0
        Hyperplane.from_coefficients([1, 1, 1]),  # x + y <= 1
    ]

    # Two parallel hyperplanes with normal a = (1,0) (vertical lines)
    hyperplanes = [
        Hyperplane(normal=[1, 0], offset=Fraction(3, 10)),  # x = 0.3
        Hyperplane(normal=[1, 0], offset=Fraction(4, 10)),  # x = 0.4
    ]

    backend = IncEnuCDDBackend(hyperplanes, support)

    # Empty sign vector: only support constraints -> polytope interior exists
    res = backend.solve([], compute_x=True)
    assert res["interior"] is True

    # Ensure interior point is indeed inside the simplex
    x_pt = res["x"]
    for ineq in support:
        assert ineq(x_pt) < 0  # strict inequality

    # Activate the first halfspace (x <= 0.3) -> should still have interior
    res1 = backend.solve([1], compute_x=True)
    assert res1["interior"] is True

    # Ensure interior point is indeed inside the simplex with x <= 0.3
    x_pt1 = res1["x"]
    for ineq in support:
        assert ineq(x_pt1) < 0  # strict inequality
    assert hyperplanes[0](x_pt1) < 0  # strict inequality

    # Activate first two halfspaces in opposite directions (x <= 0.3 and x >= 0.4)
    # This should be infeasible.
    res2 = backend.solve([1, -1], compute_x=True)
    assert res2["interior"] is False

    # Same hyperplanes
    hyperplanes = [
        Hyperplane(normal=[1, 0], offset=Fraction(3, 10)),  # x = 0.3
        Hyperplane(normal=[-1, 0], offset=Fraction(-3, 10)),  # x = 0.3
    ]
    backend = IncEnuCDDBackend(hyperplanes, support)
    res3 = backend.solve([1, 1], compute_x=False)
    assert res3["interior"] is False
