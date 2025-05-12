"""
I tried to optimize the algorithm using cddlib directly, but I cannot access
the necessary functions or I am missing something ,so this is way slower!
"""

import cdd
import numpy as np
from time import perf_counter
from utils import get_planes


def get_simplex_inequalities(n: int, r: int):
    """
    Get the inequalities for the corner simplex in n dimensions.
    """

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


class Polytope:
    """
    Ax <= b
    A: np.ndarray
    b: np.array
    """

    def __init__(self, inequalities: cdd.Matrix):
        self.poly = self.get_polytope(inequalities)
        self.V = self.get_vertices()
        self.dim = len(inequalities.array[0]) - 1

    def get_polytope(self, inequalities: cdd.Matrix) -> cdd.Polyhedron:
        """
        Get the polytope from the inequalities.
        """
        return cdd.polyhedron_from_matrix(inequalities)

    def get_vertices(self) -> np.ndarray:
        """
        Get the vertices of the polytope.
        """
        return cdd.copy_generators(self.poly)

    def is_empty(self) -> bool:
        """
        Check if the polytope is empty.
        """
        return cdd.matrix_rank(self.V)[2] - 1 < self.dim

    def split(self, halfspaces: tuple[cdd.Matrix]) -> tuple["Polytope", "Polytope"]:
        """
        Split the polytope by a hyperplane.
        hyperplane: np.ndarray of shape (1, n)
        """
        H1 = cdd.copy_inequalities(self.poly)
        cdd.matrix_append_to(H1, halfspaces[0])
        H2 = cdd.copy_inequalities(self.poly)
        cdd.matrix_append_to(H2, halfspaces[1])

        return Polytope(H1), Polytope(H2)


def get_cdd_matrix(A: np.ndarray, b: np.array) -> cdd.Matrix:
    """
    Convert a matrix to cdd.Matrix.
    A: np.ndarray of shape (m, n)
    b: np.array of shape (m,)
    """
    return cdd.matrix_from_array(
        np.hstack((b.reshape(-1, 1), -A)), rep_type=cdd.RepType.INEQUALITY
    )


def get_cdd_halfspaces(
    planes: list[tuple[np.ndarray, list[int]]],
) -> list[tuple[cdd.Matrix]]:
    """
    Convert a list of families of planes to a list of tuples of cdd.Matrix.
    planes: list of tuples (w, [b1, b2, ...])
    w: np.ndarray of shape (n,)
    b: list of int
    """
    return [
        (
            cdd.matrix_from_array(
                np.hstack((b, -w)).reshape(1, -1), rep_type=cdd.RepType.INEQUALITY
            ),
            cdd.matrix_from_array(
                np.hstack((-b, w)).reshape(1, -1), rep_type=cdd.RepType.INEQUALITY
            ),
        )
        for w, bs in planes
        for b in bs
    ]


def compute_polytope_chambers(n: int, r: int):
    """
    Compute the number of chambers in the polytope using a trivial algorithm.
    """
    print(f"Computing chambers for n={n}, r={r}")
    start = perf_counter()
    simplex_inequalities = get_simplex_inequalities(n, r - 1)
    simplex = Polytope(get_cdd_matrix(*simplex_inequalities))

    planes = get_planes(n, r, 0, use_epsilons=True)
    halfspaces = get_cdd_halfspaces(planes)

    print(f"Found {len(halfspaces)} hyperplanes")

    print(f"Computing chambers...")
    chambers = [simplex]
    for h1, h2 in halfspaces:
        new_chambers = []
        for chamber in chambers:
            poly1, poly2 = chamber.split((h1, h2))
            if poly1.is_empty() or poly2.is_empty():
                new_chambers.append(chamber)
                continue
            new_chambers.append(poly1)
            new_chambers.append(poly2)

        chambers = new_chambers

    print(f"Found {len(chambers)} chambers in {perf_counter() - start:.2f} s")
    return chambers


n, r = 1, 7
compute_polytope_chambers(n, r)
