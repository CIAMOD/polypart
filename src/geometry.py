import numpy as np
from pypoman import compute_polytope_vertices
import cdd
from typing import Iterable


class Hyperplane:
    """
    Hyperplane defined by a normal vector and an offset [w b], such that wx=b.
    """

    def __init__(self, normal: Iterable[float], offset: float):
        self.normal = np.array(normal, dtype=float).reshape(1, -1)
        self.offset = np.array(offset, dtype=float).reshape(1, 1)

    @classmethod
    def from_uniform(cls, n: int, r: int):
        """
        Create a random hyperplane in n dimensions with r-1 epsilons.
        """
        normal = np.random.uniform(-10, 10, (1, n * r))
        offset = np.random.uniform(-10, 10)
        return cls(normal, offset)

    @classmethod
    def from_numpy(cls, vector: np.ndarray):
        """ """

    def __repr__(self):
        return f"Hyperplane(normal={self.normal}, offset={self.offset})"


class Polytope:
    """
    Polytope in halfspace representation by Ax <= b.
    """

    def __init__(self, A: np.ndarray, b: np.ndarray):
        self.A = A
        self.b = b
        self._vertices: list[np.ndarray] | None = None
        self._dim = A.shape[1]

    @property
    def vertices(self):
        if self._vertices is None:
            raise ValueError("Vertices not computed yet.")
        return self._vertices

    def extreme(self):
        """
        Computes the vertices of the polytope.
        """
        self._vertices = compute_polytope_vertices(self.A, self.b)

    def is_degenerate(self):
        """
        Polytope is degenerate if it has less than dim + 1 vertices.
        """
        if self._vertices is None:
            raise ValueError("Vertices are needed to check degeneracy.")
        return len(self._vertices) < self._dim + 1

    def add_halfspace(self, halfspace: tuple[np.array, int]):
        """
        Create a new polytope by adding a halfspace defined by a normal vector and an offset.
        Removes redundant rows from the resulting halfspace representation.
        """
        A = np.concatenate((self.A, halfspace[0][None, :]), axis=0)
        b = np.concatenate((self.b, [halfspace[1]]), axis=0)
        # Remove redundant rows
        mat = cdd.matrix_from_array(np.hstack([b.reshape(-1, 1), -A]))
        redundant_rows = list(cdd.redundant_rows(mat))
        A = np.delete(A, redundant_rows, axis=0)
        b = np.delete(b, redundant_rows, axis=0)
        return Polytope(A, b)

    def __repr__(self):
        return f"Polytope(A={self.A}, b={self.b})"
