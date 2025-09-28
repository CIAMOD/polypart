from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Iterable, Tuple, Optional

import numpy as np
import cdd.gmp

from .ftyping import as_fraction_matrix, as_fraction_vector, to_fraction, SplitStrategy
from .ftyping import NumberLike, FractionVector, FractionMatrix


@dataclass(frozen=True)
class Hyperplane:
    """Affine hyperplane boundary.

    Represents the affine hyperplane given by ``normal · x = offset``. We
    adopt the halfspace convention ``normal · x ≤ offset``.

    Args:
        normal (FractionVector): normal vector (dtype=object numpy array).
        offset (Fraction): scalar offset.
    """

    normal: FractionVector
    offset: Fraction

    def __post_init__(self):
        # Validate types at runtime
        if not isinstance(self.normal, np.ndarray):
            raise TypeError(
                "normal must be a numpy.ndarray with dtype=object containing Fractions. "
                "Use Hyperplane.from_coefficients() instead."
            )
        if not isinstance(self.offset, Fraction):
            raise TypeError(
                f"offset must be Fraction, got {type(self.offset).__name__}. "
                "Use Hyperplane.from_coefficients() instead."
            )
        elif self.normal.dtype != object:
            raise TypeError(
                "normal must be a numpy.ndarray with dtype=object containing Fractions. "
                "Use Hyperplane.from_coefficients() instead."
            )

        # Check normal is non-empty and 1D
        if self.normal.ndim != 1 or self.normal.size == 0:
            raise ValueError("normal must be a non-empty 1D numpy array.")

        # Check all elements are Fractions
        if not all(isinstance(x, Fraction) for x in self.normal.flat):
            raise TypeError(
                "normal must contain only Fractions. "
                "Use Hyperplane.from_coefficients() instead."
            )

    @staticmethod
    def from_coefficients(
        coefficients: Iterable[NumberLike],
    ) -> """Hyperplane""":
        """Create Hyperplane a1*x1 + ... + ad*xd = b from coefficients [a1, ..., ad, b].

        The vector [a1, ..., ad] is the normal vector and b is the offset.
        """
        normal, offset = as_fraction_vector(coefficients[:-1]), to_fraction(
            coefficients[-1]
        )
        return Hyperplane(normal, offset)

    def as_tuple(self) -> Tuple[FractionVector, Fraction]:
        return self.normal, self.offset


class Polytope:
    """Convex polytope in H-representation ``A x ≤ b`` using rational arithmetic.

    Use :meth:`Polytope.from_hrep` or :meth:`Polytope.from_vrep` to build an
    instance. Call :meth:`extreme` to compute and cache vertices (V-rep).

    Notes:
        Matrices and vectors are stored as object-dtype numpy arrays whose
        elements are ``fractions.Fraction`` objects.
    """

    def __init__(
        self, A: Iterable[Iterable[NumberLike]], b: Iterable[NumberLike]
    ) -> None:
        """Initialize a polytope from its H-representation."""
        A = as_fraction_matrix(A)
        b = as_fraction_vector(b).reshape(-1, 1)
        if A.shape[0] != b.shape[0]:
            raise ValueError(
                "A and b incompatible: got A.shape=%s, b.shape=%s" % (A.shape, b.shape)
            )
        self.A: FractionMatrix = A
        self.b: FractionVector = b
        self._vertices: Optional[FractionMatrix] = None
        self._dim: int = A.shape[1]

    # ---------- Constructors ----------
    @classmethod
    def from_hrep(
        cls,
        A: Iterable[Iterable[NumberLike]],
        b: Iterable[NumberLike],
    ) -> """Polytope""":
        return cls(A, b)

    @classmethod
    def from_vrep(cls, V: Iterable[Iterable[NumberLike]]) -> """Polytope""":
        """Construct a Polytope from vertices by converting to H-rep via cdd.

        Args:
            V: iterable of vertex coordinates.

        Returns:
            Polytope in H-representation equivalent to the convex hull of V.
        """
        V = as_fraction_matrix(V)
        # pycddlib expects a matrix with leading column 1s for vertices
        ones = np.array([[Fraction(1)] for _ in range(V.shape[0])], dtype=object)
        mat = cdd.gmp.matrix_from_array(np.hstack([ones, V]))
        mat.rep_type = cdd.gmp.RepType.GENERATOR
        polyhedron = cdd.gmp.polyhedron_from_matrix(mat)
        H = np.array(cdd.gmp.copy_inequalities(polyhedron).array, dtype=object)
        # H rows are (b, -A)
        b = H[:, 0]
        A = -H[:, 1:]
        return cls(A, b)

    # ---------- Properties ----------
    @property
    def dim(self) -> int:
        """Dimension of the ambient space."""
        return self._dim

    @property
    def vertices(self) -> FractionMatrix:
        if self._vertices is None:
            raise ValueError("Vertices not computed yet. Call .extreme() first.")
        return self._vertices

    # ---------- Operations ----------
    def extreme(self) -> None:
        """Compute exact vertices with cdd and cache the V-representation.

        Raises:
            ValueError: if H-rep is infeasible or unbounded.
        """
        mat = cdd.gmp.matrix_from_array(np.hstack([self.b, -self.A]))
        mat.rep_type = cdd.gmp.RepType.INEQUALITY
        polyhedron = cdd.gmp.polyhedron_from_matrix(mat)
        V = np.array(cdd.gmp.copy_generators(polyhedron).array, dtype=object)
        if V.size == 0:
            raise ValueError("Empty vertex set. The H-rep might be infeasible.")
        if not np.all([v == Fraction(1) for v in V[:, 0]]):
            raise ValueError("Inequalities do not represent a bounded polytope.")
        self._vertices = V[:, 1:]

    def add_halfspace(self, halfspace: Hyperplane) -> """Polytope""":
        """Return a new Polytope obtained by adding an inequality.

        Args:
            halfspace: Hyperplane to add as an inequality (normal · x ≤ offset).

        Returns:
            New Polytope with the extra inequality appended to H-rep.
        """
        A = np.concatenate((self.A, halfspace.normal[None, :]), axis=0)
        b = np.concatenate(
            (self.b, np.array([[halfspace.offset]], dtype=object)), axis=0
        ).reshape(-1)
        return Polytope(A, b)

    def remove_redundancies(self) -> """Polytope""":
        """Remove redundant inequalities from H-representation using cdd."""
        mat = cdd.gmp.matrix_from_array(np.hstack([self.b.reshape(-1, 1), -self.A]))
        redundant_rows = list(cdd.gmp.redundant_rows(mat))
        if redundant_rows:
            self.A = np.delete(self.A, redundant_rows, axis=0)
            self.b = np.delete(self.b, redundant_rows, axis=0)
        return self

    def split_by_hyperplane(
        self, hyperplane: Hyperplane
    ) -> tuple["Polytope", "Polytope"]:
        """Split the polytope by a hyperplane.

        Args:
            hyperplane: Hyperplane to split by.

        Returns:
            Tuple of two Polytopes, one for each side of the hyperplane.
        """
        # First child: intersection with halfspace (normal · x ≤ offset)
        left = self.add_halfspace(hyperplane)
        left.extreme()

        # Second child: intersection with complement halfspace (normal · x ≥ offset)
        complement_hyperplane = Hyperplane(-hyperplane.normal, -hyperplane.offset)
        right = self.add_halfspace(complement_hyperplane)
        # Compute right._vertices as union of intersection vertices on the
        # hyperplane and original vertices on the right side.
        c_vertices = left.vertices[
            np.where((left.vertices @ hyperplane.normal) == hyperplane.offset)
        ]
        r_vertices = self.vertices[
            np.where((self.vertices @ hyperplane.normal) > hyperplane.offset)
        ]
        right._vertices = np.concatenate((c_vertices, r_vertices), axis=0)

        return left, right

    def contains(self, x: Iterable[NumberLike]) -> bool:
        """Check whether a point lies inside the polytope (A x ≤ b).

        Args:
            x: point (length d) as an iterable of number-like values.

        Returns:
            True if the point satisfies all inequalities, False otherwise.
        """
        assert self.A.shape[1] == len(
            x
        ), "Point dimension does not match polytope dimension."
        x = as_fraction_vector(x)
        vals = self.A @ x.reshape(-1, 1)
        return bool(np.all(vals.flatten() <= self.b.flatten()))

    def intersecting_hyperplanes(
        self,
        hyperplanes: Iterable[Hyperplane],
        strategy: SplitStrategy = "v-entropy",
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        """Identify hyperplanes that intersect the polytope and count vertex distribution.

        A hyperplane intersects the polytope if and only if there are vertices strictly
        on both sides of the hyperplane. For entropy-based strategies, this method also
        computes the number of vertices on each side for entropy calculations.

        Args:
            hyperplanes: Iterable of hyperplanes to test for intersection.
            strategy: Split strategy determining what additional information to compute:
                - "v-entropy": Compute vertex counts on each side for entropy calculation
                - "random": Only compute intersection mask (counts will be None)

        Returns:
            Tuple containing:
            - np.ndarray: Boolean mask indicating which hyperplanes intersect the polytope
            - Optional[np.ndarray]: Number of vertices on the "less than" side of each hyperplane
              (None if strategy is "random")
            - Optional[np.ndarray]: Number of vertices on the "greater than" side of each hyperplane
              (None if strategy is "random")

        Notes:
            The "less than" and "greater than" sides are defined relative to the hyperplane
            equation normal·x ≤ offset. Vertices exactly on the hyperplane (normal·x = offset)
            are not counted in either side, as they don't contribute to determining if the
            hyperplane truly intersects the polytope interior.
        """
        # Stack hyperplane normals and offsets for vectorized computation
        A = np.vstack([h.normal for h in hyperplanes])  # Shape: (n_hyperplanes, dim)
        b = np.array(
            [h.offset for h in hyperplanes], dtype=object
        )  # Shape: (n_hyperplanes,)

        # Compute signed distances: vertices @ normals^T - offsets
        # Shape: (n_vertices, n_hyperplanes)
        values = self.vertices @ A.T

        if strategy == "v-entropy":
            # Count vertices on each side for entropy calculation
            n_less = np.sum(values < b, axis=0)
            n_greater = np.sum(values > b, axis=0)
            # Hyperplane intersects if vertices exist on both sides
            mask = np.logical_and(n_less > 0, n_greater > 0)
        else:  # strategy == "random"
            # Only determine intersection without counting
            n_less, n_greater = None, None
            less = np.any(values < b, axis=0)
            greater = np.any(values > b, axis=0)
            mask = np.logical_and(less, greater)

        return np.asarray(mask, dtype=bool), n_less, n_greater

    # ---------- Pretty ----------
    def __repr__(self) -> str:
        if self._vertices is None:
            n_vertices = "unknown"
        else:
            n_vertices = self._vertices.shape[0]
        return (
            f"Polytope(dim={self.dim}, n_ineq={self.A.shape[0]}, "
            f"n_vertices={n_vertices})"
        )
