"""Geometry classes for polytopes and hyperplanes using rational arithmetic."""

from __future__ import annotations

from typing import Iterable, Optional, Sequence

import cdd.gmp
import numpy as np

from polypart.core.typing import (
    Fraction,
    FractionMatrix,
    FractionVector,
    NumberLike,
    SplitStrategy,
    as_fraction_matrix,
    as_fraction_vector,
    to_fraction,
)
from polypart.utils.volume import volume_nmz


class Hyperplane:
    """Affine hyperplane: normal . x = offset. Halfspace convention: normal . x <= offset."""

    def __init__(self, normal: Iterable[NumberLike], offset: NumberLike) -> None:
        self.normal: FractionVector = as_fraction_vector(normal)
        self.offset: Fraction = to_fraction(offset)

    @staticmethod
    def from_coefficients(coefficients: Iterable[NumberLike]) -> Hyperplane:
        """Create from [a1, ..., ad, b] representing a1*x1 + ... + ad*xd = b."""
        coeffs = list(coefficients)
        normal = as_fraction_vector(coeffs[:-1])
        offset = to_fraction(coeffs[-1])
        return Hyperplane(normal, offset)

    def as_tuple(self) -> tuple[FractionVector, Fraction]:
        """Return (normal, offset) tuple."""
        return self.normal, self.offset

    def as_coefficients(self) -> FractionVector:
        """Return [a1, ..., ad, b] for a1*x1 + ... + ad*xd = b."""
        return np.append(self.normal, self.offset)

    def flip(self) -> Hyperplane:
        """Return hyperplane with negated normal and offset."""
        hyperplane = Hyperplane.__new__(Hyperplane)
        hyperplane.normal = -self.normal
        hyperplane.offset = -self.offset
        return hyperplane

    def __neg__(self) -> Hyperplane:
        return self.flip()

    def __repr__(self) -> str:
        return f"Hyperplane(normal=[{' '.join(str(a) for a in self.normal)}], offset={self.offset})"

    def __call__(self, x: FractionVector) -> Fraction:
        """Evaluate normal . x - offset."""
        return np.dot(self.normal, x) - self.offset


HalfSpace = Hyperplane
"""HalfSpace represented by: normal . x <= offset."""

# Alias for backwards compatibility
Halfspace = HalfSpace


class Arrangement:
    """A collection of Hyperplanes defining a hyperplane arrangement.

    Can be constructed from:
    - A sequence of Hyperplane objects
    - A 2D array-like of coefficients where each row is [a1, ..., ad, b]
      representing a1*x1 + ... + ad*xd = b
    """

    def __init__(
        self,
        hyperplanes: Sequence[Hyperplane]
        | Iterable[Iterable[NumberLike]]
        | None = None,
    ) -> None:
        if hyperplanes is None:
            self._hyperplanes: list[Hyperplane] = []
            self._dim: int | None = None
        elif isinstance(hyperplanes, Arrangement):
            self._hyperplanes = list(hyperplanes._hyperplanes)
            self._dim = hyperplanes._dim
        else:
            hp_list = list(hyperplanes)
            if not hp_list:
                self._hyperplanes = []
                self._dim = None
            elif isinstance(hp_list[0], Hyperplane):
                self._hyperplanes = hp_list
                self._dim = len(hp_list[0].normal) if hp_list else None
            else:
                # Assume array-like of coefficients
                self._hyperplanes = [
                    Hyperplane.from_coefficients(row) for row in hp_list
                ]
                self._dim = (
                    len(self._hyperplanes[0].normal) if self._hyperplanes else None
                )

    @classmethod
    def from_coefficients(
        cls, coefficients: Iterable[Iterable[NumberLike]]
    ) -> Arrangement:
        """Create an Arrangement from coefficient rows.

        Args:
            coefficients: Iterable of [a1, ..., ad, b] rows representing
                hyperplanes a1*x1 + ... + ad*xd = b.

        Returns:
            New Arrangement instance.
        """
        return cls(coefficients)

    @property
    def dim(self) -> int | None:
        """Dimension of the ambient space, or None if empty."""
        return self._dim

    @property
    def hyperplanes(self) -> list[Hyperplane]:
        """List of hyperplanes in the arrangement."""
        return self._hyperplanes

    def __len__(self) -> int:
        return len(self._hyperplanes)

    def __getitem__(self, idx: int) -> Hyperplane:
        return self._hyperplanes[idx]

    def __iter__(self):
        return iter(self._hyperplanes)

    def __repr__(self) -> str:
        return f"Arrangement(n_hyperplanes={len(self)}, dim={self.dim})"

    def as_list(self) -> list[Hyperplane]:
        """Return hyperplanes as a list (for backwards compatibility)."""
        return self._hyperplanes

    def append(self, hyperplane: Hyperplane) -> None:
        """Add a hyperplane to the arrangement."""
        if self._dim is None:
            self._dim = len(hyperplane.normal)
        elif len(hyperplane.normal) != self._dim:
            raise ValueError(
                f"Hyperplane dimension {len(hyperplane.normal)} != arrangement dimension {self._dim}"
            )
        self._hyperplanes.append(hyperplane)

    def extend(self, hyperplanes: Iterable[Hyperplane]) -> None:
        """Extend the arrangement with multiple hyperplanes."""
        for hp in hyperplanes:
            self.append(hp)


class Polyhedron:
    """Convex polyhedron in H-representation: A x <= b."""

    def __init__(
        self, A: Iterable[Iterable[NumberLike]], b: Iterable[NumberLike]
    ) -> None:
        A = as_fraction_matrix(A)
        b = as_fraction_vector(b).reshape(-1, 1)
        if A.shape[0] != b.shape[0]:
            raise ValueError(
                f"A and b incompatible: A.shape={A.shape}, b.shape={b.shape}"
            )
        self._A = A
        self._b = b
        self._dim: int = A.shape[1]

    @property
    def n_inequalities(self) -> int:
        return self._A.shape[0]

    @property
    def A(self) -> FractionMatrix:
        return self._A

    @property
    def b(self) -> FractionVector:
        return self._b.flatten()

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def inequalities(self) -> tuple[FractionMatrix, FractionVector]:
        return self._A, self.b

    @property
    def halfspaces(self) -> list[HalfSpace]:
        """Return the polyhedron's inequalities as a list of HalfSpace objects."""
        return [HalfSpace(self._A[i], self._b[i, 0]) for i in range(self._A.shape[0])]

    @classmethod
    def from_hrep(
        cls, A: Iterable[Iterable[NumberLike]], b: Iterable[NumberLike]
    ) -> Polyhedron:
        return cls(A, b)

    @classmethod
    def _from_fraction_hrep(cls, A: FractionMatrix, b: FractionVector) -> Polyhedron:
        """Construct from pre-validated FractionMatrix/Vector."""
        b = b.reshape(-1, 1)
        if A.shape[0] != b.shape[0]:
            raise ValueError(
                f"A and b incompatible: A.shape={A.shape}, b.shape={b.shape}"
            )

        self = cls.__new__(cls)
        self._A = A
        self._b = b
        self._dim = A.shape[1]

        if isinstance(self, Polytope):
            self._vertices = None
            self._volume = None
            self._diameter = None

        return self

    def contains(self, x: Iterable[NumberLike], strict: bool = True) -> bool:
        """Check whether a point lies inside the polyhedron."""
        assert self._A.shape[1] == len(list(x)), "Dimension mismatch"
        x = as_fraction_vector(x)
        vals = self._A @ x.reshape(-1, 1)
        if strict:
            return bool(np.all(vals.flatten() < self._b.flatten()))
        return bool(np.all(vals.flatten() <= self._b.flatten()))

    def remove_redundancies(self) -> Polyhedron:
        """Remove redundant inequalities using cdd."""
        mat = cdd.gmp.matrix_from_array(np.hstack([self._b.reshape(-1, 1), -self._A]))
        redundant_rows = list(cdd.gmp.redundant_rows(mat))
        if redundant_rows:
            self._A = np.delete(self._A, redundant_rows, axis=0)
            self._b = np.delete(self._b, redundant_rows, axis=0)
        return self

    def add_halfspace(
        self,
        halfspace: HalfSpace,
        remove_redundancies: bool = True,
        hv: Optional[np.ndarray] = None,
    ) -> Polyhedron:
        """Return a new Polyhedron with the halfspace added.

        Args:
            halfspace: Hyperplane to add as inequality (normal . x <= offset).
            remove_redundancies: Whether to remove redundant inequalities.
            hv: Precomputed vertex values for Polytope (speeds up redundancy removal).
        """
        if (
            isinstance(self, Polytope)
            and self._vertices is not None
            and remove_redundancies
        ):
            A_keep, b_keep = self.filter_inequalities(halfspace, hv=hv)
        else:
            A_keep, b_keep = self._A, self._b

        A = np.concatenate((A_keep, halfspace.normal[None, :]), axis=0)
        b = np.concatenate(
            (b_keep, np.array([[halfspace.offset]], dtype=object)), axis=0
        ).reshape(-1)

        return self.__class__._from_fraction_hrep(A, b)


class Polytope(Polyhedron):
    """Bounded convex polytope in H-representation: A x <= b."""

    def __init__(
        self, A: Iterable[Iterable[NumberLike]], b: Iterable[NumberLike]
    ) -> None:
        super().__init__(A, b)
        self._vertices: Optional[FractionMatrix] = None
        self._volume: Optional[Fraction] = None
        self._diameter: Optional[Fraction] = None

    @property
    def n_vertices(self) -> int:
        if self._vertices is None:
            raise ValueError("Vertices not computed. Call .extreme() first.")
        return self._vertices.shape[0]

    @classmethod
    def from_vrep(cls, V: Iterable[Iterable[NumberLike]]) -> Polytope:
        """Construct a Polytope from vertices via cdd conversion."""
        V = as_fraction_matrix(V)
        ones = np.array([[Fraction(1)] for _ in range(V.shape[0])], dtype=object)
        mat = cdd.gmp.matrix_from_array(np.hstack([ones, V]))
        mat.rep_type = cdd.gmp.RepType.GENERATOR
        polyhedron = cdd.gmp.polyhedron_from_matrix(mat)
        H = np.array(cdd.gmp.copy_inequalities(polyhedron).array, dtype=object)
        b = H[:, 0]
        A = -H[:, 1:]
        return cls(A, b)

    @property
    def vertices(self) -> FractionMatrix:
        if self._vertices is None:
            raise ValueError("Vertices not computed. Call .extreme() first.")
        return self._vertices

    @vertices.setter
    def vertices(self, V: FractionMatrix) -> None:
        if not isinstance(V, np.ndarray) or V.dtype != object:
            raise TypeError("V must be a numpy.ndarray with dtype=object")
        self._vertices = V

    @property
    def volume(self) -> Fraction:
        if self._volume is None:
            self._volume = volume_nmz(self._A, self.b)
        return self._volume

    @property
    def diameter(self) -> Fraction:
        if self._diameter is None:
            if self._vertices is None:
                raise ValueError("Vertices not computed. Call .extreme() first.")
            max_dist = Fraction(0)
            for i in range(self._vertices.shape[0]):
                for j in range(i + 1, self._vertices.shape[0]):
                    dist_sq = sum(
                        (self._vertices[i, k] - self._vertices[j, k]) ** 2
                        for k in range(self._dim)
                    )
                    dist = Fraction(float(dist_sq) ** 0.5)
                    if dist > max_dist:
                        max_dist = dist
            self._diameter = max_dist
        return self._diameter

    def extreme(self) -> None:
        """Compute vertices with cdd and cache the V-representation."""
        mat = cdd.gmp.matrix_from_array(np.hstack([self._b, -self._A]))
        mat.rep_type = cdd.gmp.RepType.INEQUALITY
        polyhedron = cdd.gmp.polyhedron_from_matrix(mat)
        verts = cdd.gmp.copy_generators(polyhedron).array
        if len(verts) == 0:
            raise ValueError("Empty vertex set. H-rep may be infeasible.")
        V = np.array(verts, dtype=object)  # Shape [n_vertices, dim+1]
        if not np.all(V[:, 0] == 1):
            raise ValueError("Unbounded polytope: contains rays.")
        self._vertices = V[:, 1:]

    def filter_inequalities(
        self, cut_hyperplane: Hyperplane, hv: Optional[np.ndarray] = None
    ) -> tuple[FractionMatrix, FractionVector]:
        """Remove inequalities redundant after adding cut_hyperplane."""
        if self._vertices is None:
            raise ValueError("Vertices not computed. Call .extreme() first.")
        if hv is None:
            hv = self._vertices @ cut_hyperplane.normal
        lvertices = self._vertices[hv < cut_hyperplane.offset]
        values = lvertices @ self._A.T  # Shape [n_lvertices, n_inequalities]
        to_keep = [
            i for i in range(self._A.shape[0]) if np.any(values[:, i] == self._b[i, 0])
        ]
        return self._A[to_keep, :], self._b[to_keep, :]

    def split_by_hyperplane(
        self, hyperplane: Hyperplane, remove_redundancies: bool = True
    ) -> tuple[Polytope, Polytope]:
        """Split the polytope by a hyperplane into two child polytopes."""
        if self._vertices is None:
            self.extreme()

        hv = self.vertices @ hyperplane.normal

        left = self.add_halfspace(
            hyperplane, remove_redundancies=remove_redundancies, hv=hv
        )
        left.extreme()

        complement = hyperplane.flip()
        right = self.add_halfspace(
            complement, remove_redundancies=remove_redundancies, hv=-hv
        )

        # Compute right vertices from intersection + original right side
        c_vertices = left.vertices[
            (left.vertices @ hyperplane.normal) == hyperplane.offset
        ]
        r_vertices = self.vertices[hv > hyperplane.offset]
        right.vertices = np.concatenate((c_vertices, r_vertices), axis=0)

        return left, right

    def intersecting_hyperplanes(
        self,
        hyperplanes: Iterable[Hyperplane],
        strategy: SplitStrategy = "v-entropy",
    ) -> tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        """Identify hyperplanes that intersect the polytope and count vertex distribution."""
        A = np.vstack([h.normal for h in hyperplanes])  # Shape [n_hyperplanes, dim]
        b = np.array(
            [h.offset for h in hyperplanes], dtype=object
        )  # Shape [n_hyperplanes]

        values = self.vertices @ A.T  # Shape [n_vertices, n_hyperplanes]

        if strategy == "v-entropy":
            n_less = np.sum(values < b, axis=0)
            n_greater = np.sum(values > b, axis=0)
            mask = np.logical_and(n_less > 0, n_greater > 0)
        else:
            n_less, n_greater = None, None
            less = np.any(values < b, axis=0)
            greater = np.any(values > b, axis=0)
            mask = np.logical_and(less, greater)

        return np.asarray(mask, dtype=bool), n_less, n_greater

    def __repr__(self) -> str:
        n_vertices = self._vertices.shape[0] if self._vertices is not None else "?"
        return f"Polytope(dim={self.dim}, n_ineq={self._A.shape[0]}, n_vertices={n_vertices})"
