"""Core geometry and tree structures."""

from polypart.core.geometry import (
    Arrangement,
    HalfSpace,
    Halfspace,
    Hyperplane,
    Polyhedron,
    Polytope,
)
from polypart.core.tree import PartitionNode, PartitionTree
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

__all__ = [
    "Arrangement",
    "Fraction",
    "FractionMatrix",
    "FractionVector",
    "HalfSpace",
    "Halfspace",
    "Hyperplane",
    "NumberLike",
    "PartitionNode",
    "PartitionTree",
    "Polyhedron",
    "Polytope",
    "SplitStrategy",
    "as_fraction_matrix",
    "as_fraction_vector",
    "to_fraction",
]
