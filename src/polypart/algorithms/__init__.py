"""Algorithms for polytope partitioning and region counting."""

from polypart.algorithms.delres import deletion_restriction, number_of_regions
from polypart.algorithms.graph import (
    EquivalenceClass,
    PartitionGraph,
    StorageLevel,
    SymmetryGroup,
)
from polypart.algorithms.incenu import build_incenu_tree
from polypart.algorithms.ppart import build_partition_tree

__all__ = [
    "build_incenu_tree",
    "build_partition_tree",
    "ChamberClassifier",  # deprecated alias for PartitionGraph
    "deletion_restriction",
    "EquivalenceClass",
    "number_of_regions",
    "Orbit",  # deprecated alias for EquivalenceClass
    "PartitionGraph",
    "StorageLevel",
    "SymmetryGroup",
]
