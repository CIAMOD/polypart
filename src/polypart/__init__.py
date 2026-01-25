"""PolyPart: Polytope partitioning and hyperplane arrangement algorithms."""

import polypart._patch_cdd  # noqa: F401
from polypart.algorithms.delres import number_of_regions
from polypart.algorithms.graph import (
    EquivalenceClass,
    PartitionGraph,
    StorageLevel,
    SymmetryGroup,
)
from polypart.algorithms.incenu import build_incenu_tree
from polypart.algorithms.ppart import build_partition_tree
from polypart.core.geometry import (
    Arrangement,
    HalfSpace,
    Hyperplane,
    Polyhedron,
    Polytope,
)
from polypart.core.tree import PartitionNode, PartitionTree
from polypart.core.typing import Fraction
from polypart.utils.io import load_tree, save_tree

__version__ = "0.2.0"

__all__ = [
    "Arrangement",
    "build_incenu_tree",
    "build_partition_tree",
    "EquivalenceClass",
    "Fraction",
    "HalfSpace",
    "Hyperplane",
    "load_tree",
    "number_of_regions",
    "PartitionGraph",
    "PartitionNode",
    "PartitionTree",
    "Polyhedron",
    "Polytope",
    "save_tree",
    "StorageLevel",
    "SymmetryGroup",
]
