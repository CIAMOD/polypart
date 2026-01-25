"""Partition tree data structures."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from polypart.core.geometry import Hyperplane
from polypart.core.typing import FractionVector, as_fraction_vector, is_fraction_vector


@dataclass(eq=False)
class PartitionNode:
    """Node in a partition tree.

    Attributes:
        parent: Parent node.
        depth: Depth in the tree (root is 0).
        cut: Hyperplane used to split this node (None for leaves).
        data: Algorithm-specific payload.
        idx: Index of the node in the serialized tree (used for save/load).
    """

    parent: Optional[PartitionNode] = field(default=None, repr=False)
    depth: int = 0
    _children: list[PartitionNode] = field(default_factory=list, init=False, repr=False)
    cut: Optional[Hyperplane] = field(default=None, init=False)
    data: Any = field(default=None)
    idx: Optional[int] = field(default=None)

    @property
    def is_leaf(self) -> bool:
        return len(self._children) == 0

    @property
    def children(self) -> list[PartitionNode]:
        return self._children

    def add_child(self, node: PartitionNode) -> None:
        """Add a child node. Sets parent and depth accordingly."""
        node.parent = self
        node.depth = self.depth + 1
        self._children.append(node)

    def classify(self, x: FractionVector) -> PartitionNode:
        """Traverse tree to find the leaf containing point x."""
        if self.is_leaf:
            return self

        if len(self._children) == 1:
            return self._children[0].classify(x)

        # If there are two children, use the cut hyperplane to decide
        if self.cut is None:
            raise RuntimeError("Internal split node missing cutting hyperplane.")

        if (x @ self.cut.normal) <= self.cut.offset:
            return self._children[0].classify(x)

        return self._children[1].classify(x)


class PartitionTree:
    """Tree representing a partition of a polytope."""

    def __init__(self, root: PartitionNode) -> None:
        self.root = root

    def classify(self, x: FractionVector) -> PartitionNode:
        """Classify a point into one of the leaf regions."""
        if not is_fraction_vector(x):
            x = as_fraction_vector(x)
        return self.root.classify(x)
