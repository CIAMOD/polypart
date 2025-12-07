from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from .ftyping import Fraction, FractionVector
from .geometry import Hyperplane, Polytope
from .lp_backends import IncEnuCDDBackend


@dataclass
class PartitionNode:
    """Node in a binary partition tree representing a partition at a given step.

    Each node represents either an internal split (with a cutting hyperplane)
    or a leaf region (terminal partition). Internal nodes have exactly two
    children corresponding to the two half-spaces defined by the cut.

    Attributes:
        sign_vector (List[int]): Sign vector representing the partition at this node.
        parent (Optional[PartitionNode]): Parent node in the tree, None for root.
        depth (int): Depth of this node in the tree (root has depth 0).
        _children (List[PartitionNode]): Child nodes (empty for leaves).
        _cut (Optional[Hyperplane]): Cutting hyperplane for internal nodes.
        _leaf_id (Optional[int]): Unique ID for leaf nodes, None for internal nodes.
        _witness (Optional[FractionVector]): Witness point for leaf nodes.
    """

    # Attributes set at creation
    sign_vector: List[int]
    parent: Optional["PartitionNode"] = field(default=None, repr=False)
    depth: int = field(default=0)

    _children: List["PartitionNode"] = field(
        default_factory=list, init=False, repr=False
    )

    # Stored only for internal nodes
    _cut: Optional[Hyperplane] = field(default=None, init=False, repr=False)
    _witness: Optional[FractionVector] = field(default=None, init=False, repr=False)

    # Stored only for leaves, set by make_leaf()
    _leaf_id: Optional[int] = field(default=None, init=False)

    @property
    def is_leaf(self) -> bool:
        """True if this node is a leaf (final partition), False otherwise."""
        if not self._children and self.centroid is None:
            raise RuntimeError(
                "Node is neither leaf nor internal (inconsistent state)."
            )
        return not self._children

    @property
    def region_id(self) -> Optional[int]:
        """ID for leaf nodes, None otherwise."""
        return self._leaf_id

    @property
    def witness(self) -> np.ndarray:
        """
        Interior point (witness) for this node's partition.
        """
        if self._witness is None:
            raise RuntimeError("Node has no witness assigned.")
        return self._witness

    def make_leaf(self, leaf_id: int, witness: FractionVector) -> None:
        """
        Mark this node as a leaf, assign its region id, and witness point.
        """
        if self._children:
            raise RuntimeError("Cannot finalize a non-leaf node.")
        self._leaf_id = leaf_id

    def add_child(self, sign_vector: List[int]) -> "PartitionNode":
        """Create and add a child node to this node.

        Args:
            sign_vector: Sign vector for the child node.

        Returns:
            PartitionNode: The newly created child node.
        """
        node = PartitionNode(sign_vector, parent=self, depth=self.depth + 1)
        self._children.append(node)
        return node

    def set_cut(self, cut: Hyperplane) -> None:
        """Set the cutting hyperplane for this internal node.

        Args:
            cut: Hyperplane defining the cut at this node.
        """
        self._cut = cut

    def set_witness(self, witness: FractionVector) -> None:
        """Set the witness point for this leaf node.

        Args:
            witness: Witness point as a fraction vector.
        """
        self._witness = witness

    def classify(self, x: FractionVector) -> "PartitionNode":
        """Classify a point into the appropriate leaf node.

        Traverses the partition tree from this node down to find the leaf
        node that contains the given point.

        Args:
            x: Point to classify as a fraction vector.

        Returns:
            PartitionNode: The leaf node containing the point.
        """
        if not self._children:
            return self.sign_vector
        if self._cut is None:
            raise RuntimeError("Internal node missing cutting hyperplane.")
        if (x @ self.cut.normal) <= self.cut.offset:
            return self._children[0].classify(x)
        else:
            return self._children[1].classify(x)


def intersect_line(
    p: FractionVector,
    v: FractionVector,
    sing_vector: List[int],
    hyperplanes: List[Hyperplane],
    support: List[Hyperplane],
) -> FractionVector:
    """Compute intersection of line p + t v with the closest hyperplane
    considered so far (in the sign vector) or in the support.

    Args:
        p: Starting point of the line.
        v: Direction vector of the line.
        sing_vector: Sign vector defining which side of each hyperplane to consider.
        hyperplanes: List of hyperplanes defining the partition.
        support: List of hyperplanes defining the support polyhedron.

    Returns:
        Intersection point as a fraction vector.
    """
    min_t = None
    for i, sigma in enumerate(sing_vector):
        hp = hyperplanes[i] if sigma == 1 else -hyperplanes[i]
        denom = np.dot(hp.normal, v)
        if denom != 0:
            t = (hp.offset - np.dot(hp.normal, p)) / denom
            if t > 0 and (min_t is None or t < min_t):
                min_t = t
    for hp in support:
        denom = np.dot(hp.normal, v)
        if denom != 0:
            t = (hp.offset - np.dot(hp.normal, p)) / denom
            if t > 0 and (min_t is None or t < min_t):
                min_t = t
    if min_t is None:
        return None
    intersection = p + min_t * v
    return intersection


def perturb_witness(
    witness: FractionVector,
    cut: Hyperplane,
    sign_vector: List[int],
    hyperplanes: List[Hyperplane],
    support: List[Hyperplane],
) -> FractionVector:
    """Check wether witness lies on the hyperplane and perturb if necessary.

    Args:
        witness: Witness point to check.
        cut: Cutting hyperplane.
        sign_vector: Sign vector of the current partition.
        hyperplanes: List of hyperplanes defining the partition.
        support: List of hyperplanes defining the support polyhedron.
    Returns:
        Perturbed witness point.
    """
    if cut(witness) != 0:
        return witness

    intersection = intersect_line(
        witness,
        cut.normal,
        sign_vector,
        hyperplanes,
        support,
    )

    if intersection is None:
        # Move a fixed amount along the normal
        perturbed = witness + Fraction(1, 10) * cut.normal
        return perturbed

    # Take midpoint between witness and intersection, which is guaranteed to be
    # in the interior of the half-space
    midpoint = (witness + intersection) / 2
    return midpoint


def inc_enu(
    root: PartitionNode,
    hyperplanes: List[Hyperplane],
    support: List[Hyperplane],
    backend: IncEnuCDDBackend,
    verbose: bool = False,
) -> int:
    """DFS helper for incremental enumeration.
    Args:
        node: Current partition node to process.
        hyperplanes: List of hyperplanes defining the partition.
        support: Optional list of hyperplanes defining the support polyhedron.
        backend: Backend for LP operations.
    Returns:
        Number of final partitions (leaf nodes) under this node.
    """
    stack = [root]
    leaf_id = 0
    while stack:
        node = stack.pop()
        i = len(node.sign_vector)
        if i < len(hyperplanes):
            # Next cutting hyperplane
            cut = hyperplanes[i]
            # Perturb witness if it lies on the cutting hyperplane
            w = perturb_witness(
                node.witness,
                cut,
                node.sign_vector,
                hyperplanes,
                support,
            )
            # Add child for the half-space containing the witness
            sigma = 1 if cut(w) < 0 else -1
            child = node.add_child(node.sign_vector + [sigma])
            child.set_cut(cut if sigma == 1 else -cut)
            child.set_witness(w)

            # Recurse on this child
            stack.append(child)

            # Check feasibility of the opposite half-space
            s_minus = node.sign_vector + [-sigma]
            result = backend.solve(s_minus, compute_x=True)
            if result["interior"]:
                child_opposite = node.add_child(s_minus)
                child_opposite.set_cut(cut if sigma == -1 else -cut)
                child_opposite.set_witness(result["x"])
                # Recurse on this child
                stack.append(child_opposite)
        else:
            # Leaf node
            node.make_leaf(leaf_id, node.witness)
            leaf_id += 1
            if verbose and leaf_id % 1000 == 0:
                print(f"Found {leaf_id} chambers...")
    return leaf_id


def initial_witness(
    backend: IncEnuCDDBackend,
) -> FractionVector:
    """Compute an initial witness point inside the support polyhedron.

    Args:
        backend: Backend for LP operations.

    Returns:
        Witness point as a fraction vector.
    """
    result = backend.solve([], compute_x=True)
    if not result["interior"]:
        raise RuntimeError("Support is empty or has no interior.")
    return result["x"]


def build_tree_inc_enu(
    hyperplanes: List[Hyperplane],
    support: List[Hyperplane] | Polytope = [],
):
    """Incremental enumeration of the partition induced by hyperplanes.

    Args:
        hyperplanes: List of hyperplanes defining the partition.
        support: Optional list of hyperplanes defining a support polyhedron.

    Returns:
        root: Root node of the partition tree.
        n_partitions: number of final partitions (leaf nodes).
    """
    if isinstance(support, Polytope):
        support = [Hyperplane(normal=a, offset=b) for a, b in zip(support.A, support.b)]

    backend = IncEnuCDDBackend(hyperplanes, support)
    root = PartitionNode(sign_vector=[])
    root.set_witness(initial_witness(backend))
    n_partitions = inc_enu(root, hyperplanes, support, backend)
    return root, n_partitions
