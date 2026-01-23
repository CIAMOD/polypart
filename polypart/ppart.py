"""Main algorithm and classes for polytope partitioning using decision trees."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np

from .ftyping import FractionVector, SplitStrategy, as_fraction_vector
from .geometry import Hyperplane, Polytope


@dataclass(eq=False)
class PartitionNode:
    """Node in a binary partition tree representing a polytope region.

    Each node represents either an internal split (with a cutting hyperplane)
    or a leaf region (terminal partition). Internal nodes have exactly two
    children corresponding to the two half-spaces defined by the cut.

    Attributes:
        polytope: The polytope associated with this node (None for processed nodes).
        candidates: List of candidate hyperplanes for further splitting (None for processed nodes).
        parent: Parent node in the tree (None for root).
        depth: Depth of this node in the tree (0 for root).
        cut: Hyperplane used to split this node (None for leaf nodes).
        children: List of child nodes (empty for leaf nodes).
        _centroid: Cached centroid of the polytope (computed lazily).
        _id: Unique identifier for leaf nodes.
    """

    # Volatile data, discarded after processing
    polytope: Optional[Polytope]
    candidates: Optional[List[Hyperplane]]

    # Instance attributes
    parent: Optional["PartitionNode"] = None
    depth: int = 0

    _children: List["PartitionNode"] = field(default_factory=list)
    is_leaf: bool = field(default=False, init=False)

    # Stored only for internal nodes
    cut: Optional[Hyperplane] = field(default=None)

    # Stored only for leaves, set by make_leaf()
    centroid: Optional[FractionVector] = field(default=None, init=False, repr=False)

    # Statistics
    n_candidates: int = field(default=0, init=False)
    n_inequalities: int = field(default=0, init=False)
    n_vertices: int = field(default=0, init=False)

    def set_stats(self) -> None:
        """Initialize statistics just after creation."""
        if self.candidates is not None:
            self.n_candidates = len(self.candidates)
        else:
            raise RuntimeError("Candidates list is None at node initialization.")
        if self.polytope is not None:
            self.n_inequalities = self.polytope.A.shape[0]
            if self.polytope.vertices is not None:
                self.n_vertices = self.polytope.vertices.shape[0]
            else:
                raise RuntimeError("Polytope vertices are None at node initialization.")
        else:
            raise RuntimeError("Polytope is None at node initialization.")

    def make_leaf(self) -> None:
        """
        Mark this node as a leaf, compute and store its centroid.
        """
        if self._children:
            raise RuntimeError("Leaf nodes cannot have children.")
        if self.polytope is None:
            raise RuntimeError("Cannot compute centroid without polytope.")
        self.is_leaf = True
        self.centroid = np.mean(self.polytope.vertices, axis=0)

    def add_child(
        self, child_poly: Polytope, candidates: List[Hyperplane]
    ) -> "PartitionNode":
        """Create and add a child node to this node.

        Args:
            child_poly: Polytope for the child node.
            candidates: List of candidate hyperplanes for further splitting.

        Returns:
            PartitionNode: The newly created child node.
        """
        node = PartitionNode(child_poly, candidates, parent=self, depth=self.depth + 1)
        self._children.append(node)
        return node

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
            return self
        assert self.cut is not None
        x = as_fraction_vector(x)
        if (x @ self.cut.normal) <= self.cut.offset:
            return self._children[0].classify(x)
        else:
            return self._children[1].classify(x)


class PartitionTree:
    """Binary tree representing a recursive partition of a polytope.

    The tree structure represents how a polytope has been recursively split
    using hyperplanes. Each internal node corresponds to a split, and each
    leaf node represents a final partition region.

    Attributes:
        root: Root PartitionNode of the tree.
    """

    def __init__(self, root: PartitionNode):
        """Initialize the partition tree with the given root node.

        Args:
            root: The root PartitionNode of the tree.
        """
        self.root = root

        self._stat_dict: Optional[dict[str, Any]] = None

    def classify(self, x: FractionVector) -> PartitionNode:
        """Classify a point into one of the leaf regions.

        Args:
            x: Point to classify as a fraction vector.

        Returns:
            PartitionNode: The leaf node containing the point.
        """
        return self.root.classify(x)


def choose_best_split(
    polytope: Polytope,
    candidates: Sequence[Hyperplane],
    strategy: SplitStrategy = "v-entropy",
    remove_redundancies: bool = True,
) -> Tuple[
    Optional[Hyperplane],
    Optional[Tuple[Polytope, Polytope]],
    Optional[List[Hyperplane]],
]:
    """Select optimal hyperplane for splitting a polytope using specified strategy.

    This function chooses a hyperplane from the candidates that intersects the
    polytope and returns the selected hyperplane, resulting child polytopes,
    and remaining candidate hyperplanes.

    Args:
        polytope: The polytope to be split.
        candidates: Sequence of candidate hyperplanes for splitting.
        strategy: Strategy for selecting the hyperplane:
            - "random": Random selection among intersecting hyperplanes
            - "v-entropy": Select hyperplane that maximizes entropy approximation
              based on vertex distribution across the split

    Returns:
        Tuple containing:
        - Selected hyperplane (None if no valid split found)
        - Tuple of two child polytopes (None if no valid split found)
        - List of remaining candidate hyperplanes (None if no valid split found)

    """
    # Validate strategy parameter
    if strategy not in {"random", "v-entropy"}:
        raise ValueError(
            f"Invalid strategy: {strategy}. Must be 'random' or 'v-entropy'"
        )

    if not candidates:
        return None, None, None

    # Find hyperplanes that intersect the polytope and get vertex counts
    mask, n_less, n_greater = polytope.intersecting_hyperplanes(candidates, strategy)
    idxs = np.where(mask)[0]

    if idxs.size == 0:
        return None, None, None

    if strategy == "v-entropy":
        # Compute entropy approximation for each intersecting hyperplane
        total_vertices = len(polytope.vertices)

        # Calculate proportions of vertices on each side
        p_less = n_less[idxs] / total_vertices
        p_greater = n_greater[idxs] / total_vertices

        # Compute Shannon entropy: H = -p_left * log2(p_left) - p_right * log2(p_right)
        # Use np.nan_to_num to handle log(0) cases (when all vertices on one side)
        entropies = -(p_less * np.log2(p_less, where=p_less > 0)) - (
            p_greater * np.log2(p_greater, where=p_greater > 0)
        )
        entropies = np.nan_to_num(entropies, nan=0.0)

        # Select hyperplane with maximum entropy (most balanced split)
        best_entropy_idx = np.argmax(entropies)
        b_i = int(idxs[best_entropy_idx])
    else:  # strategy == "random"
        # Random selection among intersecting hyperplanes
        b_i = int(np.random.choice(idxs))

    # Get the selected hyperplane and split the polytope
    b_hyp = candidates[b_i]
    children = polytope.split_by_hyperplane(b_hyp, remove_redundancies)

    # Remove the used hyperplane from candidates
    remaining = [candidates[i] for i in idxs if i != b_i]

    return b_hyp, children, remaining


def build_partition_tree(
    polytope: Polytope,
    hyperplanes: Sequence[Hyperplane],
    strategy: SplitStrategy = "v-entropy",
    remove_redundancies: bool = True,
    verbose: bool = False,
) -> Tuple[PartitionTree, int]:
    """Build a partition tree by recursively splitting a polytope with hyperplanes.

    Constructs a binary tree where each internal node represents a split made by
    a hyperplane, and each leaf represents a final partition region. The choice
    of splitting hyperplane at each step is determined by the specified strategy.

    Args:
        polytope: Initial polytope to partition.
        hyperplanes: Sequence of candidate hyperplanes for splitting.
        strategy: Strategy for selecting hyperplanes at each split:
            - "random": Random selection among intersecting hyperplanes
            - "v-entropy": Select hyperplane maximizing entropy approximation
        remove_redundancies: Whether to remove redundant inequalities after splits.
        verbose: If True, prints progress information during tree construction.

    Returns:
        Tuple containing:
        - PartitionTree: The constructed partition tree
        - int: Number of leaf regions (partitions) created

    Notes:
        - The polytope vertices are computed automatically if not already cached
        - For reproducible results with "random" strategy, set np.random.seed(...)
        - Progress is printed every 1000 partitions for long-running computations
    """
    # Ensure vertices are computed for the polytope
    if polytope._vertices is None:
        polytope.extreme()

    # Initialize the tree with root node
    root = PartitionNode(polytope, list(hyperplanes))
    stack = [root]
    n_partitions = 0
    prev_partitions = 0

    # Process nodes using depth-first traversal
    while stack:
        node = stack.pop()

        # Attempt to split the current node
        b_hyp, children, remaining_candidates = choose_best_split(
            node.polytope,
            node.candidates,
            strategy=strategy,
            remove_redundancies=remove_redundancies,
        )

        if b_hyp is None:
            # No valid split found - this becomes a leaf node
            node.make_leaf()
            n_partitions += 1

            # Progress reporting for large partitions
            if verbose and prev_partitions != n_partitions and n_partitions % 1000 == 0:
                print(f"Found {n_partitions} chambers...")
                prev_partitions = n_partitions
        else:
            # Valid split found - create child nodes
            node.cut = b_hyp
            for child_poly in children:  # type: ignore
                child = node.add_child(
                    child_poly,
                    (
                        list(remaining_candidates)
                        if remaining_candidates is not None
                        else []
                    ),
                )
                child.set_stats()
                stack.append(child)

        # Free memory by clearing processed data
        node.polytope = None
        node.candidates = None

    tree = PartitionTree(root)
    return tree, n_partitions
