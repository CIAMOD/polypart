from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple
import numpy as np

from .geometry import Polytope, Hyperplane
from .ftyping import as_fraction_vector, FractionVector


@dataclass
class PartitionNode:
    polytope: Optional[Polytope]
    candidates: Optional[List[Hyperplane]]
    parent: Optional["PartitionNode"] = None
    depth: int = 0
    cut: Optional[Hyperplane] = None
    children: List["PartitionNode"] = field(default_factory=list)
    centroid_: Optional[FractionVector] = None
    _id: Optional[int] = None

    @property
    def centroid(self) -> FractionVector:
        if self.centroid_ is None:
            assert self.polytope is not None, "Polytope not set."
            self.centroid_ = np.mean(self.polytope.vertices, axis=0)
        return self.centroid_

    def add_child(
        self, child_poly: Polytope, candidates: List[Hyperplane]
    ) -> "PartitionNode":
        node = PartitionNode(child_poly, candidates, parent=self, depth=self.depth + 1)
        self.children.append(node)
        return node

    def classify(self, x: FractionVector) -> "PartitionNode":
        if not self.children:
            return self
        assert self.cut is not None
        x = as_fraction_vector(x)
        if (x @ self.cut.normal) <= self.cut.offset:
            return self.children[0].classify(x)
        else:
            return self.children[1].classify(x)


@dataclass
class PartitionTree:
    """
    Binary tree representing a recursive partition of a polytope.
    Args:
        root: root PartitionNode of the tree.
        n_regions: number of leaf regions (partitions).
    """

    root: PartitionNode
    n_regions: int

    def classify(self, x: FractionVector) -> PartitionNode:
        """Classify point x into one of the leaf regions.
        Args:
            x (FractionVector): point to classify.
        Returns:
            PartitionNode: leaf node containing x.
        """
        return self.root.classify(x)


def choose_best_split(polytope: Polytope, candidates: Sequence[Hyperplane]) -> Tuple[
    Optional[Hyperplane],
    Optional[Tuple[Polytope, Polytope]],
    Optional[List[Hyperplane]],
]:
    """Pick a random hyperplane that intersects the polytope.

    The function splits the polytope by a randomly chosen intersecting
    hyperplane and returns the hyperplane, the child polytopes, and the
    remaining candidate hyperplanes.

    The choice is randomized using the module-level RNG.
    """
    if not candidates:
        return None, None, None
    # Only keep hyperplanes that intersect the polytope
    mask = polytope.intersecting_hyperplanes(candidates)
    idxs = np.where(mask)[0]
    if idxs.size == 0:
        return None, None, None
    # Choose uniformly at random among intersecting hyperplanes
    b_i = int(np.random.choice(idxs))
    b_hyp = candidates[b_i]
    children = polytope.split_by_hyperplane(b_hyp)
    remaining = [candidates[i] for i in idxs if i != b_i]
    return b_hyp, children, remaining


def build_partition_tree(
    polytope: Polytope, hyperplanes: Sequence[Hyperplane]
) -> Tuple[PartitionTree, int]:
    """Build a partition tree by recursively splitting a polytope with a set of
    hyperplanes. The splitting hyperplane at each node is chosen randomly among the
    candidate hyperplanes that intersect the polytope.

    Args:
        polytope (Polytope): initial polytope to partition.
        hyperplanes (Sequence[Hyperplane]): candidate hyperplanes.

    Returns:
        Tuple[PartitionTree, int]: the constructed tree and the number of
        leaf regions.

    Note:
        Set np.random.seed(...) before calling this function to ensure
        reproducibility of the partitioning.
    """
    if polytope._vertices is None:
        polytope.extreme()
    root = PartitionNode(polytope, list(hyperplanes))
    stack = [root]
    n_partitions = 0
    prev_partitions = 0
    while stack:
        node = stack.pop()
        b_hyp, children, remaining_candidates = choose_best_split(
            node.polytope, node.candidates
        )
        if b_hyp is None:
            node.centroid  # force compute only when leaf
            node._id = n_partitions
            n_partitions += 1
            if prev_partitions != n_partitions and n_partitions % 1000 == 0:
                print(f"Found {n_partitions} chambers...")
                prev_partitions = n_partitions
        else:
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
                stack.append(child)
        node.polytope = None
        node.candidates = None
    tree = PartitionTree(root, n_partitions)
    return tree, n_partitions
