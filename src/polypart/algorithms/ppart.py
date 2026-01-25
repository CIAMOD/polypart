"""Polytope partitioning algorithm using decision trees."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from polypart.core.geometry import Arrangement, Hyperplane, Polytope
from polypart.core.tree import PartitionNode, PartitionTree
from polypart.core.typing import SplitStrategy


def choose_best_split(
    polytope: Polytope,
    candidates: Sequence[Hyperplane],
    strategy: SplitStrategy = "v-entropy",
    remove_redundancies: bool = True,
) -> tuple[
    Hyperplane | None, tuple[Polytope, Polytope] | None, Sequence[Hyperplane] | None
]:
    """Select hyperplane for splitting a polytope. If "v-entropy" strategy is chosen,
    the hyperplane that maximizes the vertex entropy is selected. If "random" strategy
    is chosen, a random intersecting hyperplane is selected.

    Args:
        polytope: Polytope to split.
        candidates: Candidate hyperplanes.
        strategy: Selection strategy ("v-entropy" or "random").
        remove_redundancies: Whether to filter redundant inequalities.

    Returns:
        Tuple of (selected hyperplane, child polytopes, remaining candidates).
    """
    if strategy not in {"random", "v-entropy"}:
        raise ValueError(f"Invalid strategy: {strategy}")

    if not candidates:
        return None, None, None

    mask, n_less, n_greater = polytope.intersecting_hyperplanes(candidates, strategy)
    idxs = np.where(mask)[0]

    if idxs.size == 0:
        return None, None, None

    if strategy == "v-entropy":
        # Maximizing product n_less * n_greater is equivalent to maximizing entropy
        scores = n_less[idxs] * n_greater[idxs]
        best_idx = int(idxs[np.argmax(scores)])
    else:
        best_idx = int(np.random.choice(idxs))

    best_hyp = candidates[best_idx]
    children = polytope.split_by_hyperplane(best_hyp, remove_redundancies)
    remaining = [candidates[i] for i in idxs if i != best_idx]

    return best_hyp, children, remaining


def build_partition_tree(
    hyperplanes: Arrangement | Sequence[Hyperplane],
    polytope: Polytope,
    strategy: SplitStrategy = "v-entropy",
    remove_redundancies: bool = True,
    verbose: bool = False,
    record_stats: bool = False,
) -> tuple[PartitionTree, int]:
    """Build a partition tree by recursively splitting a polytope.

    Args:
        polytope: Initial bounded polytope to partition. Must be a Polytope
            (not Polyhedron) since ppart requires a bounded region.
        hyperplanes: Arrangement or sequence of candidate hyperplanes for splitting.
        strategy: Hyperplane selection strategy.
        remove_redundancies: Whether to remove redundant inequalities.
        verbose: Print progress messages.
        record_stats: Attach node statistics for analysis.

    Returns:
        Tuple of (partition tree, number of leaf regions).
    """
    # Normalize hyperplanes to a list
    if isinstance(hyperplanes, Arrangement):
        hyperplanes_list = hyperplanes.as_list()
    else:
        hyperplanes_list = list(hyperplanes)

    if polytope._vertices is None:
        polytope.extreme()

    root = PartitionNode(data={"polytope": polytope, "candidates": hyperplanes_list})

    stack = [root]
    n_partitions = 0
    prev_partitions = 0

    while stack:
        node = stack.pop()
        current_polytope = node.data["polytope"]
        current_candidates = node.data["candidates"]

        if record_stats:
            _record_node_stats(node, current_polytope, current_candidates)

        best_hyp, child_polys, remaining = choose_best_split(
            current_polytope,
            current_candidates,
            strategy=strategy,
            remove_redundancies=remove_redundancies,
        )

        if best_hyp is None:
            if current_polytope.n_vertices > 0:
                centroid = np.mean(current_polytope.vertices, axis=0)
                # Store centroid as seed
                node.data["seed"] = centroid

            n_partitions += 1
            if verbose and prev_partitions != n_partitions and n_partitions % 1000 == 0:
                print(f"Found {n_partitions} chambers...")
                prev_partitions = n_partitions
        else:
            node.cut = best_hyp

            left_node = PartitionNode(
                data={"polytope": child_polys[0], "candidates": remaining}
            )
            node.add_child(left_node)
            stack.append(left_node)

            right_node = PartitionNode(
                data={"polytope": child_polys[1], "candidates": remaining}
            )
            node.add_child(right_node)
            stack.append(right_node)

        del node.data["polytope"]
        del node.data["candidates"]
        if not node.data:
            node.data = None

    return PartitionTree(root), n_partitions


def _record_node_stats(
    node: PartitionNode, polytope: Polytope, candidates: Sequence[Hyperplane]
) -> None:
    """Attach statistics to node for analysis."""
    node.n_candidates = len(candidates)
    node.n_inequalities = polytope.n_inequalities
    node.n_vertices = polytope.n_vertices
