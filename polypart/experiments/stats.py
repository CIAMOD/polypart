import json
from typing import Any

from ..ppart import PartitionTree


def get_ppart_stats(
    tree: PartitionTree,
    alphas: list[int | str] = (1, 2, 5, 10, "inf"),
    include_per_depth_stats: bool = True,
) -> dict[str, Any]:
    """Compute statistics of the partition tree.
    Args:
        alphas: List of alpha values for which to compute moment statistics (e.g., 1 for average, 2 for variance).
        include_per_depth_stats: Whether to include per-depth statistics.

    Returns:
        dict: A dictionary with statistics including:
            - total_nodes: Total number of nodes in the tree.
            - avg_depth: Average depth of leaf nodes.
            - max_depth: Maximum depth of the tree.
            - avg_candidates: Average number of candidates per node.
            - avg_inequalities: Average number of inequalities per node.
            - avg_vertices: Average number of vertices per node.
            - per_depth_nodes: Number of nodes at each depth.
            - per_depth_avg_candidates: Average number of candidates per node at each depth.
            - per_depth_avg_inequalities: Average number of inequalities per node at each depth.
            - per_depth_avg_vertices: Average number of vertices per node at each depth.
            - per_depth_moments_candidates: Moments of candidates per depth for specified alphas.
            - per_depth_moments_inequalities: Moments of inequalities per depth for specified alphas.
            - per_depth_moments_vertices: Moments of vertices per depth for specified alphas.
    """
    total_nodes = 0
    max_depth = 0
    cum_depth = 0
    leaf_count = 0

    # Compute avg_candidates, avg_candidates per depth and number of nodes per depth
    per_depth_counts = {}
    avg_candidates = 0
    avg_inequalities = 0
    avg_vertices = 0
    per_depth_candidate_sums = {}
    per_depth_inequality_sums = {}
    per_depth_vertex_sums = {}
    per_depth_moments_candidates = {alpha: {} for alpha in alphas}
    per_depth_moments_inequalities = {alpha: {} for alpha in alphas}
    per_depth_moments_vertices = {alpha: {} for alpha in alphas}

    stack = [tree.root]
    while stack:
        node = stack.pop()
        total_nodes += 1
        avg_candidates += node.n_candidates
        avg_inequalities += node.n_inequalities
        avg_vertices += node.n_vertices
        # Update per-depth counts and sums
        per_depth_counts[node.depth] = per_depth_counts.get(node.depth, 0) + 1
        per_depth_candidate_sums[node.depth] = (
            per_depth_candidate_sums.get(node.depth, 0) + node.n_candidates
        )
        per_depth_inequality_sums[node.depth] = (
            per_depth_inequality_sums.get(node.depth, 0) + node.n_inequalities
        )
        per_depth_vertex_sums[node.depth] = (
            per_depth_vertex_sums.get(node.depth, 0) + node.n_vertices
        )
        # Compute moments
        for alpha in alphas:
            if alpha == "inf":
                per_depth_moments_candidates[alpha][node.depth] = max(
                    per_depth_moments_candidates[alpha].get(node.depth, 0),
                    node.n_candidates,
                )
                per_depth_moments_inequalities[alpha][node.depth] = max(
                    per_depth_moments_inequalities[alpha].get(node.depth, 0),
                    node.n_inequalities,
                )
                per_depth_moments_vertices[alpha][node.depth] = max(
                    per_depth_moments_vertices[alpha].get(node.depth, 0),
                    node.n_vertices,
                )
                continue
            per_depth_moments_candidates[alpha][node.depth] = (
                per_depth_moments_candidates[alpha].get(node.depth, 0)
                + node.n_candidates**alpha
            )
            per_depth_moments_inequalities[alpha][node.depth] = (
                per_depth_moments_inequalities[alpha].get(node.depth, 0)
                + node.n_inequalities**alpha
            )
            per_depth_moments_vertices[alpha][node.depth] = (
                per_depth_moments_vertices[alpha].get(node.depth, 0)
                + node.n_vertices**alpha
            )

        if node.depth > max_depth:
            max_depth = node.depth
        if node.is_leaf:
            cum_depth += node.depth
            leaf_count += 1
        else:
            stack.extend(node._children)

    avg_depth = cum_depth / leaf_count if leaf_count > 0 else 0
    avg_candidates /= total_nodes
    avg_inequalities /= total_nodes
    avg_vertices /= total_nodes

    # Normalize moments
    for alpha in alphas:
        if alpha == "inf":
            continue
        try:
            for depth in per_depth_moments_candidates[alpha]:
                count = per_depth_counts[depth]
                per_depth_moments_candidates[alpha][depth] /= count
                per_depth_moments_candidates[alpha][depth] **= 1 / alpha
        except ZeroDivisionError:
            for depth in per_depth_moments_candidates[alpha]:
                per_depth_moments_candidates[alpha][depth] = "NaN"
        try:
            for depth in per_depth_moments_inequalities[alpha]:
                count = per_depth_counts[depth]
                per_depth_moments_inequalities[alpha][depth] /= count
                per_depth_moments_inequalities[alpha][depth] **= 1 / alpha
        except ZeroDivisionError:
            for depth in per_depth_moments_inequalities[alpha]:
                per_depth_moments_inequalities[alpha][depth] = "NaN"
        try:
            for depth in per_depth_moments_vertices[alpha]:
                count = per_depth_counts[depth]
                per_depth_moments_vertices[alpha][depth] /= count
                per_depth_moments_vertices[alpha][depth] **= 1 / alpha
        except ZeroDivisionError:
            for depth in per_depth_moments_vertices[alpha]:
                per_depth_moments_vertices[alpha][depth] = "NaN"

    _stat_dict = {
        "total_nodes": total_nodes,
        "avg_depth": avg_depth,
        "max_depth": max_depth,
        "per_depth_nodes": per_depth_counts,
        "avg_candidates": avg_candidates,
        "avg_inequalities": avg_inequalities,
        "avg_vertices": avg_vertices,
        "per_depth_avg_candidates": {
            depth: per_depth_candidate_sums[depth] / count
            for depth, count in per_depth_counts.items()
        },
        "per_depth_avg_inequalities": {
            depth: per_depth_inequality_sums[depth] / count
            for depth, count in per_depth_counts.items()
        },
        "per_depth_avg_vertices": {
            depth: per_depth_vertex_sums[depth] / count
            for depth, count in per_depth_counts.items()
        },
        "per_depth_moments_candidates": per_depth_moments_candidates,
        "per_depth_moments_inequalities": per_depth_moments_inequalities,
        "per_depth_moments_vertices": per_depth_moments_vertices,
    }
    # Remove per-depth stats if not requested
    if not include_per_depth_stats:
        for key in list(_stat_dict.keys()):
            if "per_depth" in key:
                del _stat_dict[key]
    return _stat_dict


def print_ppart_stats(tree, include_per_depth_stats: bool = False) -> None:
    """Print the statistics of the partition tree in a readable format."""
    stat_dict = get_ppart_stats(tree, include_per_depth_stats=include_per_depth_stats)

    # Parse floats for better readability
    for key, value in stat_dict.items():
        if isinstance(value, float):
            stat_dict[key] = round(value, 4)

    print(json.dumps(stat_dict, indent=4))
