"""Statistics computation for partition trees."""

from __future__ import annotations

import json
from typing import Any

from polypart.core.tree import PartitionTree


def _has_recorded_stats(tree: PartitionTree) -> bool:
    """Check if the tree has recorded statistics (created with record_stats=True)."""
    return hasattr(tree.root, "n_candidates")


def get_ppart_stats(
    tree: PartitionTree,
    alphas: list[int | str] = (1, 2, 5, 10, "inf"),
    include_per_depth_stats: bool = True,
) -> dict[str, Any]:
    """Compute statistics of the partition tree.

    Args:
        tree: Partition tree to analyze. For detailed statistics (avg_candidates,
            avg_inequalities, avg_vertices), the tree must have been created with
            record_stats=True.
        alphas: Alpha values for moment statistics.
        include_per_depth_stats: Include per-depth breakdowns.

    Returns:
        Dictionary with tree statistics.

    Raises:
        ValueError: If detailed stats are requested but tree was created without
            record_stats=True.
    """
    has_stats = _has_recorded_stats(tree)

    total_nodes = 0
    max_depth = 0
    cum_depth = 0
    leaf_count = 0
    per_depth_counts: dict[int, int] = {}

    # Stats that require record_stats=True
    avg_candidates = 0
    avg_inequalities = 0
    avg_vertices = 0
    per_depth_candidate_sums: dict[int, int] = {}
    per_depth_inequality_sums: dict[int, int] = {}
    per_depth_vertex_sums: dict[int, int] = {}
    per_depth_moments_candidates: dict[Any, dict[int, float]] = {
        alpha: {} for alpha in alphas
    }
    per_depth_moments_inequalities: dict[Any, dict[int, float]] = {
        alpha: {} for alpha in alphas
    }
    per_depth_moments_vertices: dict[Any, dict[int, float]] = {
        alpha: {} for alpha in alphas
    }

    stack = [tree.root]
    while stack:
        node = stack.pop()
        total_nodes += 1

        # Update per-depth counts
        per_depth_counts[node.depth] = per_depth_counts.get(node.depth, 0) + 1

        # Only compute detailed stats if they were recorded
        if has_stats:
            avg_candidates += node.n_candidates
            avg_inequalities += node.n_inequalities
            avg_vertices += node.n_vertices

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
                else:
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

    # Build result dictionary - basic stats always available
    _stat_dict: dict[str, Any] = {
        "total_nodes": total_nodes,
        "leaf_count": leaf_count,
        "avg_depth": avg_depth,
        "max_depth": max_depth,
        "has_detailed_stats": has_stats,
    }

    if include_per_depth_stats:
        _stat_dict["per_depth_nodes"] = per_depth_counts

    # Add detailed stats only if recorded
    if has_stats:
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
                    per_depth_moments_candidates[alpha][depth] = float("nan")
            try:
                for depth in per_depth_moments_inequalities[alpha]:
                    count = per_depth_counts[depth]
                    per_depth_moments_inequalities[alpha][depth] /= count
                    per_depth_moments_inequalities[alpha][depth] **= 1 / alpha
            except ZeroDivisionError:
                for depth in per_depth_moments_inequalities[alpha]:
                    per_depth_moments_inequalities[alpha][depth] = float("nan")
            try:
                for depth in per_depth_moments_vertices[alpha]:
                    count = per_depth_counts[depth]
                    per_depth_moments_vertices[alpha][depth] /= count
                    per_depth_moments_vertices[alpha][depth] **= 1 / alpha
            except ZeroDivisionError:
                for depth in per_depth_moments_vertices[alpha]:
                    per_depth_moments_vertices[alpha][depth] = float("nan")

        _stat_dict.update(
            {
                "avg_candidates": avg_candidates,
                "avg_inequalities": avg_inequalities,
                "avg_vertices": avg_vertices,
            }
        )

        if include_per_depth_stats:
            _stat_dict.update(
                {
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
            )

    return _stat_dict


def print_ppart_stats(
    tree: PartitionTree, include_per_depth_stats: bool = False
) -> None:
    """Print the statistics of the partition tree in a readable format.

    Args:
        tree: Partition tree to analyze. For detailed statistics, the tree
            must have been created with record_stats=True.
        include_per_depth_stats: Include per-depth breakdowns.
    """
    stat_dict = get_ppart_stats(tree, include_per_depth_stats=include_per_depth_stats)

    # Parse floats for better readability
    for key, value in stat_dict.items():
        if isinstance(value, float):
            stat_dict[key] = round(value, 4)

    print(json.dumps(stat_dict, indent=4))
