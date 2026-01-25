"""Functions to save and load PartitionTree objects to/from JSON files."""

from __future__ import annotations

import json
import os

import numpy as np

from polypart.core.geometry import Hyperplane
from polypart.core.tree import PartitionNode, PartitionTree
from polypart.core.typing import Fraction


def _frac_to_str(x: Fraction) -> str:
    return f"{x.numerator}/{x.denominator}" if x.denominator != 1 else f"{x.numerator}"


def _str_to_frac(s: str) -> Fraction:
    if "/" in s:
        num, den = s.split("/", 1)
        return Fraction(int(num), int(den))
    return Fraction(int(s), 1)


def vector_to_str(vec) -> str:
    """Convert array of Fractions to bracketed string."""
    pieces = (_frac_to_str(v) for v in list(vec))
    return "[" + ", ".join(pieces) + "]"


def str_to_vector(s: str) -> np.ndarray:
    """Parse bracketed vector string into numpy array of Fractions."""
    if s is None or s == "":
        return np.empty((0,), dtype=object)
    s = s.strip()
    if s.startswith("[") and s.endswith("]"):
        s = s[1:-1]
    parts = [p.strip() for p in s.split(",") if p.strip() != ""]
    return np.array([_str_to_frac(p) for p in parts], dtype=object)


def save_tree(root: PartitionTree, path: str) -> None:
    """Save a PartitionTree to JSON file with structural statistics.

    Args:
        root: The PartitionTree to save.
        path: File path to save the JSON file to.

    The saved JSON includes:
        - n_partitions: Number of leaf nodes (chambers/regions).
        - n_nodes: Total number of nodes in the tree.
        - max_depth: Maximum depth of any leaf node.
        - avg_depth: Average depth of leaf nodes.
        - tree: List of node records in BFS order, each containing:
            - idx: Node index in this list.
            - depth: Node depth.
            - cut_hyperplane: [normal_vector, offset] for internal nodes.
            - parent_idx: Index of parent node (None for root).
            - seed: Representative point for leaf nodes (from ppart or incenu).
    """
    tree_json: dict = {
        "n_partitions": 0,
        "n_nodes": 0,
        "max_depth": 0,
        "avg_depth": 0,
        "tree": [],
    }

    queue = [root.root if isinstance(root, PartitionTree) else root]

    while queue:
        node = queue.pop(0)
        tree_json["n_nodes"] += 1
        node.idx = len(tree_json["tree"])

        # Extract seed from leaf nodes (both ppart and incenu store it as "seed")
        seed = None
        if node.is_leaf and node.data and isinstance(node.data, dict):
            seed_vec = node.data.get("seed")
            if seed_vec is not None:
                seed = vector_to_str(seed_vec)

        tree_json["tree"].append(
            {
                "idx": node.idx,
                "parent_idx": node.parent.idx if node.parent is not None else None,
                "depth": node.depth,
                "cut_hyperplane": (
                    [vector_to_str(node.cut.normal), _frac_to_str(node.cut.offset)]
                    if node.cut is not None
                    else None
                ),
                "seed": seed,
            }
        )

        if node.is_leaf:
            tree_json["n_partitions"] += 1
            tree_json["max_depth"] = max(tree_json["max_depth"], node.depth)
            tree_json["avg_depth"] += node.depth

        queue.extend(node.children)

    if tree_json["n_partitions"] > 0:
        tree_json["avg_depth"] = round(
            tree_json["avg_depth"] / tree_json["n_partitions"], 2
        )
    else:
        tree_json["avg_depth"] = 0

    out_folder = os.path.dirname(path) or "."
    if not os.path.exists(out_folder):
        os.makedirs(out_folder, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(tree_json, f, indent=4)


def load_tree(path: str) -> PartitionTree:
    """Load a PartitionTree from JSON file.

    Args:
        path: File path to the JSON file.

    Returns:
        The loaded PartitionTree with node structure and seed data restored.

    Raises:
        ValueError: If the loaded tree has no nodes.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    nodes = []
    for n in data.get("tree", []):
        node = PartitionNode(parent=None, depth=n["depth"])

        # Restore the idx if present (for consistency with saved data)
        if "idx" in n:
            node.idx = n["idx"]

        # Restore seed data for leaf nodes
        if n.get("seed") is not None:
            node.data = {"seed": str_to_vector(n["seed"])}

        # Restore cut hyperplane for internal nodes
        ch = n.get("cut_hyperplane")
        if ch is not None:
            normal = str_to_vector(ch[0])
            offset = _str_to_frac(ch[1])
            node.cut = Hyperplane(normal, offset)

        nodes.append(node)

    if len(nodes) == 0:
        raise ValueError("Loaded tree has no nodes.")

    # Rebuild parent-child relationships
    for idx, n in enumerate(data.get("tree", [])):
        parent_idx = n.get("parent_idx")
        if parent_idx is not None:
            nodes[parent_idx].add_child(nodes[idx])

    return PartitionTree(nodes[0])
