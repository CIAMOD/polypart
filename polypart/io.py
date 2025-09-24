from __future__ import annotations

import json
import os
from fractions import Fraction

import numpy as np

from .ppart import PartitionNode, PartitionTree
from .geometry import Hyperplane


def _frac_to_str(x: Fraction) -> str:
    return f"{x.numerator}/{x.denominator}" if x.denominator != 1 else f"{x.numerator}"


def _str_to_frac(s: str) -> Fraction:
    if "/" in s:
        num, den = s.split("/", 1)
        return Fraction(int(num), int(den))
    return Fraction(int(s), 1)


def vector_to_str(vec) -> str:
    """Convert a sequence/array of Fractions to a compact bracketed string.

    Args:
        vec: sequence or array of Fraction objects.

    Returns:
        Compact single-line string like ``[a/b, c/d, ...]``.
    """
    pieces = (_frac_to_str(v) for v in list(vec))
    return "[" + ", ".join(pieces) + "]"


def str_to_vector(s: str) -> np.ndarray:
    """Parse a compact bracketed vector string into a numpy array of Fractions.

    Args:
        s: string like ``[1/2,3,4/5]`` or ``1/2,3,4/5``.

    Returns:
        1-D numpy array (dtype=object) of Fraction objects.
    """
    if s is None or s == "":
        return np.empty((0,), dtype=object)
    s = s.strip()
    if s.startswith("[") and s.endswith("]"):
        s = s[1:-1]
    parts = [p.strip() for p in s.split(",") if p.strip() != ""]
    return np.array([_str_to_frac(p) for p in parts], dtype=object)


def save_tree(root: PartitionTree, path: str) -> None:
    """Save a PartitionTree to a JSON file.

    Args:
        root: PartitionTree or root PartitionNode to serialize.
        path: output file path where JSON will be written.
    """
    tree_json = {
        "n_partitions": 0,
        "n_nodes": 0,
        "max_depth": 0,
        "avg_depth": 0,
        "tree": [],
    }

    queue = [root.root if isinstance(root, PartitionTree) else root]
    while queue:
        node = queue.pop(0)
        # count node
        tree_json["n_nodes"] += 1
        # assign temporary index
        node.index = len(tree_json["tree"])
        tree_json["tree"].append(
            {
                "depth": node.depth,
                "cut_hyperplane": (
                    [vector_to_str(node.cut.normal), _frac_to_str(node.cut.offset)]
                    if node.cut is not None
                    else None
                ),
                "parent_idx": node.parent.index if node.parent is not None else None,
                "centroid": (
                    vector_to_str(node.centroid_)
                    if node.centroid_ is not None
                    else None
                ),
            }
        )
        if len(node.children) == 0:
            tree_json["n_partitions"] += 1
            tree_json["max_depth"] = max(tree_json["max_depth"], node.depth)
            tree_json["avg_depth"] += node.depth

        # enqueue children
        queue.extend(node.children)

    if tree_json["n_partitions"] > 0:
        tree_json["avg_depth"] = round(
            tree_json["avg_depth"] / tree_json["n_partitions"], 2
        )
    else:
        tree_json["avg_depth"] = 0

    OUT_FOLDER = os.path.dirname(path) or "."
    if not os.path.exists(OUT_FOLDER):
        os.makedirs(OUT_FOLDER, exist_ok=True)

    out_path = path
    with open(out_path, "w") as f:
        json.dump(tree_json, f, indent=4)


def load_tree(path: str) -> PartitionTree:
    """Load a PartitionTree previously written with :func:`save_tree`.

    Args:
        path: JSON file path produced by :func:`save_tree`.

    Returns:
        Reconstructed PartitionTree. Polytope and candidate fields are left
        as ``None`` (only structural data required for classification is
        restored).
    """
    with open(path, "r") as f:
        data = json.load(f)

    nodes = []
    # first pass: create nodes without wiring
    for n in data.get("tree", []):
        node = PartitionNode(
            polytope=None, candidates=None, parent=None, depth=n["depth"]
        )
        # centroid
        if n.get("centroid") is not None:
            node.centroid_ = str_to_vector(n["centroid"])
        # cut
        ch = n.get("cut_hyperplane")
        if ch is not None:
            normal = str_to_vector(ch[0])
            offset = _str_to_frac(ch[1])
            node.cut = Hyperplane(normal, offset)
        nodes.append(node)

    # second pass: wire parent/children
    for idx, n in enumerate(data.get("tree", [])):
        parent_idx = n.get("parent_idx")
        if parent_idx is not None:
            nodes[idx].parent = nodes[parent_idx]
            nodes[parent_idx].children.append(nodes[idx])

    root = (
        nodes[0]
        if nodes
        else PartitionNode(polytope=None, candidates=None, parent=None, depth=0)
    )
    n_regions = data.get("n_partitions", 0)
    return PartitionTree(root, n_regions)
