"""
Tests for saving and loading partition trees.
"""

import os
import tempfile

import numpy as np

from polypart.ppart import PartitionTree
from polypart.geometry import Hyperplane, Polytope
from polypart.io import save_tree, load_tree
from polypart.ppart import build_partition_tree


def make_simple_tree() -> PartitionTree:
    """Create a simple partition tree for testing."""
    # unit square in 2D: inequalities x>=0, x<=1, y>=0, y<=1 -> expressed as A x <= b
    A = [[-1, 0], [1, 0], [0, -1], [0, 1]]
    b = [0, 1, 0, 1]
    square = Polytope(A, b)
    square.extreme()

    # two axis-aligned hyperplanes: x = 0.2 and y = 0.2
    h1 = Hyperplane.from_coefficients([1, 0, 0.2])
    h2 = Hyperplane.from_coefficients([0, 1, 0.2])

    tree, _ = build_partition_tree(square, [h1, h2])

    return tree


def test_save_load_roundtrip():
    """Test saving and loading a simple partition tree."""
    tree = make_simple_tree()
    tmpdir = tempfile.mkdtemp()
    path = os.path.join(tmpdir, "tree.json")

    save_tree(tree, path)
    loaded = load_tree(path)

    # Compare structures iterating the trees in parallel,
    # checking if they have the same cut_hyperplane and children
    nodes1 = [tree.root]
    nodes2 = [loaded.root]
    while nodes1 and nodes2:
        n1 = nodes1.pop()
        n2 = nodes2.pop()
        assert (n1.cut is None) == (n2.cut is None)
        if n1.cut is not None and n2.cut is not None:
            assert np.array_equal(n1.cut.normal, n2.cut.normal)
            assert n1.cut.offset == n2.cut.offset
        assert len(n1.children) == len(n2.children)
        nodes1.extend(n1.children)
        nodes2.extend(n2.children)
