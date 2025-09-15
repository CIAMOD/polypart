import os
import tempfile
import numpy as np
from fractions import Fraction

from polypart.ppart import PartitionNode, PartitionTree
from polypart.geometry import Hyperplane
from polypart.io import save_tree, load_tree


def make_simple_tree():
    # create a root node with two children (one leaf, one internal)
    v1 = np.array([Fraction(1, 2), Fraction(1, 3)], dtype=object)
    v2 = np.array([Fraction(2, 3), Fraction(1, 4)], dtype=object)
    h = Hyperplane(np.array([Fraction(1), Fraction(-1)], dtype=object), Fraction(0))

    root = PartitionNode(polytope=None, candidates=None, parent=None, depth=0)
    root.centroid_ = v1
    root.cut = h

    c1 = PartitionNode(polytope=None, candidates=None, parent=root, depth=1)
    c1.centroid_ = v2
    c2 = PartitionNode(polytope=None, candidates=None, parent=root, depth=1)

    root.children = [c1, c2]
    return PartitionTree(root, n_regions=2)


def test_save_load_roundtrip():
    tree = make_simple_tree()
    tmpdir = tempfile.mkdtemp()
    path = os.path.join(tmpdir, "tree.json")

    save_tree(tree, path)
    loaded = load_tree(path)

    # Check top-level counts
    assert loaded.n_regions == tree.n_regions
    assert loaded.root.depth == tree.root.depth

    # Check centroid round-trip for root
    orig_cent = tree.root.centroid_
    loaded_cent = loaded.root.centroid_
    assert np.array_equal(orig_cent, loaded_cent)

    # Check cut hyperplane round-trip
    assert loaded.root.cut is not None
    assert np.array_equal(loaded.root.cut.normal, tree.root.cut.normal)
    assert loaded.root.cut.offset == tree.root.cut.offset
