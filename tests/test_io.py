import json
import os
import tempfile

import numpy as np
import pytest

from polypart import (
    PartitionNode,
    PartitionTree,
    build_partition_tree,
    load_tree,
    save_tree,
)
from polypart.algorithms.incenu import build_incenu_tree
from polypart.core.typing import Fraction
from polypart.generators.arrangements import get_braid_arrangement
from polypart.generators.polytopes import get_hypercube


def test_save_load_roundtrip():
    """Test that saving and loading a tree preserves structure and data."""
    d = 4
    P = get_hypercube(d)
    A = get_braid_arrangement(d)

    original_tree, _ = build_partition_tree(A, P, record_stats=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "test_tree.json")
        save_tree(original_tree, filepath)

        assert os.path.exists(filepath)

        loaded_tree = load_tree(filepath)

        # Verify structure matches
        o_nodes = []
        l_nodes = []

        q = [original_tree.root]
        while q:
            n = q.pop(0)
            o_nodes.append(n)
            q.extend(n.children)

        q = [loaded_tree.root]
        while q:
            n = q.pop(0)
            l_nodes.append(n)
            q.extend(n.children)

        assert len(o_nodes) == len(l_nodes)

        for o, l in zip(o_nodes, l_nodes):
            assert o.depth == l.depth
            assert len(o.children) == len(l.children)
            if o.cut is not None:
                # Compare cut hyperplane
                assert np.all(o.cut.normal == l.cut.normal)
                assert o.cut.offset == l.cut.offset
            else:
                assert l.cut is None

            # Check seed if leaf (both ppart and incenu store as "seed")
            if o.is_leaf:
                o_seed = o.data.get("seed") if o.data else None
                l_seed = l.data.get("seed") if l.data else None

                if o_seed is not None:
                    assert l_seed is not None
                    # Compare values
                    for v1, v2 in zip(o_seed, l_seed):
                        assert v1 == v2


def test_incenu_tree_roundtrip():
    """Test that incenu trees save and load correctly with seed data."""
    d = 3
    P = get_hypercube(d)
    A = get_braid_arrangement(d)

    original_tree, n_parts = build_incenu_tree(A, P)

    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "incenu_tree.json")
        save_tree(original_tree, filepath)
        loaded_tree = load_tree(filepath)

        # Count leaves and verify seed data
        o_leaves = []
        l_leaves = []

        q = [original_tree.root]
        while q:
            n = q.pop(0)
            if n.is_leaf:
                o_leaves.append(n)
            q.extend(n.children)

        q = [loaded_tree.root]
        while q:
            n = q.pop(0)
            if n.is_leaf:
                l_leaves.append(n)
            q.extend(n.children)

        assert len(o_leaves) == len(l_leaves) == n_parts

        # Verify all leaves have seeds preserved
        for o, l in zip(o_leaves, l_leaves):
            o_seed = o.data.get("seed") if o.data else None
            l_seed = l.data.get("seed") if l.data else None
            assert o_seed is not None, "incenu should always produce seeds"
            assert l_seed is not None, "loaded tree should preserve seeds"
            for v1, v2 in zip(o_seed, l_seed):
                assert v1 == v2


def test_json_structure():
    """Verify the JSON keys are explicitly what we expect for external tools compatibility."""
    d = 2
    P = get_hypercube(d)
    A = get_braid_arrangement(d)
    tree, _ = build_partition_tree(A, P)

    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "structure.json")
        save_tree(tree, filepath)

        with open(filepath, "r") as f:
            data = json.load(f)

        assert "n_partitions" in data
        assert "n_nodes" in data
        assert "max_depth" in data
        assert "avg_depth" in data
        assert "tree" in data
        assert isinstance(data["tree"], list)

        # Check node structure
        node_0 = data["tree"][0]
        assert "idx" in node_0
        assert "depth" in node_0
        assert "cut_hyperplane" in node_0
        assert "parent_idx" in node_0
        assert "seed" in node_0

        # Verify idx is correctly set
        for i, node in enumerate(data["tree"]):
            assert node["idx"] == i


def test_idx_attribute_preserved():
    """Test that idx attribute is set on nodes after save and load."""
    d = 2
    P = get_hypercube(d)
    A = get_braid_arrangement(d)
    tree, _ = build_partition_tree(A, P)

    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "idx_test.json")
        save_tree(tree, filepath)

        # Verify idx was set on original nodes during save
        q = [tree.root]
        idx = 0
        while q:
            n = q.pop(0)
            assert n.idx == idx
            idx += 1
            q.extend(n.children)

        # Load and verify idx is restored
        loaded_tree = load_tree(filepath)
        q = [loaded_tree.root]
        idx = 0
        while q:
            n = q.pop(0)
            assert n.idx == idx
            idx += 1
            q.extend(n.children)


def test_classify_after_load():
    """Test that classify works correctly on loaded tree."""
    d = 3
    P = get_hypercube(d)
    A = get_braid_arrangement(d)
    tree, _ = build_partition_tree(A, P)

    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "classify_test.json")
        save_tree(tree, filepath)
        loaded_tree = load_tree(filepath)

        # Test classification with a few random points
        test_points = [
            [Fraction(1, 4), Fraction(1, 4), Fraction(1, 4)],
            [Fraction(3, 4), Fraction(1, 4), Fraction(1, 2)],
            [Fraction(1, 2), Fraction(1, 2), Fraction(1, 2)],
        ]

        for point in test_points:
            original_leaf = tree.classify(point)
            loaded_leaf = loaded_tree.classify(point)
            # Seeds should match (same region)
            o_seed = original_leaf.data.get("seed") if original_leaf.data else None
            l_seed = loaded_leaf.data.get("seed") if loaded_leaf.data else None
            if o_seed is not None and l_seed is not None:
                for v1, v2 in zip(o_seed, l_seed):
                    assert v1 == v2
