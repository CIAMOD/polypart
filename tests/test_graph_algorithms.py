"""Tests for symmetry reduction and graph algorithms."""

from __future__ import annotations

from collections import Counter
from functools import partial

import numpy as np
import pytest

from polypart import (
    PartitionGraph,
    StorageLevel,
    SymmetryGroup,
    build_incenu_tree,
    build_partition_tree,
)
from polypart.apps.moduli import (
    basic_transformation,
    generate_d_invariant_transformations,
)
from polypart.core.typing import FractionVector, as_fraction_vector
from polypart.generators import get_moduli_arrangement, get_product_of_simplices


def _apply_wrapper(
    vector: FractionVector,
    sigma: tuple[int, ...],
    s: int,
    H: tuple[int, ...],
    n: int,
    r: int,
    d: int,
) -> FractionVector:
    """Reshape flat seed vector, apply transformation, flatten back."""
    zeros = as_fraction_vector(np.zeros(n)).reshape(n, 1)
    alphas = np.hstack((zeros, vector.reshape(n, r - 1))).reshape(1, n, r)
    d_arr = np.array([d])

    new_alphas, _ = basic_transformation(alphas, d_arr, sigma, s, H)

    return new_alphas[0, :, 1:].flatten()


def get_moduli_symmetries(n: int, r: int, d: int) -> SymmetryGroup:
    """Create SymmetryGroup from moduli space transformations."""
    transforms = {}
    generator = generate_d_invariant_transformations(n, r, d)

    for i, (sigma, s, H) in enumerate(generator):
        name = f"T_{i}_sig{sigma}_s{s}_H{H}"
        func = partial(_apply_wrapper, sigma=sigma, s=s, H=H, n=n, r=r, d=d)
        transforms[name] = func

    return SymmetryGroup(transforms)


@pytest.mark.parametrize(
    "n,r",
    [
        (1, 2),
        (1, 3),
        (1, 4),
        (1, 5),
        (1, 6),
        (2, 2),
        (2, 3),
        (2, 4),
        (3, 2),
        (3, 3),
        (4, 2),
    ],
)
def test_moduli_symmetry_reduction_consistency(n: int, r: int):
    """Test that ppart and incenu produce same number of chambers for moduli cases."""
    # Create polytope and arrangement
    polytope = get_product_of_simplices(n, r - 1)
    arrangement = get_moduli_arrangement(n, r, 0)

    # Build partition trees with both algorithms
    tree_ppart, n_chambers_ppart = build_partition_tree(arrangement, polytope)
    tree_incenu, n_chambers_incenu = build_incenu_tree(arrangement, polytope)

    # They should find the same number of chambers
    assert n_chambers_ppart == n_chambers_incenu

    # Create symmetry group
    symmetries = get_moduli_symmetries(n, r, 0)

    # Test that ppart tree can be reduced with symmetries
    graph_ppart = PartitionGraph(tree_ppart, symmetries)
    n_classes_ppart = graph_ppart.reduce(
        storage=StorageLevel.STABILIZERS, verbose=False
    )

    # Number of classes should be between 1 and n_chambers
    assert 1 <= n_classes_ppart <= n_chambers_ppart

    # Total members across all classes should equal n_chambers
    total_members = sum(cls.size for cls in graph_ppart.classes)
    assert total_members == n_chambers_ppart

    # Test incenu tree reduction as well
    graph_incenu = PartitionGraph(tree_incenu, symmetries)
    n_classes_incenu = graph_incenu.reduce(
        storage=StorageLevel.STABILIZERS, verbose=False
    )
    assert 1 <= n_classes_incenu <= n_chambers_incenu
    total_members_incenu = sum(cls.size for cls in graph_incenu.classes)
    assert total_members_incenu == n_chambers_incenu

    assert n_classes_ppart == n_classes_incenu

    # Assert distribution of stabilizer sizes matches between both graphs
    stabilizer_sizes_ppart = Counter(
        len(cls.stabilizers) for cls in graph_ppart.classes
    )
    stabilizer_sizes_incenu = Counter(
        len(cls.stabilizers) for cls in graph_incenu.classes
    )
    assert stabilizer_sizes_ppart == stabilizer_sizes_incenu


@pytest.mark.parametrize("r", [2, 3, 4, 5, 6])  # Keep small for performance
def test_moduli_n1_small_cases(r: int):
    """Test specific moduli n=1 cases with detailed verification."""
    n = 1
    d = 0

    polytope = get_product_of_simplices(n, r - 1)
    arrangement = get_moduli_arrangement(n, r, d)
    symmetries = get_moduli_symmetries(n, r, d)

    # Test ppart algorithm with different storage levels
    tree, n_chambers = build_partition_tree(arrangement, polytope)
    graph = PartitionGraph(tree, symmetries)

    # Test different storage levels
    for storage in [
        StorageLevel.COUNT_ONLY,
        StorageLevel.SEEDS,
        StorageLevel.STABILIZERS,
        StorageLevel.FULL,
    ]:
        n_classes = graph.reduce(storage=storage)
        assert n_classes > 0

        if storage >= StorageLevel.SEEDS:
            assert len(graph.classes) == n_classes
            assert all(cls.size >= 1 for cls in graph.classes)

        if storage >= StorageLevel.STABILIZERS:
            # Each class should have some stabilizers (at least the identity)
            assert all(len(cls.stabilizers) >= 1 for cls in graph.classes)

        if storage == StorageLevel.FULL:
            # Each class should have mappings for each transformation
            for cls in graph.classes:
                if cls.seed is not None:
                    # Should have mapping entries for relevant transformations
                    assert isinstance(cls.mappings, dict)


def test_symmetry_group_edge_cases():
    """Test SymmetryGroup with various input formats."""
    # Test with empty group
    empty_group = SymmetryGroup({})
    assert len(empty_group) == 0

    # Test with single transformation
    def identity(x):
        return x

    single_group = SymmetryGroup({"id": identity})
    assert len(single_group) == 1
    assert "id" in single_group.names()

    # Test with list of functions (auto-named)
    list_group = SymmetryGroup([identity, identity])
    assert len(list_group) == 2
    assert "sym_0" in list_group.names()
    assert "sym_1" in list_group.names()

    # Test with list of tuples
    tuple_group = SymmetryGroup([("first", identity), ("second", identity)])
    assert len(tuple_group) == 2
    assert "first" in tuple_group.names()
    assert "second" in tuple_group.names()


def test_partition_graph_error_handling():
    """Test error conditions in PartitionGraph."""
    n, r, d = 1, 2, 0
    polytope = get_product_of_simplices(n, r - 1)
    arrangement = get_moduli_arrangement(n, r, d)
    tree, _ = build_partition_tree(arrangement, polytope)

    symmetries = SymmetryGroup({})
    graph = PartitionGraph(tree, symmetries)

    # Should raise error when accessing classes before reduction
    with pytest.raises(RuntimeError, match="Call reduce"):
        _ = graph.classes

    with pytest.raises(RuntimeError, match="Call reduce"):
        graph.to_dict()

    # After reduction, should work
    n_classes = graph.reduce()
    assert n_classes >= 0
    _ = graph.classes  # Should not raise
    _ = graph.to_dict()  # Should not raise


def test_equivalence_class_serialization():
    """Test EquivalenceClass serialization to dict."""
    n, r, d = 2, 4, 0
    polytope = get_product_of_simplices(n, r - 1)
    arrangement = get_moduli_arrangement(n, r, d)
    tree, _ = build_partition_tree(arrangement, polytope)

    symmetries = get_moduli_symmetries(n, r, d)
    graph = PartitionGraph(tree, symmetries)
    graph.reduce(storage=StorageLevel.FULL)

    # Test that each class can be serialized
    for cls in graph.classes:
        data = cls.to_dict()
        assert "class_id" in data
        assert "seed" in data
        assert "size" in data
        assert "stabilizers" in data
        assert data["class_id"] == cls.id
        assert data["size"] == cls.size
        assert data["stabilizers"] == cls.stabilizers
