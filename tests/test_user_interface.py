"""Tests for user interface and typical end-user workflows."""

import os
import tempfile

import numpy as np
import pytest

from polypart import (
    Arrangement,
    HalfSpace,
    Hyperplane,
    PartitionGraph,
    Polyhedron,
    Polytope,
    StorageLevel,
    SymmetryGroup,
    build_incenu_tree,
    build_partition_tree,
    load_tree,
    number_of_regions,
    save_tree,
)
from polypart.core.typing import as_fraction_vector
from polypart.generators import (
    get_braid_arrangement,
    get_hypercube,
    get_moduli_arrangement,
    get_product_of_simplices,
    get_random_arrangement,
    get_resonance_arrangement,
    get_simplex,
    sample_circumscribed_polytope,
    sample_poisson_zero_cell_polytope,
)


class TestPolytopecreation:
    """Test various ways users can create polytopes."""

    def test_hypercube_creation(self):
        """Test hypercube creation with different parameters."""
        # Standard unit hypercube
        P = get_hypercube(3)
        assert P.dim == 3
        P.extreme()
        assert P.n_vertices == 8

        # Centered hypercube
        from polypart.generators import get_centered_hypercube

        P_centered = get_centered_hypercube(2, r=2)
        P_centered.extreme()
        assert P_centered.n_vertices == 4

    def test_simplex_creation(self):
        """Test simplex creation in various dimensions."""
        for d in range(2, 5):
            P = get_simplex(d)
            assert P.dim == d
            P.extreme()
            assert P.n_vertices == d + 1

    def test_product_of_simplices(self):
        """Test product of simplices for moduli spaces."""
        # Product of two 2-simplices (triangles)
        P = get_product_of_simplices(2, 2)
        assert P.dim == 4  # 2*2
        P.extreme()
        assert P.n_vertices == 9  # 3*3

    def test_random_polytope_generation(self):
        """Test random polytope generation."""
        # Circumscribed polytope
        P = sample_circumscribed_polytope(d=3, m=8, seed=42)
        P.extreme()
        assert P.dim == 3
        assert P.n_vertices > 0

        # Poisson zero cell polytope
        P_poisson = sample_poisson_zero_cell_polytope(
            d=2, intensity=1.0, window_radius=5.0, seed=42
        )
        P_poisson.extreme()
        assert P_poisson.dim == 2

    def test_manual_polytope_creation(self):
        """Test creating polytopes manually from constraints."""
        # Unit square: 0 <= x <= 1, 0 <= y <= 1
        A = np.array(
            [
                [1, 0],  # x <= 1
                [-1, 0],  # x >= 0 (i.e., -x <= 0)
                [0, 1],  # y <= 1
                [0, -1],  # y >= 0 (i.e., -y <= 0)
            ]
        )
        b = np.array([1, 0, 1, 0])
        P = Polytope(A, b)
        P.extreme()
        assert P.n_vertices == 4


class TestArrangementCreation:
    """Test various ways users can create hyperplane arrangements."""

    def test_braid_arrangement(self):
        """Test braid arrangement creation."""
        for d in range(2, 5):
            A = get_braid_arrangement(d)
            assert len(A) == (d * (d - 1)) // 2  # Number of pairs

    def test_moduli_arrangement(self):
        """Test moduli arrangement for different parameters."""
        A = get_moduli_arrangement(n=2, r=3, d=0)
        assert len(A) > 0
        assert isinstance(A, Arrangement)

    def test_resonance_arrangement(self):
        """Test resonance arrangement."""
        A = get_resonance_arrangement(d=3)
        assert len(A) > 0

    def test_random_arrangement(self):
        """Test random arrangement generation."""
        P = get_simplex(3)
        A = get_random_arrangement(P, m=5, seed=42)
        assert len(A) == 5

    def test_manual_arrangement_creation(self):
        """Test creating arrangements manually."""
        # Create individual hyperplanes
        h1 = Hyperplane([1, 0], 0.5)  # x = 0.5
        h2 = Hyperplane([0, 1], 0.5)  # y = 0.5
        h3 = Hyperplane([1, 1], 1.0)  # x + y = 1

        # Create arrangement from list
        A = Arrangement([h1, h2, h3])
        assert len(A) == 3

        # Create hyperplane from coefficients
        h4 = Hyperplane.from_coefficients([1, -1, 0])  # x - y = 0
        assert h4.offset == 0
        assert h4.normal[0] == 1 and h4.normal[1] == -1


class TestEndToEndWorkflows:
    """Test complete end-to-end workflows that users would follow."""

    def test_basic_partitioning_workflow(self):
        """Test basic partitioning workflow: create polytope + arrangement + partition."""
        # Step 1: Create polytope
        P = get_hypercube(3)

        # Step 2: Create arrangement
        A = get_braid_arrangement(3)

        # Step 3: Partition with different algorithms
        tree_ppart, n_ppart = build_partition_tree(A, P)
        tree_incenu, n_incenu = build_incenu_tree(A, P)
        n_delres = number_of_regions(A, P)

        # Step 4: Verify consistency
        assert n_ppart == n_incenu == n_delres
        assert tree_ppart.root is not None
        assert tree_incenu.root is not None

    def test_moduli_space_workflow(self):
        """Test complete moduli space analysis workflow."""
        # Step 1: Set parameters
        n, r, d = 1, 3, 0

        # Step 2: Create moduli polytope and arrangement
        polytope = get_product_of_simplices(n, r - 1)
        arrangement = get_moduli_arrangement(n, r, d)

        # Step 3: Partition the space
        tree, n_chambers = build_partition_tree(arrangement, polytope)

        # Step 4: Create symmetry group (simplified)
        def identity_transform(x):
            return x

        def swap_transform(x):
            # Simple coordinate swap for demonstration
            if len(x) >= 2:
                result = x.copy()
                result[0], result[1] = result[1], result[0]
                return result
            return x

        symmetries = SymmetryGroup(
            {"identity": identity_transform, "swap": swap_transform}
        )

        # Step 5: Reduce to equivalence classes
        graph = PartitionGraph(tree, symmetries)
        n_classes = graph.reduce(storage=StorageLevel.STABILIZERS)

        # Step 6: Analyze results
        assert n_classes > 0
        assert n_classes <= n_chambers

    def test_save_load_workflow(self):
        """Test saving and loading partition trees."""
        # Create and partition
        P = get_hypercube(2)
        A = get_braid_arrangement(2)
        tree, _ = build_partition_tree(A, P, record_stats=True)

        # Save to temporary file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_path = f.name

        try:
            save_tree(tree, temp_path)

            # Load back
            loaded_tree = load_tree(temp_path)

            # Verify structure is preserved
            original_nodes = []
            loaded_nodes = []
            queue = [tree.root]
            while queue:
                node = queue.pop(0)
                original_nodes.append(node)
                queue.extend(node.children)
            queue = [loaded_tree.root]
            while queue:
                node = queue.pop(0)
                loaded_nodes.append(node)
                queue.extend(node.children)
            assert len(original_nodes) == len(loaded_nodes)
            for onode, lnode in zip(original_nodes, loaded_nodes):
                assert onode.depth == lnode.depth
                assert len(onode.children) == len(lnode.children)
                if onode.cut is not None:
                    assert lnode.cut is not None
                    assert np.all(onode.cut.normal == lnode.cut.normal)
                    assert onode.cut.offset == lnode.cut.offset
                else:
                    assert lnode.cut is None
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_algorithm_comparison_workflow(self):
        """Test workflow for comparing different algorithms."""
        P2 = get_simplex(3)
        A2 = get_random_arrangement(P2, m=5, seed=42)
        test_cases = [
            (get_hypercube(2), get_braid_arrangement(2)),
            (P2, A2),
            (get_product_of_simplices(3, 2), get_moduli_arrangement(3, 3, 0)),
        ]

        for polytope, arrangement in test_cases:
            # Test all three algorithms
            tree_ppart, n_ppart = build_partition_tree(arrangement, polytope)
            tree_incenu, n_incenu = build_incenu_tree(arrangement, polytope)
            n_delres = number_of_regions(arrangement, polytope)

            # Verify consistency
            assert n_ppart == n_incenu == n_delres

            # Verify tree structures
            assert tree_ppart.root is not None
            assert tree_incenu.root is not None


class TestErrorHandling:
    """Test proper error handling for common user mistakes."""

    def test_invalid_polytope_arrangement_combinations(self):
        """Test error handling for mismatched dimensions."""
        A_3d = get_braid_arrangement(3)
        P_2d = get_hypercube(2)

        # The algorithms should handle dimension mismatches gracefully
        try:
            tree, n = build_partition_tree(A_3d, P_2d)
        except Exception as e:
            # If exception occurs, it should be informative
            assert "dimension" in str(e).lower() or "incompatible" in str(e).lower()

    def test_empty_arrangements(self):
        """Test handling of empty arrangements."""
        P = get_hypercube(2)
        A_empty = Arrangement([])
        _, n_incenu = build_incenu_tree(A_empty, P)
        n_delres = number_of_regions(A_empty, P)
        _, n_ppart = build_partition_tree(A_empty, P)
        assert n_incenu == n_delres == n_ppart == 1

    def test_invalid_symmetry_group_inputs(self):
        """Test error handling for invalid symmetry group inputs."""
        with pytest.raises(ValueError):
            SymmetryGroup([])  # Empty iterable

        with pytest.raises(ValueError):
            SymmetryGroup([1, 2, 3])  # Non-callable elements


class TestPerformanceAndScalability:
    """Test performance characteristics and scalability."""

    def test_small_scale_performance(self):
        """Test that small problems complete quickly."""
        import time

        P = get_hypercube(3)
        A = get_braid_arrangement(3)

        start_time = time.time()
        tree, n_chambers = build_partition_tree(A, P)
        elapsed = time.time() - start_time

        # Should complete in reasonable time
        assert elapsed < 1.0
        assert n_chambers > 0

    def test_memory_efficiency(self):
        """Test that trees don't hold onto unnecessary data."""
        P = get_hypercube(2)
        A = get_braid_arrangement(2)

        tree, _ = build_partition_tree(A, P)

        # After partitioning, internal polytope data should be cleaned up
        def check_node_cleanup(node):
            if node._children:
                # Check that no data is stored in internal nodes
                assert node.data is None or node.data == {}
                for child in node._children:
                    check_node_cleanup(child)
            else:
                # Leaf node should oly contain seed data
                assert node.data is not None
                assert node.data.keys() == {"seed"}

        check_node_cleanup(tree.root)

    def test_stats_recording(self):
        """Test that stats are recorded correctly when requested."""
        P = get_hypercube(2)
        A = get_braid_arrangement(2)

        tree, _ = build_partition_tree(A, P, record_stats=True)

        # Check that stats exist in leaf nodes
        def check_stats(node):
            assert hasattr(node, "n_candidates")
            for child in node.children:
                check_stats(child)

        check_stats(tree.root)
