"""Tests for statistics functionality."""

import pytest

from polypart import build_incenu_tree, build_partition_tree
from polypart.core.tree import PartitionNode, PartitionTree
from polypart.experiments.stats import (
    _has_recorded_stats,
    get_ppart_stats,
    print_ppart_stats,
)
from polypart.generators.arrangements import get_braid_arrangement
from polypart.generators.polytopes import get_hypercube


class TestStatsRecording:
    """Test that stats are properly recorded when record_stats=True and not otherwise."""

    def test_ppart_with_record_stats_has_attributes(self):
        """When record_stats=True, nodes should have n_candidates, n_inequalities, n_vertices."""
        P = get_hypercube(3)
        A = get_braid_arrangement(3)
        tree, n_chambers = build_partition_tree(A, P, record_stats=True)

        # Check root node has stats
        assert hasattr(tree.root, "n_candidates")
        assert hasattr(tree.root, "n_inequalities")
        assert hasattr(tree.root, "n_vertices")

        # Check that stats are sensible values
        assert tree.root.n_candidates >= 0
        assert tree.root.n_inequalities >= 0
        assert tree.root.n_vertices >= 0

    def test_ppart_without_record_stats_no_attributes(self):
        """When record_stats=False, nodes should NOT have n_candidates etc."""
        P = get_hypercube(3)
        A = get_braid_arrangement(3)
        tree, n_chambers = build_partition_tree(A, P, record_stats=False)

        # Check root node does NOT have stats
        assert not hasattr(tree.root, "n_candidates")
        assert not hasattr(tree.root, "n_inequalities")
        assert not hasattr(tree.root, "n_vertices")

    def test_has_recorded_stats_true(self):
        """_has_recorded_stats returns True for trees with stats."""
        P = get_hypercube(3)
        A = get_braid_arrangement(3)
        tree, _ = build_partition_tree(A, P, record_stats=True)

        assert _has_recorded_stats(tree) is True

    def test_has_recorded_stats_false(self):
        """_has_recorded_stats returns False for trees without stats."""
        P = get_hypercube(3)
        A = get_braid_arrangement(3)
        tree, _ = build_partition_tree(A, P, record_stats=False)

        assert _has_recorded_stats(tree) is False

    def test_incenu_tree_no_stats(self):
        """IncEnu trees don't have recorded stats (no record_stats option)."""
        P = get_hypercube(3)
        A = get_braid_arrangement(3)
        tree, _ = build_incenu_tree(A, P)

        assert _has_recorded_stats(tree) is False


class TestGetPpartStats:
    """Test get_ppart_stats function."""

    def test_with_stats_returns_detailed_stats(self):
        """With record_stats=True, get_ppart_stats returns detailed statistics."""
        P = get_hypercube(3)
        A = get_braid_arrangement(3)
        tree, n_chambers = build_partition_tree(A, P, record_stats=True)

        stats = get_ppart_stats(tree)

        # Basic stats always present
        assert "total_nodes" in stats
        assert "leaf_count" in stats
        assert "avg_depth" in stats
        assert "max_depth" in stats
        assert stats["has_detailed_stats"] is True

        # Detailed stats present when record_stats=True
        assert "avg_candidates" in stats
        assert "avg_inequalities" in stats
        assert "avg_vertices" in stats

        # Verify leaf count matches n_chambers
        assert stats["leaf_count"] == n_chambers

    def test_without_stats_returns_basic_stats(self):
        """Without record_stats, get_ppart_stats returns only basic statistics."""
        P = get_hypercube(3)
        A = get_braid_arrangement(3)
        tree, n_chambers = build_partition_tree(A, P, record_stats=False)

        stats = get_ppart_stats(tree)

        # Basic stats always present
        assert "total_nodes" in stats
        assert "leaf_count" in stats
        assert "avg_depth" in stats
        assert "max_depth" in stats
        assert stats["has_detailed_stats"] is False

        # Detailed stats NOT present
        assert "avg_candidates" not in stats
        assert "avg_inequalities" not in stats
        assert "avg_vertices" not in stats

        # Verify leaf count matches n_chambers
        assert stats["leaf_count"] == n_chambers

    def test_per_depth_stats_included(self):
        """Per-depth stats are included when requested."""
        P = get_hypercube(3)
        A = get_braid_arrangement(3)
        tree, _ = build_partition_tree(A, P, record_stats=True)

        stats = get_ppart_stats(tree, include_per_depth_stats=True)

        assert "per_depth_nodes" in stats
        assert "per_depth_avg_candidates" in stats
        assert "per_depth_avg_inequalities" in stats
        assert "per_depth_avg_vertices" in stats

    def test_per_depth_stats_excluded(self):
        """Per-depth stats are excluded when not requested."""
        P = get_hypercube(3)
        A = get_braid_arrangement(3)
        tree, _ = build_partition_tree(A, P, record_stats=True)

        stats = get_ppart_stats(tree, include_per_depth_stats=False)

        assert "per_depth_nodes" not in stats
        assert "per_depth_avg_candidates" not in stats

    def test_incenu_tree_basic_stats(self):
        """IncEnu trees can still get basic stats."""
        P = get_hypercube(3)
        A = get_braid_arrangement(3)
        tree, n_chambers = build_incenu_tree(A, P)

        stats = get_ppart_stats(tree)

        assert stats["has_detailed_stats"] is False
        assert stats["leaf_count"] == n_chambers
        assert "avg_candidates" not in stats


class TestPrintPpartStats:
    """Test print_ppart_stats function."""

    def test_print_with_stats(self, capsys):
        """print_ppart_stats works with recorded stats."""
        P = get_hypercube(3)
        A = get_braid_arrangement(3)
        tree, _ = build_partition_tree(A, P, record_stats=True)

        print_ppart_stats(tree)

        captured = capsys.readouterr()
        assert "total_nodes" in captured.out
        assert "avg_candidates" in captured.out

    def test_print_without_stats(self, capsys):
        """print_ppart_stats works without recorded stats (shows basic stats)."""
        P = get_hypercube(3)
        A = get_braid_arrangement(3)
        tree, _ = build_partition_tree(A, P, record_stats=False)

        print_ppart_stats(tree)

        captured = capsys.readouterr()
        assert "total_nodes" in captured.out
        assert "has_detailed_stats" in captured.out
        # avg_candidates should not be in output
        assert "avg_candidates" not in captured.out


class TestStatsPerformance:
    """Test that record_stats=False is more memory efficient."""

    def test_record_stats_false_smaller_memory(self):
        """Trees without stats should have smaller nodes."""
        P = get_hypercube(3)
        A = get_braid_arrangement(3)

        tree_with_stats, _ = build_partition_tree(A, P, record_stats=True)
        tree_without_stats, _ = build_partition_tree(A, P, record_stats=False)

        # Count attributes on root
        attrs_with = len(
            [a for a in dir(tree_with_stats.root) if not a.startswith("_")]
        )
        attrs_without = len(
            [a for a in dir(tree_without_stats.root) if not a.startswith("_")]
        )

        # Tree with stats should have more attributes
        assert attrs_with > attrs_without
