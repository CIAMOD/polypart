"""Partition graph for classifying cells into equivalence classes under symmetry.

This module provides tools to reduce a PartitionTree to a quotient graph,
grouping cells (regions/chambers) into equivalence classes under the action
of a symmetry group.

Terminology:
    - Cell: A leaf region in the partition tree (also called chamber or partition)
    - Seed: A representative point inside a cell (centroid from ppart, witness from incenu)
    - Class: An equivalence class of cells under symmetry (also called orbit)
    - Stabilizer: A symmetry that maps a cell to itself (automorphism)
"""

from __future__ import annotations

import itertools
import time
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any

from tqdm import tqdm

from polypart.core.tree import PartitionNode, PartitionTree
from polypart.core.typing import FractionVector

TransformFunc = Callable[[FractionVector], FractionVector]


class StorageLevel(IntEnum):
    """How much information to store during reduction.

    Attributes:
        COUNT_ONLY: Only count classes, no storage beyond running queues.
        SEEDS: Store the seed (representative point) for each class.
        STABILIZERS: Store seeds plus stabilizer symmetries for each class.
        FULL: Store seeds, stabilizers, and mapping dict {transform: target_cell}.
    """

    COUNT_ONLY = 1
    SEEDS = 2
    STABILIZERS = 3
    FULL = 4


@dataclass
class EquivalenceClass:
    """An equivalence class of cells under symmetry group action.

    Also known as an orbit in group theory terminology.

    Attributes:
        id: Unique class identifier.
        seed_node: The representative cell (first discovered) for this class.
        members: All cells belonging to this class (only populated if storage >= SEEDS).
        stabilizers: Names of symmetries fixing the seed (only if storage >= STABILIZERS).
        mappings: Dict {transform_name: target_seed} showing orbit structure (only if FULL).
    """

    id: int
    seed_node: PartitionNode
    members: set[PartitionNode] = field(default_factory=set)
    stabilizers: list[str] = field(default_factory=list)
    mappings: dict[str, FractionVector] = field(default_factory=dict)

    @property
    def size(self) -> int:
        """Number of cells in this equivalence class."""
        return len(self.members) if self.members else 1

    @property
    def seed(self) -> FractionVector | None:
        """The representative point (seed) of this class."""
        if self.seed_node.data is None:
            return None
        return self.seed_node.data.get("seed")

    def to_dict(self) -> dict[str, Any]:
        """Serialize class for JSON export."""
        seed = self.seed
        return {
            "class_id": self.id,
            "seed": seed.tolist() if seed is not None else None,
            "size": self.size,
            "stabilizers": self.stabilizers,
            "mappings": {k: v.tolist() for k, v in self.mappings.items()}
            if self.mappings
            else None,
        }


class SymmetryGroup:
    """A group of symmetry transformations acting on points.

    The group consists of named transformations that map points (seeds)
    to other points, inducing an equivalence relation on cells.
    """

    def __init__(
        self,
        transforms: (
            dict[str, TransformFunc]
            | Iterable[TransformFunc]
            | Iterable[tuple[str, TransformFunc]]
        ),
    ) -> None:
        """Initialize symmetry group.

        Args:
            transforms: One of:
                - Dict mapping names to transformation functions
                - Iterable of (name, function) tuples
                - Iterable of callables (auto-named as sym_0, sym_1, ...)
        """
        self._transforms: dict[str, TransformFunc] = {}

        if isinstance(transforms, dict):
            self._transforms = transforms
        elif isinstance(transforms, Iterable):
            iterator = iter(transforms)
            try:
                first = next(iterator)
            except StopIteration:
                raise ValueError("No transformations provided.")

            full_iter = itertools.chain([first], iterator)
            if isinstance(first, tuple) and len(first) == 2:
                self._transforms = dict(full_iter)  # type: ignore[arg-type]
            elif callable(first):
                for i, func in enumerate(full_iter):
                    self._transforms[f"sym_{i}"] = func  # type: ignore[assignment]
            else:
                raise ValueError(
                    "Transforms must be callables or (name, callable) tuples."
                )

    def __iter__(self) -> Iterator[tuple[str, TransformFunc]]:
        """Yield (name, function) pairs."""
        return iter(self._transforms.items())

    def __len__(self) -> int:
        return len(self._transforms)

    def names(self) -> list[str]:
        """Return list of transformation names."""
        return list(self._transforms.keys())


class PartitionGraph:
    """Quotient graph of a partition tree under symmetry group action.

    Given a PartitionTree and a SymmetryGroup, this class computes
    equivalence classes (orbits) of cells, effectively building a
    quotient structure where symmetric cells are identified.

    Example:
        >>> graph = PartitionGraph(tree, symmetries)
        >>> n_classes = graph.reduce(storage=StorageLevel.STABILIZERS)
        >>> for cls in graph.classes:
        ...     print(f"Class {cls.id}: {cls.size} cells, stabilizers: {cls.stabilizers}")
    """

    def __init__(
        self,
        tree: PartitionTree,
        symmetries: SymmetryGroup | Iterable[TransformFunc],
    ) -> None:
        """Initialize partition graph.

        Args:
            tree: Partition tree containing cells (from ppart or incenu).
            symmetries: SymmetryGroup or iterable of transformation functions.
        """
        self.tree = tree
        if isinstance(symmetries, SymmetryGroup):
            self.symmetries = symmetries
        else:
            self.symmetries = SymmetryGroup(symmetries)

        self._classes: list[EquivalenceClass] = []
        self._n_classes: int = 0
        self._reduced: bool = False

    @property
    def classes(self) -> list[EquivalenceClass]:
        """List of equivalence classes after reduction."""
        if not self._reduced:
            raise RuntimeError("Call reduce() before accessing classes.")
        return self._classes

    @property
    def n_classes(self) -> int:
        """Number of equivalence classes."""
        return self._n_classes

    def _get_cells(self) -> list[PartitionNode]:
        """Retrieve all leaf nodes (cells) from the tree.

        Filters out empty nodes from incenu trees (nodes marked with 'empty': True).
        """
        cells = []
        stack = [self.tree.root]
        while stack:
            node = stack.pop()
            if node._children:
                stack.extend(node._children)
            else:
                cells.append(node)
        return cells

    def _get_seed(self, node: PartitionNode) -> FractionVector | None:
        """Get the seed point from a node."""
        if node.data is None:
            raise RuntimeError("Leaf node has no data. Corrupted tree.")

        seed = node.data.get("seed")
        if seed is not None:
            return seed

        raise RuntimeError("Leaf node has no seed point. Corrupted tree.")

    def reduce(
        self,
        storage: StorageLevel = StorageLevel.STABILIZERS,
        verbose: bool = False,
    ) -> int:
        """Compute equivalence classes of cells under symmetry action.

        Uses BFS from each unvisited cell, applying all symmetries
        to discover equivalent cells.

        Args:
            storage: How much information to store (see StorageLevel).
            verbose: Print progress messages.

        Returns:
            Number of equivalence classes found.
        """
        cells = self._get_cells()
        visited: set[PartitionNode] = set()
        self._classes = []
        self._n_classes = 0

        if verbose:
            print(
                f"Reducing {len(cells)} cells with {len(self.symmetries)} symmetries..."
            )
            start_time = time.perf_counter()

        iterator = tqdm(cells, desc="Computing classes") if verbose else cells

        for seed_node in iterator:
            if seed_node in visited:
                continue

            class_id = self._n_classes
            self._n_classes += 1

            # Minimal storage: just mark visited and count
            if storage == StorageLevel.COUNT_ONLY:
                visited.add(seed_node)
                seed_point = self._get_seed(seed_node)
                if seed_point is not None:
                    for _, func in self.symmetries:
                        mapped = func(seed_point)
                        target = self.tree.classify(mapped)
                        visited.add(target)
                continue

            # Create equivalence class with appropriate storage
            eq_class = EquivalenceClass(id=class_id, seed_node=seed_node)
            visited.add(seed_node)

            if storage >= StorageLevel.SEEDS:
                eq_class.members.add(seed_node)

            seed_point = self._get_seed(seed_node)

            for name, func in self.symmetries:
                mapped_point = func(seed_point)
                target_node = self.tree.classify(mapped_point)

                if target_node == seed_node:
                    # Stabilizer found
                    if storage >= StorageLevel.STABILIZERS:
                        eq_class.stabilizers.append(name)
                    if storage == StorageLevel.FULL:
                        eq_class.mappings[name] = mapped_point
                    continue

                if target_node not in visited:
                    visited.add(target_node)
                    if storage >= StorageLevel.SEEDS:
                        eq_class.members.add(target_node)
                    if storage == StorageLevel.FULL:
                        eq_class.mappings[name] = mapped_point

            self._classes.append(eq_class)

        self._reduced = True

        if verbose:
            elapsed = time.perf_counter() - start_time
            print(f"Found {self._n_classes} equivalence classes in {elapsed:.2f}s")

        return self._n_classes

    def to_dict(self) -> dict[str, Any]:
        """Serialize graph to dictionary."""
        if not self._reduced:
            raise RuntimeError("Call reduce() before serializing.")
        return {
            "n_classes": self._n_classes,
            "classes": [c.to_dict() for c in self._classes],
        }
