"""
Algorithms for computing isomorphism classes (orbits) and automorphisms (stabilizers)
of geometric chambers given a set of symmetries.
"""

from __future__ import annotations

import itertools
import time
from dataclasses import dataclass, field
from typing import (
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Set,
    Tuple,
    Union,
)

from tqdm import tqdm

from .ftyping import FractionVector
from .ppart import PartitionNode, PartitionTree

# Type alias for a transformation function: f(vector) -> vector
TransformationFunc = Callable[[FractionVector], FractionVector]
"""Function that takes a FractionVector and returns a FractionVector."""


@dataclass
class Orbit:
    """Represents an isomorphism class (connected component) of chambers.

    Attributes:
        id: Unique identifier for this class.
        representative: The chosen canonical chamber (node) for this class.
        members: Set of all chamber nodes belonging to this class.
        stabilizers: Names of provided transformations that map the
                     representative to itself.
    """

    id: int
    representative: PartitionNode
    members: Set[PartitionNode] = field(default_factory=set)
    stabilizers: List[str] = field(default_factory=list)

    @property
    def size(self) -> int:
        """Return the number of chambers in this class."""
        return len(self.members)

    def to_dict(self) -> Dict:
        """Serialize the orbit info (useful for JSON export)."""
        return {
            "class_id": self.id,
            "representative_centroid": (
                self.representative.centroid.tolist()
                if self.representative.centroid is not None
                else None
            ),
            "size": self.size,
            "stabilizers": self.stabilizers,
        }


class SymmetryGroup:
    """Manages the set of symmetry functions (isomorphisms)."""

    def __init__(
        self,
        transformations: Union[
            Dict[str, TransformationFunc],
            Iterable[TransformationFunc],
            Iterable[Tuple[str, TransformationFunc]],
        ],
    ):
        """Initialize with a list or dict of functions.

        Args:
            transformations: Can be:
                - Dict: {'name': func}
                - List of tuples: [('name', func)]
                - List of funcs: [func] (names will be auto-generated)
        """
        self._transforms: Dict[str, TransformationFunc] = {}

        if isinstance(transformations, dict):
            self._transforms = transformations
        elif isinstance(transformations, Iterable):
            # Check if it's a list of tuples or just funcs
            iterator = iter(transformations)
            try:
                first = next(iterator)
            except StopIteration:
                raise ValueError("No transformations provided.")
            else:
                # Reconstruct iterator
                full_iter = itertools.chain([first], iterator)
                if isinstance(first, tuple) and len(first) == 2:
                    self._transforms = dict(full_iter)  # type: ignore
                elif callable(first):
                    for i, func in enumerate(full_iter):
                        self._transforms[f"t_{i}"] = func  # type: ignore
                else:
                    raise ValueError(
                        "Transformations must be callables or (name, callable) tuples."
                    )

    def __iter__(self) -> Iterator[Tuple[str, TransformationFunc]]:
        """Yield (name, function) pairs."""
        return iter(self._transforms.items())

    def __len__(self) -> int:
        return len(self._transforms)


class ChamberClassifier:
    """Classifies chambers of a PartitionTree into isomorphism classes."""

    def __init__(
        self,
        tree: PartitionTree,
        symmetries: Union[SymmetryGroup, Iterable[TransformationFunc]],
    ):
        """
        Args:
            tree: The PartitionTree containing the chambers.
            symmetries: A SymmetryGroup instance or a raw list of TransformationFunc.
        """
        self.tree = tree
        if isinstance(symmetries, SymmetryGroup):
            self.symmetries = symmetries
        else:
            self.symmetries = SymmetryGroup(symmetries)

    def _get_leaves(self) -> List[PartitionNode]:
        """Retrieve all leaf nodes (chambers) from the tree."""
        leaves = []
        stack = [self.tree.root]
        while stack:
            node = stack.pop()
            if node._children:
                stack.extend(node._children)
            else:
                leaves.append(node)
        return leaves

    def compute_classes(self, verbose: bool = True) -> List[Orbit]:
        """
        Compute connected components (orbits) of chambers.

        Algorithm:
            1. Collect all chambers.
            2. Iterate through undefined chambers.
            3. For each new chamber, perform a BFS (Orbit Traversal)
               using the symmetry functions to find all connected chambers.

        Returns:
            List of Orbit objects.
        """
        chambers = self._get_leaves()

        visited: Set[PartitionNode] = set()
        orbits: List[Orbit] = []

        if verbose:
            print(f"Classifying {len(chambers)} chambers...")
            start_time = time.perf_counter()

        # Iterate over all chambers. If we find one not visited, it starts a new Orbit.
        iterator = tqdm(chambers) if verbose else chambers

        for seed_node in iterator:
            if seed_node in visited:
                continue

            # Start a new Orbit
            orbit_id = len(orbits)
            current_orbit = Orbit(id=orbit_id, representative=seed_node)

            print(f"Processing new orbit with representative: {seed_node.centroid}")

            visited.add(seed_node)
            current_orbit.members.add(seed_node)

            # Apply all symmetries
            for name, func in self.symmetries:
                print(f"Applying transformation: {name}, {func}")
                # 1. Apply map to centroid
                mapped_centroid = func(seed_node.centroid)
                print(f"Mapped centroid: {mapped_centroid}")

                # 2. Find which chamber this point lands in
                target_node = self.tree.classify(mapped_centroid)

                # 3. Process the result
                if target_node == seed_node:
                    print(f"Found stabilizer: {name}")
                    current_orbit.stabilizers.append(name)
                    continue

                print(f"Found iso with centroid: {target_node.centroid}")

                if target_node not in visited:
                    visited.add(target_node)
                    current_orbit.members.add(target_node)

            orbits.append(current_orbit)

        if verbose:
            elapsed = time.perf_counter() - start_time
            print(f"Found {len(orbits)} isomorphism classes in {elapsed:.2f}s")

        return orbits
