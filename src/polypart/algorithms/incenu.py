"""Incremental enumeration algorithm for hyperplane arrangements."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from polypart.core.geometry import (
    Arrangement,
    HalfSpace,
    Hyperplane,
    Polyhedron,
)
from polypart.core.tree import PartitionNode, PartitionTree
from polypart.core.typing import Fraction, FractionVector
from polypart.utils.solvers import IncEnuCDDBackend


def intersect_line(
    p: FractionVector,
    v: FractionVector,
    sign_vector: list[int],
    hyperplanes: Sequence[Hyperplane],
    support: Sequence[HalfSpace],
) -> FractionVector | None:
    """Compute intersection of line p + tv with closest hyperplane in direction v."""
    min_t: Fraction | None = None

    for i, sigma in enumerate(sign_vector):
        hp = hyperplanes[i] if sigma == 1 else -hyperplanes[i]
        denom = np.dot(hp.normal, v)
        if denom != 0:
            t = (hp.offset - np.dot(hp.normal, p)) / denom
            if t > 0 and (min_t is None or t < min_t):
                min_t = t

    for hp in support:
        denom = np.dot(hp.normal, v)
        if denom != 0:
            t = (hp.offset - np.dot(hp.normal, p)) / denom
            if t > 0 and (min_t is None or t < min_t):
                min_t = t

    if min_t is None:
        return None
    return p + min_t * v


def perturb_witness(
    witness: FractionVector,
    cut: Hyperplane,
    sign_vector: list[int],
    hyperplanes: Sequence[Hyperplane],
    support: Sequence[HalfSpace],
) -> FractionVector:
    """Perturb witness if it lies on the cutting hyperplane."""
    if cut(witness) != 0:
        return witness

    intersection = intersect_line(
        witness, cut.normal, sign_vector, hyperplanes, support
    )

    if intersection is None:
        return witness + Fraction(1, 10) * cut.normal

    return (witness + intersection) / 2


def inc_enu(
    root: PartitionNode,
    hyperplanes: Sequence[Hyperplane],
    support: Sequence[HalfSpace],
    backend: IncEnuCDDBackend,
    verbose: bool = False,
) -> int:
    """DFS helper for incremental enumeration."""
    stack = [root]
    n_chambers = 0

    while stack:
        node = stack.pop()
        sign_vector = node.data["sign_vector"]
        witness = node.data["witness"]
        i = len(sign_vector)

        if i < len(hyperplanes):
            cut = hyperplanes[i]
            w_perturbed = perturb_witness(
                witness, cut, sign_vector, hyperplanes, support
            )
            sigma_w = 1 if cut(w_perturbed) > 0 else -1
            first_child = PartitionNode(
                data={
                    "sign_vector": sign_vector + [sigma_w],
                    "witness": w_perturbed,
                }
            )
            node.add_child(first_child)
            stack.append(first_child)

            # Check feasibility on opposite side
            sigma_w_minus = -sigma_w
            res = backend.solve(sign_vector + [sigma_w_minus], compute_x=True)
            opp_witness = res["x"] if res["interior"] else None

            if opp_witness is not None:
                node.cut = -cut if sigma_w == 1 else cut
                second_child = PartitionNode(
                    data={
                        "sign_vector": sign_vector + [sigma_w_minus],
                        "witness": opp_witness,
                    }
                )
                node.add_child(second_child)
                stack.append(second_child)

            # Discard data to save memory
            del node.data["sign_vector"]
            del node.data["witness"]
        else:
            # Leaf node: store witness as seed (canonical name) before clearing
            n_chambers += 1
            if verbose and n_chambers % 1000 == 0:
                print(f"Found {n_chambers} chambers...")
            # Store witness as seed
            node.data["seed"] = witness
            del node.data["witness"]

    return n_chambers


def initial_witness(backend: IncEnuCDDBackend) -> FractionVector:
    """Compute an initial witness point inside the support polyhedron."""
    result = backend.solve([], compute_x=True)
    if not result["interior"]:
        raise RuntimeError("Support is empty or has no interior.")
    return result["x"]


def build_incenu_tree(
    hyperplanes: Arrangement | Sequence[Hyperplane],
    support: Polyhedron | Sequence[HalfSpace] | None = None,
    verbose: bool = False,
) -> tuple[PartitionTree, int]:
    """Build partition tree via incremental enumeration.

    Args:
        hyperplanes: Arrangement or sequence of hyperplanes defining the arrangement.
        support: Bounding polyhedron or sequence of halfspace constraints.
            Unlike ppart, incenu does not require a bounded polytope.
        verbose: Print progress messages.

    Returns:
        Tuple of (partition tree, number of chambers).
    """
    # Normalize hyperplanes to a list
    if isinstance(hyperplanes, Arrangement):
        hyperplanes_list = hyperplanes.as_list()
    else:
        hyperplanes_list = list(hyperplanes)

    # Normalize support to a list of HalfSpaces
    if support is None:
        support_list: list[HalfSpace] = []
    elif isinstance(support, Polyhedron):
        support_list = support.halfspaces
    else:
        support_list = list(support)

    backend = IncEnuCDDBackend(hyperplanes_list, support_list)

    if not hyperplanes_list:
        root = PartitionNode(data={"sign_vector": [], "seed": initial_witness(backend)})
        return PartitionTree(root), 1

    root = PartitionNode(data={"sign_vector": [], "witness": initial_witness(backend)})
    n_partitions = inc_enu(root, hyperplanes_list, support_list, backend, verbose)

    return PartitionTree(root), n_partitions
