from polypart import Polytope, Hyperplane, build_partition_tree, save_tree
from polypart.ftyping import as_fraction_vector
import numpy as np
from math import gcd
from functools import reduce
from moduli import *


def sample_point_in_polytope(polytope: Polytope) -> np.ndarray:
    """Sample a point uniformly at random from the polytope using rejection sampling."""
    # Get bounding box of the polytope
    mins = np.min(polytope.vertices, axis=0).astype(float)
    maxs = np.max(polytope.vertices, axis=0).astype(float)

    while True:
        # Sample a random point in the bounding box
        point = np.random.uniform(mins, maxs)
        # Check if the point is in the polytope
        if polytope.contains(point):
            return point


def compute_hyperplane_from_points(
    points: np.ndarray,
) -> tuple[np.ndarray, float] | None:
    """
    Compute hyperplane from d points in d-dimensional space.
    Returns (normal, offset) or None if degenerate.
    """
    P = np.asarray(points, dtype=float)

    if P.ndim != 2 or P.shape[0] != P.shape[1]:
        raise ValueError("points must be a (d, d) array: d points in d-D space")

    p0 = P[0]
    V = P[1:] - p0  # (d-1, d); rows span the hyperplane
    if np.linalg.matrix_rank(V) < P.shape[1] - 1:
        return None

    _, _, vh = np.linalg.svd(V)
    normal = vh[-1]
    norm = np.linalg.norm(normal)
    if norm == 0:
        return None

    normal /= norm
    offset = float(normal @ p0)

    # Round normal and offset to get prettier numbers
    normal = np.round(normal, decimals=1)
    offset = np.round(offset, decimals=1)

    return normal, offset


def get_intersecting_hyperplanes(polytope: Polytope, m: int) -> list[Hyperplane]:
    hyperplanes = []
    dim = polytope.A.shape[1]
    while len(hyperplanes) < m:
        points = np.array([sample_point_in_polytope(polytope) for _ in range(dim)])
        result = compute_hyperplane_from_points(points)
        if result is None:
            print("Degenerate points, resampling...")
            continue  # Degenerate points, resample

        normal, offset = result
        hyperplane = Hyperplane.from_coefficients(np.append(normal, offset))

        if not polytope.intersecting_hyperplanes([hyperplane])[0]:
            continue  # Does not intersect, resample

        hyperplanes.append(hyperplane)

    # Find gcd of all coefficients denominators and multiply all coefficients by it
    for i, h in enumerate(hyperplanes):
        coeffs = np.append(h.normal, h.offset)
        denominators = [frac.denominator for frac in coeffs if frac != 0]
        common_denom = reduce(gcd, denominators)
        new_coeffs = coeffs * common_denom
        hyperplanes[i] = Hyperplane.from_coefficients(new_coeffs)

    return hyperplanes


if __name__ == "__main__":
    # np.random.seed(1)
    n, r = 1, 6
    A, b = get_simplex_inequalities(n=n, r=r, use_epsilons=True)
    # Print the inequalities one by one as a_1*x + a_2*y + a_3*z <= b
    for i in range(A.shape[0]):
        print(f"{A[i,0]}x+{A[i,1]}y+{A[i,2]}z <= {b[i]}")

    print(f"Simplex with {A.shape[0]} inequalities in {A.shape[1]} dimensions.")

    polytope = Polytope(A, b)
    polytope.extreme()
    print(
        f"Initial polytope has {len(polytope.vertices)} vertices and dim {polytope.A.shape[1]}."
    )

    if True:
        m = 21  # number of hyperplanes to split by
        hyperplanes = get_intersecting_hyperplanes(polytope, m)
        # Print all the hyperplanes as a_1*x + a_2*y + a_3*z = b
        for i, h in enumerate(hyperplanes):
            print(
                f"Hyperplane {i}: {h.normal[0]}x + {h.normal[1]}y + {h.normal[2]}z = {h.offset}"
            )
        print(f"Generated {len(hyperplanes)} hyperplanes intersecting the polytope.")
    else:
        planes = get_planes(n, r, 0, use_epsilons=True)
        print(f"Using {len(planes)} planes.")
        hyperplanes = [
            Hyperplane.from_coefficients(np.append(v, k))
            for v, ks in planes
            for k in ks
        ]

    tree, n_partitions = build_partition_tree(polytope, hyperplanes, "random")
    print(f"Polytope partitioned into {n_partitions} regions.")

    save_tree(tree, "experiments/jubound_tree.json")
