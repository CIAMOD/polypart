from polypart import Polytope, Hyperplane, build_partition_tree
import numpy as np
from time import perf_counter
from jubound import get_intersecting_hyperplanes
import itertools
import math
import matplotlib.pyplot as plt


def get_simplex_inequalities(n: int, r: int):
    A = np.zeros((n * (r + 1), n * r), dtype=int)
    b = np.zeros(n * (r + 1), dtype=int)
    for i in range(n):
        for j in range(r):
            A[i * (r + 1) + j, i * r + j] = -1
            A[i * (r + 1) + j + 1, i * r + j] = 1
    for i in range(n):
        b[i * (r + 1) : (i + 1) * (r + 1)] = 0
        b[i * (r + 1) + r] = 1
    return A, b


def generate_admissible_matrices_fixed_r_prime(
    n: int, r: int, r_prime: int, remove_even_symetry: bool = False
):
    combs = itertools.combinations(range(r), r_prime)
    variations = itertools.product(combs, repeat=n)
    N = math.comb(r, r_prime) ** n
    for i, variation in enumerate(variations):
        if remove_even_symetry and r == 2 * r_prime and i >= N // 2:
            return
        n_ = np.zeros((n, r), dtype=int)
        for ii in range(n):
            for jj in variation[ii]:
                n_[ii, jj] = 1
        yield n_


def get_plane_intercept_bounds(w: np.ndarray):
    w = w[:, ::-1]
    cumsums = np.cumsum(w, axis=1)
    cumsums = np.hstack((np.zeros_like(cumsums[:, :1]), cumsums))
    lower_bound = cumsums.min(axis=1).sum()
    upper_bound = cumsums.max(axis=1).sum()
    return lower_bound, upper_bound


def get_planes(n: int, r: int, d: int, use_epsilons=False):
    planes = []
    for r_prime in range(1, r // 2 + 1):
        new_planes = []
        for n_ in generate_admissible_matrices_fixed_r_prime(n, r, r_prime, True):
            if use_epsilons:
                n_ = n_[:, 1:]
            v = r_prime - r * n_.flatten()
            lower, upper = get_plane_intercept_bounds(r_prime - r * n_)
            ks2 = [kp for kp in range(lower + 1, upper) if (kp + r_prime * d) % r == 0]
            if len(ks2) > 0:
                new_planes.append((v, ks2))
        planes += new_planes
    return planes


def get_partitions(polytope: Polytope, hyperplanes: list[Hyperplane]):
    pool = [polytope]
    for h in hyperplanes:
        new_pool = []
        for p in pool:
            if p.intersecting_hyperplanes([h])[0]:
                left, right = p.split_by_hyperplane(h)
                new_pool.append(left)
                new_pool.append(right)
            else:
                new_pool.append(p)
        pool = new_pool
    return pool


def compare_algorithms(
    dim: int, min_n_hyperplanes: int, max_n_hyperplanes: int, step: int
):
    """
    Compare the performance of the two partitioning algorithms by varying the number
    of hyperplanes used to partition a fixed polytope of dimension `dim`.

    Plots the time taken by each algorithm as a function of the number of hyperplanes.
    """

    A, b = get_simplex_inequalities(n=1, r=dim)
    polytope = Polytope(A, b)
    polytope.extreme()
    print(
        f"Initial polytope has {len(polytope.vertices)} vertices and dim {polytope.A.shape[1]}."
    )

    hyperplanes = get_intersecting_hyperplanes(polytope, max_n_hyperplanes)
    print(f"Generated {len(hyperplanes)} hyperplanes intersecting the polytope.")

    n_hyperplanes_list = list(range(min_n_hyperplanes, max_n_hyperplanes + 1, step))
    times_partition = []
    times_tree = []
    n_partitions_list = []

    for n_hyperplanes in n_hyperplanes_list:
        selected_hyperplanes = hyperplanes[:n_hyperplanes]

        start = perf_counter()
        partitions = get_partitions(polytope, selected_hyperplanes)
        end = perf_counter()
        times_partition.append(end - start)

        start = perf_counter()
        _, n_partitions = build_partition_tree(polytope, selected_hyperplanes)
        end = perf_counter()
        times_tree.append(end - start)

        n_partitions_list.append(len(partitions))

        print(
            f"n_hyperplanes: {n_hyperplanes}, n_partitions: {len(partitions)}, "
            f"partition_time: {times_partition[-1]:.4f}s, tree_time: {times_tree[-1]:.4f}s"
        )

    plt.figure(figsize=(10, 6))
    plt.plot(n_hyperplanes_list, times_partition, label="Trivial Algorithm", marker="o")
    plt.plot(n_hyperplanes_list, times_tree, label="Tree Algorithm", marker="o")
    plt.xlabel("Number of Hyperplanes")
    plt.ylabel("Time (seconds)")
    plt.title(f"Performance Comparison of Partitioning Algorithms (dim={dim})")
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(
        n_hyperplanes_list, n_partitions_list, label="Number of Partitions", marker="o"
    )
    plt.xlabel("Number of Hyperplanes")
    plt.ylabel("Number of Partitions")
    plt.title(f"Number of Partitions vs Number of Hyperplanes (dim={dim})")
    plt.legend


if __name__ == "__main__":
    compare_algorithms(dim=5, min_n_hyperplanes=3, max_n_hyperplanes=21, step=3)
