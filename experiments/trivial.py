from polypart import Polytope, Hyperplane, build_partition_tree
import numpy as np
from time import perf_counter
from jubound import get_intersecting_hyperplanes
import matplotlib.pyplot as plt
from moduli import *


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
    times_trivial = []
    times_tree_random = []
    times_tree_ventropy = []
    n_partitions_list = []

    for n_hyperplanes in n_hyperplanes_list:
        selected_hyperplanes = hyperplanes[:n_hyperplanes]

        start = perf_counter()
        partitions = get_partitions(polytope, selected_hyperplanes)
        end = perf_counter()
        times_trivial.append(end - start)

        start = perf_counter()
        _, n_partitions = build_partition_tree(polytope, selected_hyperplanes, "random")
        end = perf_counter()
        times_tree_random.append(end - start)

        start = perf_counter()
        _, n_partitions = build_partition_tree(
            polytope, selected_hyperplanes, "v-entropy"
        )
        end = perf_counter()
        times_tree_ventropy.append(end - start)

        n_partitions_list.append(len(partitions))

        print(
            f"n_hyperplanes: {n_hyperplanes}, n_partitions: {len(partitions)}, "
            f"partition_time: {times_trivial[-1]:.4f}s, tree_time: {times_tree_random[-1]:.4f}s"
            f", tree_ventropy_time: {times_tree_ventropy[-1]:.4f}s"
        )

    plt.figure(figsize=(10, 6))
    plt.plot(n_hyperplanes_list, times_trivial, label="Trivial Algorithm", marker="o")
    plt.plot(
        n_hyperplanes_list,
        times_tree_random,
        label="Tree Algorithm (Random)",
        marker="o",
    )
    plt.plot(
        n_hyperplanes_list,
        times_tree_ventropy,
        label="Tree Algorithm (V-Entropy)",
        marker="o",
    )
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
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    compare_algorithms(dim=4, min_n_hyperplanes=1, max_n_hyperplanes=30, step=5)
