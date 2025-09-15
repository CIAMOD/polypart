import numpy as np
from fractions import Fraction
import matplotlib.pyplot as plt
import os

from polypart.geometry import Polytope, Hyperplane
from polypart.ftyping import as_fraction_vector
from polypart.ppart import build_partition_tree, PartitionTree
from polypart.io import save_tree

# Using seaborn's style
plt.style.use("seaborn-v0_8")


def plot_partitions(
    tree: PartitionTree, hyperplanes, path="images/square_partitions.png"
):
    """Plot random points colored by partition and hyperplanes.

    Args:
        tree: PartitionTree used to classify points.
        hyperplanes: list of Hyperplane objects to overlay.
        path: output image path.
    """

    # Generate random points
    n_points = 2000
    points = np.random.rand(n_points, 2)

    # Query the partition tree for each point
    partitions = [tree.classify(as_fraction_vector(point)) for point in points]

    # Plot the points colored by their partition
    fig, ax = plt.subplots(figsize=(8, 8))
    labels = np.array([p._id for p in partitions])
    unique_labels = np.unique(labels)
    for i, lab in enumerate(unique_labels):
        mask = labels == lab
        ax.scatter(
            points[mask, 0],
            points[mask, 1],
            label=f"Region {lab}",
        )

    # Plot the hyperplanes as dashed lines with labels
    x = np.linspace(0, 1, 100)
    for i, h in enumerate(hyperplanes):
        # convert Fraction-based coefficients to floats for plotting
        a = float(h.normal[0])
        b = float(h.normal[1])
        c = float(h.offset)
        if b != 0:
            y = (c - a * x) / b
            ax.plot(x, y, "k--", label=f"Hyperplane {i+1}")
        else:
            x_const = c / a
            ax.axvline(x=x_const, color="k", linestyle="--", label=f"Hyperplane {i+1}")
    # Place legend on the axis after plotting everything
    ax.legend(
        loc="upper left",
        frameon=True,
        fontsize=12,
        shadow=True,
        framealpha=0.9,
        bbox_to_anchor=(0.75, 0.95),
        handletextpad=0.5,
    )

    dirname = os.path.dirname(path) or "."
    os.makedirs(dirname, exist_ok=True)

    ax.set_aspect("equal")
    ax.set_xlim(-0.04, 1.04)
    ax.set_ylim(-0.04, 1.04)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Partitioned Unit Square")
    fig.savefig(path)
    plt.close(fig)


if __name__ == "__main__":
    # unit square
    A = [[-1, 0], [1, 0], [0, -1], [0, 1]]
    b = [0, 1, 0, 1]
    square = Polytope(A, b)
    square.extreme()

    h1 = Hyperplane.from_coefficients([1, 0, Fraction(1, 3)])
    h2 = Hyperplane.from_coefficients([0, 1, Fraction(1, 3)])

    tree, n_parts = build_partition_tree(square, [h1, h2])
    print(f"Number of partitions: {n_parts}")

    save_tree(tree, "data/square_partitions.json")
    plot_partitions(tree, [h1, h2], path="images/square_partitions.png")
