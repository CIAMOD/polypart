from polypart import Polytope, Hyperplane, build_partition_tree
import numpy as np


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


# Define a function to take in a polytope and return m number of hyperplanes
# intersecting it. For that, generate 3 points uniformly at random from the polytope,
# and then return the hyperplane defined by these 3 points.


def sample_point_in_polytope(polytope: Polytope) -> np.ndarray:
    """Sample a point uniformly at random from the polytope using rejection sampling."""
    # Get bounding box of the polytope
    mins = np.min(polytope.vertices, axis=0).astype(float)
    maxs = np.max(polytope.vertices, axis=0).astype(float)

    while True:
        # Sample a random point in the bounding box
        point = np.random.uniform(mins, maxs)
        if polytope.contains(point):
            return point


def get_intersecting_hyperplanes(polytope: Polytope, m: int) -> list[Hyperplane]:
    hyperplanes = []
    dim = polytope.A.shape[1]
    for _ in range(m):
        points = np.array([sample_point_in_polytope(polytope) for _ in range(dim)])
        if dim == 2:
            p1, p2 = points[0], points[1]
            normal = np.array([p2[1] - p1[1], p1[0] - p2[0]])
            offset = normal @ p1
            hyperplanes.append(Hyperplane(normal, offset))
        elif dim == 3:
            p1, p2, p3 = points[0], points[1], points[2]
            normal = np.cross(p2 - p1, p3 - p1)
            offset = normal @ p1
            hyperplanes.append(Hyperplane(normal, offset))
        else:
            raise ValueError("Dimension not supported")
    return hyperplanes


if __name__ == "__main__":
    A, b = get_simplex_inequalities(n=1, r=3)
    # Print the inequalities one by one as a_1*x + a_2*y + a_3*z <= b
    for i in range(A.shape[0]):
        print(f"{A[i,0]}x+{A[i,1]}y+{A[i,2]}z <= {b[i]}")
    polytope = Polytope(A, b)
    polytope.extreme()
    print(
        f"Initial polytope has {len(polytope.vertices)} vertices and dim {polytope.A.shape[1]}."
    )

    m = 5  # number of hyperplanes to split by
    hyperplanes = get_intersecting_hyperplanes(polytope, m)
    # Print all the hyperplanes as a_1*x + a_2*y + a_3*z = b
    for i, h in enumerate(hyperplanes):
        print(
            f"Hyperplane {i}: {h.normal[0]}x + {h.normal[1]}y + {h.normal[2]}z = {h.offset}"
        )
    print(f"Generated {len(hyperplanes)} hyperplanes intersecting the polytope.")

    tree = build_partition_tree(polytope, hyperplanes)
    leaves = tree.get_leaves()
    print(f"Partitioned into {len(leaves)} polytopes.")
    for i, leaf in enumerate(leaves):
        print(f"Polytope {i} has {len(leaf.vertices)} vertices.")
