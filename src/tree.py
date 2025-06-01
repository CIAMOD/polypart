import numpy as np
from time import perf_counter
from pypoman import compute_polytope_vertices  # pypoman->cdd->double description method
import json
import os
import cdd
import cdd.gmp

try:
    from src.utils import get_planes
except:
    from utils import get_planes

OUT_FOLDER = "data/tree"


class Polytope:
    """
    Polytope in halfspace representation by Ax <= b.
    """

    def __init__(self, A: np.ndarray, b: np.ndarray):
        self.A = A
        self.b = b.reshape(b.shape[0], 1)  # Ensure b is a column vector
        self._vertices: np.ndarray | None = None
        self._dim = A.shape[1]

    @property
    def vertices(self):
        if self._vertices is None:
            raise ValueError("Vertices not computed yet.")
        return self._vertices

    def extreme(self):
        """
        Computes the vertices of the polytope.
        """
        mat = cdd.gmp.matrix_from_array(np.hstack([self.b, -self.A]))
        mat.rep_type = cdd.gmp.RepType.INEQUALITY
        polyhedron = cdd.gmp.polyhedron_from_matrix(mat)
        V = np.array(cdd.gmp.copy_generators(polyhedron).array)  # ndarray of Fraction
        assert np.all(
            V[:, 0] == 1
        ), "Inequalites do not represent a polytope. Vertices should start with 1."
        self._vertices = V[:, 1:]  # Exclude the first column which is 1 for vertices

    def is_degenerate(self):
        """
        Polytope is degenerate if it has less than dim + 1 vertices.
        """
        if self._vertices is None:
            raise ValueError("Vertices are needed to check degeneracy.")
        return self._vertices.shape[0] < self._dim + 1

    def add_halfspace(self, halfspace: tuple[np.array, int]):
        """
        Create a new polytope by adding a halfspace defined by a normal vector and an offset.
        Removes redundant rows from the resulting halfspace representation.
        """
        A = np.concatenate((self.A, halfspace[0][None, :]), axis=0)
        b = np.concatenate((self.b, np.array([halfspace[1]])[None, :]), axis=0)
        return Polytope(A, b)

    def remove_redundancies(self) -> "Polytope":
        """
        Removes redundant inequalities
        """
        mat = cdd.gmp.matrix_from_array(np.hstack([self.b.reshape(-1, 1), -self.A]))
        redundant_rows = list(cdd.gmp.redundant_rows(mat))
        self.A = np.delete(self.A, redundant_rows, axis=0)
        self.b = np.delete(self.b, redundant_rows, axis=0)
        return self

    def __repr__(self):
        return f"Polytope(A={self.A}, b={self.b})"


class TreeNode:
    def __init__(
        self,
        polytope: Polytope,
        candidate_hyperplanes: list,
        parent: "TreeNode" = None,
        depth: int = 0,
    ):
        self.polytope = polytope
        self.candidate_hyperplanes = candidate_hyperplanes
        self.parent = parent
        self.depth = depth

        self.children = []
        self.cut_hyperplane = None
        self._centroid = None
        self._alpha = None

    @property
    def centroid(self):
        if self._centroid is None:
            assert self.polytope is not None, f"Polytope not set."
            # assert len(self.polytope.vertices) > 0, f"Polytope has no vertices."
            self._centroid = np.mean(self.polytope.vertices, axis=0)
        return self._centroid

    def alpha_representative(self, n: int, r: int):
        """
        Compute the alpha-representative of the polytope.
        """
        if self._alpha is None:
            alpha_eps = self.centroid.reshape(n, r - 1)
            alpha = np.hstack((np.zeros((n, 1)), alpha_eps))
            self._alpha = alpha
        return self._alpha

    def add_child(
        self,
        child_polytope: Polytope,
        candidate_hyperplanes: list,
    ):
        # Create a new TreeNode for the child and add it to this node's children
        child_node = TreeNode(
            child_polytope,
            candidate_hyperplanes,
            parent=self,
            depth=self.depth + 1,
        )
        self.children.append(child_node)
        return child_node

    def set_cut(self, hyperplane: tuple[np.array, int]):
        self.cut_hyperplane = hyperplane

    def clean(self):
        self.polytope = None
        self.candidate_hyperplanes = None

    def classify(self, alpha: np.array):
        """
        Classify the given point recusively. Assume alpha is contained in the polytope.

        Parameters
        ----------
        alpha : np.array
            The point to classify (flattened alpha matrix).
        """
        if len(self.children) == 0:
            return self

        if alpha @ self.cut_hyperplane[0] <= self.cut_hyperplane[1]:
            return self.children[0].classify(alpha)
        else:
            return self.children[1].classify(alpha)

    def __repr__(self):
        return (
            f"""TreeNode(depth={self.depth}, """
            f"""is_leaf={len(self.children) == 0}, """
            f"""cut={self.cut_hyperplane})"""
        )


def get_simplex_inequalities(n: int, r: int):
    """
    Get the inequalities for the corner simplex in n dimensions.
    """

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


def cut_polytope_by_hyperplane(
    polytope: Polytope, hyperplane: tuple[np.array, int]
) -> tuple[Polytope, Polytope]:
    """
    Divide the region by the hyperplane. Return the two child polygons.
    """
    intersection = polytope.add_halfspace(hyperplane)
    intersection.extreme()

    c_hyperplane = (-hyperplane[0], -hyperplane[1])
    complement = polytope.add_halfspace(c_hyperplane)
    complement.extreme()
    return intersection, complement


def cut_polytope_by_hyperplane_fast(
    polytope: Polytope, hyperplane: tuple[np.array, int]
) -> tuple[Polytope, Polytope]:
    """
    Divide the region by the hyperplane. Return the two child polygons.
    """
    intersection = polytope.add_halfspace(hyperplane)
    intersection.extreme()

    c_hyperplane = (-hyperplane[0], -hyperplane[1])
    complement = polytope.add_halfspace(c_hyperplane)
    c_vertices = intersection.vertices[
        np.where(intersection.vertices @ hyperplane[0] == hyperplane[1])
    ]
    r_vertices = polytope.vertices[
        np.where(polytope.vertices @ hyperplane[0] > hyperplane[1])
    ]
    complement._vertices = np.concatenate((c_vertices, r_vertices), axis=0)
    return intersection, complement


def hyperplane_intersects_polytope(
    polytope: Polytope, hyperplane: tuple[np.array, int]
) -> bool:
    """
    Check if the hyperplane intersects the polytope.
    """
    values = polytope.vertices @ hyperplane[0]
    return np.any(values < hyperplane[1]) and np.any(values > hyperplane[1])


def hyperplanes_intersect_polytope(
    polytope: Polytope, hyperplanes: list[tuple[np.array, int]]
) -> np.ndarray[bool]:
    """
    Optimized version to check if multiple hyperplanes intersect the polytope.
    """
    W = np.array([h[0] for h in hyperplanes])
    b = np.array([h[1] for h in hyperplanes])
    values = polytope.vertices @ W.T
    return np.any(values < b, axis=0) & np.any(values > b, axis=0)


def get_best_split(
    polytope: Polytope, candidate_hyperplanes: list
) -> tuple[tuple, tuple[Polytope], list]:
    """
    Get the hyperplane that best splits the polytope, the two resulting polytopes and the remaining candidate hyperplanes.
    """
    if len(candidate_hyperplanes) == 0:
        return None, None, None

    # Get the hyperplanes that intersect the polytope
    hyp_intersects = hyperplanes_intersect_polytope(polytope, candidate_hyperplanes)
    new_candidate_indices = np.where(hyp_intersects)[0]

    if len(new_candidate_indices) == 0:
        return None, None, None

    # Split the polytope by the hyperplane
    b_iH = np.random.choice(new_candidate_indices)
    b_hyperplane = candidate_hyperplanes[b_iH]
    b_polys = cut_polytope_by_hyperplane_fast(polytope, b_hyperplane)

    # Remove used hyperplane from the list of candidate hyperplanes
    new_candidate_hyperplanes = [
        candidate_hyperplanes[i] for i in new_candidate_indices if i != b_iH
    ]

    return b_hyperplane, b_polys, new_candidate_hyperplanes


def build_tree(simplex: Polytope, candidate_hyperplanes: list) -> tuple[TreeNode, int]:
    """
    Build a tree of polytopes by recursively splitting a polytope using hyperplanes.
    """
    root = TreeNode(simplex, candidate_hyperplanes)
    queue = [root]

    prev_chambers = 0
    n_chambers: int = 0
    while queue:
        node = queue.pop()

        hyperplane, polys, new_candidate_hyperplanes = get_best_split(
            node.polytope, node.candidate_hyperplanes
        )

        if hyperplane is None:
            n_chambers += 1  # Count leaf nodes
            if prev_chambers != n_chambers and n_chambers % 1000 == 0:
                print(f"Found {n_chambers} chambers.")
                prev_chambers = n_chambers
            node.centroid
        else:
            node.set_cut(hyperplane)
            for poly in polys:
                child = node.add_child(poly, new_candidate_hyperplanes)
                queue.append(child)
            node.clean()

    return root, n_chambers


def compute_tree(n: int, r: int, verbose: bool = True) -> TreeNode:
    """
    Compute and save the tree of polytopes for a given n and r.
    """
    if verbose:
        print(f"Computing tree for n={n}, r={r}")
    start = perf_counter()
    simplex_inequalities = get_simplex_inequalities(n, r - 1)
    simplex = Polytope(*simplex_inequalities)
    simplex.extreme()

    planes = get_planes(n, r, 0, use_epsilons=True)
    candidate_hyperplanes = [(p[0], b) for p in planes for b in p[1]]

    root, n_chambers = build_tree(simplex, candidate_hyperplanes)
    if verbose:
        print(f"Found {n_chambers} chambers in {perf_counter() - start:.2f} s")
        print("Saving tree...")

    start_save = perf_counter()
    save_tree(root, f"tree_n{n}_r{r}.json")
    if verbose:
        print(f"Saved tree in {perf_counter() - start_save:.2f} s")

    return root


def compute_polytope_chambers(
    inequalities: np.ndarray, hyperplanes: list, verbose: bool = False
) -> list:
    """
    Compute the number of chambers in the polytope using a trivial algorithm.
    """
    if verbose:
        print(
            f"Computing chambers for dim={inequalities.shape[1]} and {len(hyperplanes)} hyperplanes..."
        )
    simplex = Polytope(*inequalities)
    simplex.extreme()

    if verbose:
        print(f"Computing chambers...")
    chambers = [simplex]
    for hyperplane in hyperplanes:
        new_chambers = []
        for chamber in chambers:
            if not hyperplane_intersects_polytope(chamber, hyperplane):
                new_chambers.append(chamber)
                continue
            poly1, poly2 = cut_polytope_by_hyperplane(chamber, hyperplane)

            new_chambers.append(poly1)
            new_chambers.append(poly2)
            assert (
                len(poly1.vertices) >= 4 and len(poly2.vertices) >= 4
            ), f"Chamber is empty! {len(poly1.vertices)} {len(poly2.vertices)}"

        chambers = new_chambers

    if verbose:
        print(f"Found {len(chambers)} chambers")
    return chambers


def save_tree(root: TreeNode, filename: str):
    """
    Save the tree structure to a JSON file. Only store minimal information to classify alphas.
    """
    tree_json = {
        "n_leaves": 0,
        "n_nodes": 0,
        "max_depth": 0,
        "avg_depth": 0,
        "tree": [],
    }

    queue = [root]
    while queue:
        node = queue.pop()
        node.index = len(tree_json["tree"])
        tree_json["tree"].append(
            {
                "depth": node.depth,
                "cut_hyperplane": (
                    str((node.cut_hyperplane[0].tolist(), node.cut_hyperplane[1]))
                    if node.cut_hyperplane is not None
                    else None
                ),
                "parent_idx": node.parent.index if node.parent is not None else None,
                "centroid": (
                    str(node.centroid.tolist()) if node._centroid is not None else None
                ),
            }
        )
        tree_json["n_nodes"] += 1
        if len(node.children) == 0:
            tree_json["n_leaves"] += 1
        tree_json["max_depth"] = max(tree_json["max_depth"], node.depth)
        tree_json["avg_depth"] += node.depth
        queue.extend(node.children)

    tree_json["avg_depth"] = round(tree_json["avg_depth"] / tree_json["n_nodes"], 2)

    if not os.path.exists(OUT_FOLDER):
        os.makedirs(OUT_FOLDER)

    out_path = os.path.join(OUT_FOLDER, filename)

    with open(out_path, "w") as f:
        json.dump(tree_json, f, indent=4)


def load_tree(filename: str) -> TreeNode:
    """
    Load a tree structure from a JSON file.
    """
    out_path = os.path.join(OUT_FOLDER, filename)

    with open(out_path, "r") as f:
        tree_json = json.load(f)

    nodes = [None] * len(tree_json["tree"])

    for i, node in enumerate(tree_json["tree"]):
        node["cut_hyperplane"] = (
            eval(node["cut_hyperplane"]) if node["cut_hyperplane"] is not None else None
        )
        node["centroid"] = (
            eval(node["centroid"]) if node["centroid"] is not None else None
        )
        nodes[i] = TreeNode(None, None, None, node["depth"])
        nodes[i].cut_hyperplane = (
            (
                np.array(node["cut_hyperplane"][0]),
                node["cut_hyperplane"][1],
            )
            if node["cut_hyperplane"] is not None
            else None
        )
        nodes[i]._centroid = np.array(node["centroid"])

        if node["parent_idx"] is not None:
            nodes[node["parent_idx"]].children.insert(0, nodes[i])
            nodes[i].parent = nodes[node["parent_idx"]]

    return nodes[0]


# @memory_profiler.profile
def main():
    n, r = 1, 7
    _ = compute_tree(n, r)


def test_load_tree(n: int, r: int):
    root = compute_tree(n, r)
    filename = f"test_tree_n{n}_r{r}.json"
    save_tree(root, filename)
    root2 = load_tree(filename)

    # Traverse the tree and compare the cut hyperplanes
    queue = [(root, root2)]
    while queue:
        node1, node2 = queue.pop()
        if node1.cut_hyperplane is None or node2.cut_hyperplane is None:
            assert (
                node1.cut_hyperplane is None and node2.cut_hyperplane is None
            ), f"Nodes differ in nodes {node1} and {node2}"
            continue
        assert np.allclose(
            node1.cut_hyperplane[0], node2.cut_hyperplane[0]
        ), f"Cut hyperplanes differ in nodes {node1} and {node2}"
        queue.extend(zip(node1.children, node2.children))

    # Try classifying a point
    alpha = np.random.rand(n * (r - 1))
    alpha.sort()

    node = root.classify(alpha)
    print(f"Classified alpha={alpha} in node {node}")

    # Try alpha outside the hypercube
    alpha = 1 + np.random.rand(n * (r - 1))

    node = root.classify(alpha)
    print(f"Classified alpha={alpha} in node {node}")


if __name__ == "__main__":
    main()
    # test_load_tree(3, 2)
