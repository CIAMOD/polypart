import numpy as np
from time import perf_counter
import multiprocessing as mp
from pypoman import (
    compute_polytope_vertices,
)  # pypoman->cdd->double description method (https://link.springer.com/chapter/10.1007/3-540-61576-8_77)
import ctypes
import json
import memory_profiler
import cProfile

try:
    from src.utils import get_planes
except:
    from utils import get_planes


class Polytope:
    def __init__(self, A: np.ndarray, b: np.array):
        self.A = A
        self.b = b
        self._vertices: list[np.array] = None

    @property
    def vertices(self):
        if self._vertices is None:
            raise ValueError("Vertices not computed yet.")
        return self._vertices

    def extreme(self):
        """
        Computes the vertices of the polytope.
        """
        self._vertices = compute_polytope_vertices(self.A, self.b)

    def add_halfspace(self, hyperplane: tuple[np.array, int]):
        A = np.concatenate((self.A, hyperplane[0][None, :]), axis=0)
        b = np.concatenate((self.b, [hyperplane[1]]), axis=0)
        return Polytope(A, b)

    def is_degenerate(self) -> bool:
        if len(self.vertices) <= self.A.shape[1]:
            return True

        V = np.array(self.vertices)
        V -= V[0]
        return np.linalg.matrix_rank(V) < self.A.shape[1]

    def serialize(self):
        return self.A.tolist(), self.b.tolist(), [v.tolist() for v in self.vertices]

    @classmethod
    def deserialize(cls, data):
        A, b, vertices = data
        p = cls(np.array(A), np.array(b))
        p._vertices = [np.array(v) for v in vertices]
        return p

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
        self.centroid
        self._alpha = None
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


def estimate_volume(polytope: Polytope, nsamples: int = 1000000):
    """
    Estimate the volume of a polytope using Monte Carlo sampling.
    """

    # Find the bounding box of the polytope
    min_coords = np.min(polytope.vertices, axis=0)
    max_coords = np.max(polytope.vertices, axis=0)
    ranges = max_coords - min_coords

    # Generate random samples within the bounding box
    samples = np.random.rand(nsamples, polytope.A.shape[1]) * ranges + min_coords

    # Check how many samples are inside the polytope. (Ax <= b)
    n_inside = np.sum(np.all(polytope.A @ samples.T <= polytope.b[:, None], axis=0))

    # Estimate the volume of the polytope
    volume = n_inside / nsamples * np.prod(ranges)

    return volume


def estimate_entropy(
    polytope: Polytope, hyperplane: tuple[np.array, int], nsamples: int = 1000
):
    """
    Estimate the entropy of a polytope using Monte Carlo sampling.
    """
    vertices = np.array(polytope.vertices)

    # Generate random samples within the bounding box
    weights = np.random.random((nsamples, vertices.shape[0]))
    weights /= np.sum(weights, axis=1, keepdims=True)

    # Use isometric coordinates to generate samples in the polytope
    samples = weights @ vertices

    # Double check if all samples are truly inside polygon
    n_inside = np.sum(np.all(polytope.A @ samples.T <= polytope.b[:, None], axis=0))

    assert (
        nsamples == n_inside
    ), f"Some samples not in polygon, n_inside={n_inside}, nsamples={nsamples}, samples={samples}, vertices={polytope.vertices}, tv={compute_polytope_vertices(polytope.A, polytope.b)}"

    # Check how many samples are one one side of the hyperplane
    n_left = np.sum(samples @ hyperplane[0] <= hyperplane[1])

    # Estimate the entropy of the polytope
    gini2 = n_left * (nsamples - n_left) / nsamples**2

    return gini2


def cut_polytope_by_hyperplane(
    polytope: Polytope, hyperplane: tuple[np.array, int]
) -> tuple[Polytope, Polytope, bool]:
    """
    Divide the region by the hyperplane. Return the two child polygons and whether the intersection is empty.
    """
    intersection = polytope.add_halfspace(hyperplane)
    intersection.extreme()
    if intersection.is_degenerate():
        return None, None, True

    hyperplane = (-hyperplane[0], -hyperplane[1])
    complement = polytope.add_halfspace(hyperplane)

    # REDUNDANT IF CHECKING IF INTERSECTION VERTICES ARE SAME AS PARENT OR NONE?
    # ------
    complement.extreme()
    if complement.is_degenerate():
        return None, None, True
    # -------

    return intersection, complement, False


def cut_polytope_by_hyperplane_fast(
    polytope: Polytope, hyperplane: tuple[np.array, int]
) -> tuple[Polytope, Polytope]:
    """
    Divide the region by the hyperplane. Return the two child polygons and whether the intersection is empty.
    """
    intersection = polytope.add_halfspace(hyperplane)
    intersection.extreme()

    c_hyperplane = (-hyperplane[0], -hyperplane[1])
    complement = polytope.add_halfspace(c_hyperplane)
    complement.extreme()

    return intersection, complement


def hyperplane_intersects_polytope(
    polytope: Polytope, hyperplane: tuple[np.array, int]
) -> bool:
    """
    Check if the hyperplane intersects the polytope.
    """
    v, b = hyperplane

    # Evaluate all vertices of the polytope
    vertices = np.array(polytope.vertices)
    values = vertices @ v

    # Check if any vertex is on the opposite side of the hyperplane
    tolerance = 1e-10
    return np.any(values < b - tolerance) and np.any(values > b + tolerance)


def get_best_split(polytope: Polytope, candidate_hyperplanes: list) -> tuple:
    """
    Get the hyperplane that best splits the polytope, the two resulting polytopes and the remaining candidate hyperplanes.
    """
    b_score = float("-inf")
    b_hyperplane: tuple[np.array, int] = None
    b_polys = None
    b_iH = None
    new_candidate_hyperplanes = []

    for iH, hyp in enumerate(candidate_hyperplanes):
        poly1, poly2, is_empty = cut_polytope_by_hyperplane(polytope, hyp)

        assert (not is_empty) == hyperplane_intersects_polytope(
            polytope, hyp
        ), f"Is empty={is_empty}, intersects={hyperplane_intersects_polytope(polytope, hyp)}, vertices={polytope.vertices}, hyp={hyp}"

        # Split the poly using the intersection line
        if not is_empty:
            gini2 = np.random.rand()
            new_candidate_hyperplanes.append(iH)
        else:
            gini2 = 0

        if gini2 > b_score:
            b_score = gini2
            b_hyperplane = hyp
            b_polys = (poly1, poly2)
            b_iH = iH

    if len(new_candidate_hyperplanes) == 0:
        return None, None, None

    # Remove best hyperplane from the list of candidate hyperplanes
    new_candidate_hyperplanes.remove(b_iH)

    new_candidate_hyperplanes = [
        candidate_hyperplanes[i] for i in new_candidate_hyperplanes
    ]

    return b_hyperplane, b_polys, new_candidate_hyperplanes


def get_best_split_fast(polytope: Polytope, candidate_hyperplanes: list) -> tuple:
    """
    Get the hyperplane that best splits the polytope, the two resulting polytopes and the remaining candidate hyperplanes.
    """
    new_candidate_hyperplanes = []

    for iH, hyp in enumerate(candidate_hyperplanes):
        if hyperplane_intersects_polytope(polytope, hyp):
            new_candidate_hyperplanes.append(iH)

    if len(new_candidate_hyperplanes) == 0:
        return None, None, None

    # Pick a random hyperplane
    b_iH = np.random.choice(new_candidate_hyperplanes)
    b_hyperplane = candidate_hyperplanes[b_iH]
    b_polys = cut_polytope_by_hyperplane_fast(polytope, b_hyperplane)

    # Remove best hyperplane from the list of candidate hyperplanes
    new_candidate_hyperplanes.remove(b_iH)

    new_candidate_hyperplanes = [
        candidate_hyperplanes[i] for i in new_candidate_hyperplanes
    ]

    return b_hyperplane, b_polys, new_candidate_hyperplanes


def build_tree(simplex: Polytope, candidate_hyperplanes: list) -> tuple[TreeNode, int]:
    """
    Build a tree of polytopes by recursively splitting a polytope using hyperplanes.
    """
    root = TreeNode(simplex, candidate_hyperplanes)
    queue = [root]

    n_chambers: int = 0

    while queue:
        node = queue.pop()

        hyperplane, polys, new_candidate_hyperplanes = get_best_split_fast(
            node.polytope, node.candidate_hyperplanes
        )

        node.clean()

        if hyperplane is None:
            n_chambers += 1  # Count leaf nodes
            continue

        node.set_cut(hyperplane)

        for poly in polys:
            child = node.add_child(poly, new_candidate_hyperplanes)
            queue.append(child)

    return root, n_chambers


def worker(a_queue: mp.JoinableQueue, u_queue: mp.JoinableQueue):
    while True:
        resource = a_queue.get()

        node_id, s_polytope, s_candidate_hyperplanes = resource

        # Deserialize polytope and candidate hyperplanes
        poly = Polytope.deserialize(s_polytope)
        cand_hyprs = [(np.array(h[0]), h[1]) for h in s_candidate_hyperplanes]

        hyperplane, polys, new_cand_hyprs = get_best_split(poly, cand_hyprs)

        if hyperplane is None:
            u_queue.put((node_id, None, None, None))
        else:
            # Serialize hyperplane, polys and new_cand_hyprs
            s_hyperplane = (hyperplane[0].tolist(), hyperplane[1])
            s_polys = [p.serialize() for p in polys]
            s_new_cand_hyprs = [(h[0].tolist(), h[1]) for h in new_cand_hyprs]

            u_queue.put((node_id, s_hyperplane, s_polys, s_new_cand_hyprs))

        a_queue.task_done()


def build_tree_mp(
    simplex: Polytope,
    candidate_hyperplanes: list,
    num_workers: int,
) -> tuple[TreeNode, int]:
    """
    Build a tree of polytopes using a multiprocessing approach.
    """

    root = TreeNode(simplex, candidate_hyperplanes)

    a_queue = mp.JoinableQueue()  # Queue for attached nodes
    u_queue = mp.JoinableQueue()  # Queue for unattached nodes

    # Start worker processes
    workers = []
    for _ in range(num_workers):
        p = mp.Process(target=worker, args=(a_queue, u_queue))
        p.start()
        workers.append(p)

    # First task is the root node
    a_queue.put(
        (
            id(root),
            root.polytope.serialize(),
            [(h[0].tolist(), h[1]) for h in root.candidate_hyperplanes],
        )
    )

    n_chambers: int = 0
    while (
        not a_queue._unfinished_tasks._semlock._is_zero()
        or not u_queue._unfinished_tasks._semlock._is_zero()
    ):
        resource = u_queue.get(block=True)

        node_id, s_hyperplane, s_polys, s_new_cand_hyprs = resource

        if s_hyperplane is None:
            n_chambers += 1
            u_queue.task_done()
            continue

        node = ctypes.cast(node_id, ctypes.py_object).value

        # Deserialize new candidate hyperplanes and hyperplane
        hyperplane = (np.array(s_hyperplane[0]), s_hyperplane[1])
        new_cand_hyprs = [(np.array(h[0]), h[1]) for h in s_new_cand_hyprs]

        node.set_cut(hyperplane)

        for s_poly in s_polys:
            poly = Polytope.deserialize(s_poly)
            child = node.add_child(poly, new_cand_hyprs)
            a_queue.put(
                (
                    id(child),
                    s_poly,
                    s_new_cand_hyprs,
                )
            )

        u_queue.task_done()

    # Kill worker processes
    for w in workers:
        w.terminate()

    return root, n_chambers


def compute_tree(n: int, r: int, use_mp: bool = False, num_workers: int = 8):
    print(f"Computing tree for n={n}, r={r}")
    start = perf_counter()
    simplex_inequalities = get_simplex_inequalities(n, r - 1)
    simplex = Polytope(*simplex_inequalities)
    simplex.extreme()

    planes = get_planes(n, r, 0, use_epsilons=True)
    candidate_hyperplanes = [(p[0], b) for p in planes for b in p[1]]

    if use_mp:
        print(f"Using multiprocessing, num_workers={num_workers}")
        root, n_chambers = build_tree_mp(simplex, candidate_hyperplanes, num_workers)
    else:
        print("Using single process")
        root, n_chambers = build_tree(simplex, candidate_hyperplanes)

    start_save = perf_counter()
    save_tree(root, f"data/tree_n{n}_r{r}.json")
    print(f"Saved tree in {perf_counter() - start_save:.2f} s")

    print(f"Found {n_chambers} chambers in {perf_counter() - start:.2f} s")

    return root


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
                "centroid": str(node.centroid.tolist()),
            }
        )
        tree_json["n_nodes"] += 1
        if len(node.children) == 0:
            tree_json["n_leaves"] += 1
        tree_json["max_depth"] = max(tree_json["max_depth"], node.depth)
        tree_json["avg_depth"] += node.depth
        queue.extend(node.children)

    tree_json["avg_depth"] = round(tree_json["avg_depth"] / tree_json["n_nodes"], 2)

    with open(filename, "w") as f:
        json.dump(tree_json, f, indent=4)


def load_tree(filename: str) -> TreeNode:
    """
    Load a tree structure from a JSON file.
    """
    with open(filename, "r") as f:
        tree_json = json.load(f)

    nodes = [None] * len(tree_json["tree"])

    for i, node in enumerate(tree_json["tree"]):
        node["cut_hyperplane"] = (
            eval(node["cut_hyperplane"]) if node["cut_hyperplane"] is not None else None
        )
        node["centroid"] = eval(node["centroid"])
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


def compute_polytope_chambers(n: int, r: int):
    """
    Compute the number of chambers in the polytope using a trivial algorithm.
    """
    print(f"Computing chambers for n={n}, r={r}")
    start = perf_counter()
    simplex_inequalities = get_simplex_inequalities(n, r - 1)
    simplex = Polytope(*simplex_inequalities)
    simplex.extreme()

    planes = get_planes(n, r, 0, use_epsilons=True)
    candidate_hyperplanes = [(p[0], b) for p in planes for b in p[1]]

    print(f"Found {len(candidate_hyperplanes)} candidate hyperplanes")

    print(f"Computing chambers...")
    chambers = [simplex]
    for hyperplane in candidate_hyperplanes:
        new_chambers = []
        for chamber in chambers:
            if not hyperplane_intersects_polytope(chamber, hyperplane):
                new_chambers.append(chamber)
                continue
            poly1, poly2 = cut_polytope_by_hyperplane_fast(chamber, hyperplane)

            new_chambers.append(poly1)
            new_chambers.append(poly2)

        chambers = new_chambers

    print(f"Found {len(chambers)} chambers in {perf_counter() - start:.2f} s")
    return chambers


def classify_polytope(hyperplane: tuple[np.array, int], polytope: Polytope) -> int:
    """
    Return 0 if the polytope intersects the hyperplane, 1 if it is on one side and -1 if it is on the other side.
    """
    vertices = np.array(polytope.vertices)
    values = vertices @ hyperplane[0]

    # Check if any vertex is on the opposite side of the hyperplane
    tolerance = 1e-10
    min_value = np.min(values)
    max_value = np.max(values)

    if min_value + tolerance > hyperplane[1]:
        return 1
    elif max_value - tolerance < hyperplane[1]:
        return -1
    else:
        return 0


def split_chambers_by_hyperplane_family(
    chambers: list[Polytope], w: np.array, bs: list[int]
) -> list[Polytope]:
    """
    Split the chambers by the hyperplane family.
    """
    if len(chambers) == 0 or len(bs) == 0:
        return chambers

    new_chambers = []
    chambers_low = []
    chambers_high = []

    b = bs[len(bs) // 2]
    for chamber in chambers:
        classification = classify_polytope((w, b), chamber)
        if classification == 0:
            poly1, poly2 = cut_polytope_by_hyperplane_fast(chamber, (w, b))
            new_chambers.append(poly1)
            new_chambers.append(poly2)
        elif classification == -1:
            chambers_low.append(chamber)
        elif classification == 1:
            chambers_high.append(chamber)

    return (
        new_chambers
        + split_chambers_by_hyperplane_family(chambers_low, w, bs[: len(bs) // 2])
        + split_chambers_by_hyperplane_family(chambers_high, w, bs[len(bs) // 2 + 1 :])
    )


def compute_polytope_chambers_efficient(n: int, r: int):
    """
    Compute the number of chambers in the polytope using a trivial algorithm.
    """
    print(f"Computing chambers for n={n}, r={r}")
    start = perf_counter()
    simplex_inequalities = get_simplex_inequalities(n, r - 1)
    simplex = Polytope(*simplex_inequalities)
    simplex.extreme()

    planes = get_planes(n, r, 0, use_epsilons=True)

    print(f"Found {sum(len(bs) for _,bs in planes)} candidate hyperplanes")

    # print(f"Min number of intercepts: {min(len(bs) for _,bs in planes)}")
    # print(f"Max number of intercepts: {max(len(bs) for _,bs in planes)}")
    # input("Press enter to continue...")
    print(f"Computing chambers...")
    chambers = [simplex]
    for w, bs in planes:
        chambers = split_chambers_by_hyperplane_family(chambers, w, bs)
    print(f"Found {len(chambers)} chambers in {perf_counter() - start:.2f} s")
    return chambers


# @memory_profiler.profile
def main():
    n, r = 1, 7
    # _ = compute_tree(n, r, use_mp=False)
    # _ = compute_polytope_chambers(n, r)
    _ = compute_polytope_chambers_efficient(n, r)


def test_load_tree(n: int, r: int):
    root = compute_tree(n, r, use_mp=False)
    filename = f"data/test_tree_n{n}_r{r}.json"
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

    # cProfile.run("main()", sort="cumtime")

    # nrs = [(2, 7), (2, 4), (2, 3), (2, 2), (2, 2), (2, 2)]
    # for n, rs in enumerate(nrs, start=1):
    #     for r in range(rs[0], rs[1] + 1):
    #         compute_tree(n, r, use_mp=False)
