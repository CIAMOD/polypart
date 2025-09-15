import numpy as np
import itertools
import math
from polypart.geometry import Polytope, Hyperplane
from polypart.ppart import build_partition_tree
from polypart.ftyping import as_fraction_vector
from polypart.io import save_tree
from time import perf_counter
from fractions import Fraction


def generate_admissible_matrices_fixed_r_prime(
    n: int, r: int, r_prime: int, remove_even_symetry: bool = False
):
    """
    Generate all possible admissible matrices of shape (n, r) with a fixed
    sum of the first row.

    Parameters
    ----------
    n : int
        Number of points
    r : int
        Rank of the matrix
    r_prime : int
        Sum of the first row of the admissible matrix

    Returns
    -------
    N_rp : Iterator[np.ndarray]
        All (r choose r')^n admissible matrices of shape (n, r) for a fixed
        r'.
    """

    def generate_admissible_matrices_fixed_r_prime(
        n: int, r: int, r_prime: int, remove_even_symetry: bool = False
    ):
        """Generate admissible matrices with fixed r' (first-row sum).

        Args:
            n: number of points.
            r: rank of the matrix.
            r_prime: desired sum of the first row.
            remove_even_symetry: if True, cut symmetric duplicates.

        Yields:
            Admissible matrices as numpy arrays of shape (n, r).
        """

    # Generate all r choose r' combinations
    combs = itertools.combinations(range(r), r_prime)

    # Generate all possible variations with repetitions of the combinations
    variations = itertools.product(combs, repeat=n)

    N = math.comb(r, r_prime) ** n

    # Fill the admissible matrices
    for i, variation in enumerate(variations):
        if remove_even_symetry and r == 2 * r_prime and i >= N // 2:
            return
        # Generate the admissible matrix from the variation
        n_ = np.zeros((n, r), dtype=int)
        for i in range(n):
            for j in variation[i]:
                n_[i, j] = 1

        # Yield the admissible matrix
        yield n_


def get_plane_intercept_bounds(w: np.ndarray):
    r"""Bounds for intersecting with the simplex. Closed interval (l(n_), u(n_)).

    For every vertex of the simplex ([0,0,...,1,1...,1])
    w_\bar{n_}=r'-r*n_, where \bar{n_} \in \Omega_{n,r,r'}
    """
    # Reverse v in dim 1
    w = w[:, ::-1]
    # Cumsom along dim 1
    cumsums = np.cumsum(w, axis=1)
    # Add column of zeros
    cumsums = np.hstack((np.zeros_like(cumsums[:, :1]), cumsums))
    # Get the lower and upper bounds
    lower_bound = cumsums.min(axis=1).sum()
    upper_bound = cumsums.max(axis=1).sum()

    return lower_bound, upper_bound


def get_planes(
    n: int, r: int, d: int, use_epsilons=False
) -> list[tuple[np.ndarray, list[int]]]:
    """
    Find all planes that intersect with the simplex.
    a_11*e_11 + a_12*e_12 + ... + a_nr*e_nr = intercept

    Returns:
    --------
    tuple : (v, ks2)
        List of tuples[np.ndarray, list[int]] where v
        is the normal vector and ks2 is the
        list of intercepts for which the plane intersects the simplex.
    """
    # print(f"n={n}, r={r}, d={d}")
    planes = []
    for r_prime in range(1, r // 2 + 1):  # Remove the odd symmetry
        # print(f"r_prime={r_prime}")
        new_planes = []
        for n_ in generate_admissible_matrices_fixed_r_prime(
            n, r, r_prime, True  # Remove even symetry
        ):
            if use_epsilons:
                n_ = n_[:, 1:]

            # Normal vector
            v = r_prime - r * n_.flatten()
            lower, upper = get_plane_intercept_bounds(r_prime - r * n_)
            ks2 = [kp for kp in range(lower + 1, upper) if (kp + r_prime * d) % r == 0]

            if len(ks2) > 0:
                new_planes.append((v, ks2))

        planes += new_planes

    return planes


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


def compute_tree(n: int, r: int, verbose: bool = True) -> None:
    """
    Compute and save the tree of polytopes for a given n and r.
    This is a use case of the algorithm for a math problem,
    with a specified initial polytope (simplex) and a set of candidate hyperplanes.
    """
    if verbose:
        print(f"Computing tree for n={n}, r={r}")
    start = perf_counter()

    # Create the initial polytope (simplex)
    simplex_inequalities = get_simplex_inequalities(n, r - 1)
    simplex = Polytope(*simplex_inequalities)
    simplex.extreme()
    if verbose:
        print(f"  Simplex:\t\t{simplex}")

    # Generate candidate hyperplanes
    planes = get_planes(n, r, 0, use_epsilons=True)
    candidate_hyperplanes = [
        Hyperplane(as_fraction_vector(p[0]), Fraction(b)) for p in planes for b in p[1]
    ]
    if verbose:
        print(f"  # of hyperplanes:\t{len(planes)}")

    # Build the partition tree
    tree, n_chambers = build_partition_tree(simplex, candidate_hyperplanes)

    save_tree(tree, f"data/moduli_n{n}_r{r}.json")

    if verbose:
        print(f"Found {n_chambers} chambers in {perf_counter() - start:.2f} s")


# Example usage for n=1, r=7
if __name__ == "__main__":
    compute_tree(n=1, r=7, verbose=True)
